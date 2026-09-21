# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+
from __future__ import annotations

from magnetron import Tensor, nn, dtype
from magnetron_models.models import SnapshotModule
from .config import VAEConfig

_EMPTY = nn.init.EmptyInitStrategy()


def _conv(in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> nn.Conv2D:
    return nn.Conv2D(in_channels, out_channels, kernel_size, padding=padding, weight_init=_EMPTY, bias_init=_EMPTY)


class ChannelRMSNorm(nn.Module):
    def __init__(self, dim: int, gamma_rank: int) -> None:
        super().__init__()
        self.dim = dim
        self.scale = dim**0.5
        self.gamma = nn.Parameter(Tensor.empty(dim, *([1] * (gamma_rank - 1))))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.cast(dtype.float32)
        norm = xf.sqr().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
        normalized = (xf * norm.rcp()).cast(x.dtype)
        return normalized * self.scale * self.gamma.reshape(1, self.dim, 1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = ChannelRMSNorm(in_dim, gamma_rank=4)
        self.conv1 = _conv(in_dim, out_dim, 3, padding=1)
        self.norm2 = ChannelRMSNorm(out_dim, gamma_rank=4)
        self.conv2 = _conv(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = _conv(in_dim, out_dim, 1) if in_dim != out_dim else None

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        x = self.conv1(self.norm1(x).silu())
        x = self.conv2(self.norm2(x).silu())
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = ChannelRMSNorm(dim, gamma_rank=3)
        self.to_qkv = _conv(dim, dim * 3, 1)
        self.proj = _conv(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        qkv = self.to_qkv(self.norm(x)).reshape(B, 3 * C, H * W).transpose(1, 2)
        q, k, v = (t.contiguous() for t in qkv.split(C, dim=-1))
        scores = Tensor.einsum('bqc,bkc->bqk', q, k) * (1.0 / C**0.5)
        out = Tensor.einsum('bqk,bkc->bqc', scores.softmax(dim=-1), v)
        out = self.proj(out.transpose(1, 2).reshape(B, C, H, W))
        return out + x


class Upsampler(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = _conv(dim, out_dim, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x.interpolate(scale_factor=2.0, mode='nearest'))


class DupUpShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        factor: int = factor_t * factor_s * factor_s
        if out_channels * factor % in_channels:
            raise ValueError(f'out_channels * factor ({out_channels * factor}) must be a multiple of in_channels ({in_channels})')
        self.repeats: int = out_channels * factor // in_channels

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.reshape(B, self.out_channels, ft, fs, fs, H, W).permute(0, 1, 2, 5, 3, 6, 4).contiguous()
        x = x.reshape(B, self.out_channels, ft, H * fs, W * fs)
        return x[:, :, ft - 1]


class UpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temporal_upsample: bool, up: bool) -> None:
        super().__init__()
        resnets: list[nn.Module] = []
        dim: int = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(ResidualBlock(dim, out_dim))
            dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsampler = Upsampler(out_dim, out_dim) if up else None
        self.avg_shortcut = DupUpShortcut(in_dim, out_dim, factor_t=2 if temporal_upsample else 1) if up else None

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(shortcut)
        return x


class MidBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResidualBlock(dim, dim), ResidualBlock(dim, dim)])
        self.attentions = nn.ModuleList([AttentionBlock(dim)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        dims: list[int] = [cfg.decoder_base_dim * m for m in (cfg.dim_mult[-1], *reversed(cfg.dim_mult))]
        self.conv_in = _conv(cfg.z_dim, dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0])
        up_blocks: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up: bool = i != len(cfg.dim_mult) - 1
            up_blocks.append(UpBlock(in_dim, out_dim, cfg.num_res_blocks, temporal_upsample=up and cfg.temporal_upsample[i], up=up))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.norm_out = ChannelRMSNorm(dims[-1], gamma_rank=4)
        self.conv_out = _conv(dims[-1], cfg.out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(self.norm_out(x).silu())


class QwenImageVAE(SnapshotModule):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.post_quant_conv = _conv(cfg.z_dim, cfg.z_dim, 1)
        self.decoder = Decoder(cfg)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.post_quant_conv(z)).clamp(-1.0, 1.0)

    def denormalize_latents(self, latents: Tensor) -> Tensor:
        mean = Tensor(list(self.cfg.latents_mean), dtype=dtype.float32).reshape(1, self.cfg.z_dim, 1, 1).cast(latents.dtype)
        std = Tensor(list(self.cfg.latents_std), dtype=dtype.float32).reshape(1, self.cfg.z_dim, 1, 1).cast(latents.dtype)
        return latents * std + mean

    def forward(self, z: Tensor) -> Tensor:
        return self.decode(z)
