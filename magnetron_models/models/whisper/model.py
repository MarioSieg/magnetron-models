# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import math
from collections.abc import Iterator

from magnetron import Tensor, context, dtype, nn
from magnetron.snapshot import deserialize

from magnetron_models.models import SnapshotModule, decode_config

from .config import Config


def _linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias, weight_init=nn.init.EmptyInitStrategy(), bias_init=nn.init.EmptyInitStrategy())


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        if x.dtype == dtype.float32:
            return super().forward(x)
        xf = x.cast(dtype.float32)
        mean = xf.mean(dim=-1, keepdim=True)
        xm = xf - mean
        y = xm * (xm.sqr().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (y.cast(x.dtype) * self.weight) + self.bias


def _layer_norm(dim: int) -> FP32LayerNorm:
    return FP32LayerNorm(dim, weight_init=nn.init.EmptyInitStrategy(), bias_init=nn.init.EmptyInitStrategy())


class KVLayerCache:
    def __init__(self, num_heads: int, max_seq_len: int, head_dim: int) -> None:
        self.k: Tensor = Tensor.zeros(1, num_heads, max_seq_len, head_dim)
        self.v: Tensor = Tensor.zeros(1, num_heads, max_seq_len, head_dim)
        self.pos: int = 0

    def append(self, curr_k: Tensor, curr_v: Tensor) -> tuple[Tensor, Tensor]:
        start: int = self.pos
        end: int = start + curr_k.shape[2]
        if end > self.k.shape[2]:
            raise RuntimeError(f'KV cache overflow: {end} > {self.k.shape[2]}')
        self.k[:, :, start:end, :] = curr_k
        self.v[:, :, start:end, :] = curr_v
        self.pos = end
        return self.k[:, :, :end, :], self.v[:, :, :end, :]


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = _linear(d_model, d_model)
        self.k_proj = _linear(d_model, d_model, bias=False)
        self.v_proj = _linear(d_model, d_model)
        self.out_proj = _linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._split_heads(self.k_proj(x)), self._split_heads(self.v_proj(x))

    def attend(self, x: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
        B, T, C = x.shape
        q = self._split_heads(self.q_proj(x))
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if causal and T > 1:
            S: int = k.shape[2]
            k_pos = Tensor.arange(S).reshape(1, -1)
            q_pos = Tensor.arange(start=S - T, stop=S).reshape(-1, 1)
            mask = Tensor.where(k_pos <= q_pos, 0.0, -1e4).cast(scores.dtype).reshape(1, 1, T, S)
            scores = scores + mask
        out = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

    def forward(self, x: Tensor) -> Tensor:
        k, v = self.project_kv(x)
        return self.attend(x, k, v)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.self_attn = Attention(cfg.d_model, cfg.encoder_attention_heads)
        self.self_attn_layer_norm = _layer_norm(cfg.d_model)
        self.fc1 = _linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = _linear(cfg.encoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = _layer_norm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(self.fc1(self.final_layer_norm(x)).gelu())


class Encoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.conv1 = nn.Conv1D(
            cfg.num_mel_bins, cfg.d_model, 3, padding=1, weight_init=nn.init.EmptyInitStrategy(), bias_init=nn.init.EmptyInitStrategy()
        )
        self.conv2 = nn.Conv1D(
            cfg.d_model, cfg.d_model, 3, stride=2, padding=1, weight_init=nn.init.EmptyInitStrategy(), bias_init=nn.init.EmptyInitStrategy()
        )
        self.embed_positions = nn.Embedding(cfg.max_source_positions, cfg.d_model, weight_init=nn.init.EmptyInitStrategy())
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.encoder_layers)])
        self.layer_norm = _layer_norm(cfg.d_model)

    def forward(self, mel: Tensor) -> Tensor:
        x = self.conv1(mel).gelu()
        x = self.conv2(x).gelu()
        x = x.transpose(1, 2)
        x = x + self.embed_positions.weight[: x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.self_attn = Attention(cfg.d_model, cfg.decoder_attention_heads)
        self.self_attn_layer_norm = _layer_norm(cfg.d_model)
        self.encoder_attn = Attention(cfg.d_model, cfg.decoder_attention_heads)
        self.encoder_attn_layer_norm = _layer_norm(cfg.d_model)
        self.fc1 = _linear(cfg.d_model, cfg.decoder_ffn_dim)
        self.fc2 = _linear(cfg.decoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = _layer_norm(cfg.d_model)

    def forward(self, x: Tensor, cache: KVLayerCache, cross_kv: tuple[Tensor, Tensor]) -> Tensor:
        h = self.self_attn_layer_norm(x)
        k, v = cache.append(*self.self_attn.project_kv(h))
        x = x + self.self_attn.attend(h, k, v, causal=True)
        x = x + self.encoder_attn.attend(self.encoder_attn_layer_norm(x), *cross_kv)
        return x + self.fc2(self.fc1(self.final_layer_norm(x)).gelu())


class Decoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model, weight_init=nn.init.EmptyInitStrategy())
        self.embed_positions = nn.Embedding(cfg.max_target_positions, cfg.d_model, weight_init=nn.init.EmptyInitStrategy())
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.decoder_layers)])
        self.layer_norm = _layer_norm(cfg.d_model)
        self.caches: list[KVLayerCache] = []
        self.cross_kv: list[tuple[Tensor, Tensor]] = []

    def reset(self, encoder_out: Tensor) -> None:
        head_dim: int = self.cfg.d_model // self.cfg.decoder_attention_heads
        self.caches = [KVLayerCache(self.cfg.decoder_attention_heads, self.cfg.max_target_positions, head_dim) for _ in self.layers]
        self.cross_kv = [layer.encoder_attn.project_kv(encoder_out) for layer in self.layers]

    @property
    def pos(self) -> int:
        return self.caches[0].pos

    def forward(self, tokens: Tensor) -> Tensor:
        start: int = self.pos
        x = self.embed_tokens(tokens) + self.embed_positions.weight[start : start + tokens.shape[1]]
        for layer, cache, cross_kv in zip(self.layers, self.caches, self.cross_kv, strict=True):
            x = layer(x, cache, cross_kv)
        x = self.layer_norm(x[:, -1:, :])
        return (x @ self.embed_tokens.weight.T).reshape(-1)


class WhisperModel(SnapshotModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    @staticmethod
    def from_snapshot(snapshot_file: str) -> WhisperModel:
        tensors, metadata = deserialize(snapshot_file)
        if metadata.get('architecture') != 'whisper':
            raise RuntimeError(f'{snapshot_file} holds a {metadata.get("architecture") or "nameless"} model, not whisper')
        context.set_default_dtype(getattr(dtype, metadata['dtype']))
        model = WhisperModel(decode_config(Config, metadata['model_config']))
        model.load_tensors(tensors, metadata, source=snapshot_file)
        return model

    def encode(self, mel: Tensor) -> Tensor:
        return self.encoder(mel)

    def detect_language(self, encoder_out: Tensor, sot: int, lang_ids: list[int]) -> int:
        self.decoder.reset(encoder_out)
        logits = self.decoder(Tensor([[sot]], dtype=dtype.int64))
        return lang_ids[int(logits[Tensor(lang_ids, dtype=dtype.int64)].argmax(dim=0).item())]

    def generate(self, encoder_out: Tensor, prompt: list[int], eos: int, suppress: Tensor, begin_suppress: Tensor, max_tokens: int) -> Iterator[int]:
        self.decoder.reset(encoder_out)
        logits = self.decoder(Tensor([prompt], dtype=dtype.int64))
        max_tokens = min(max_tokens, self.cfg.max_target_positions - len(prompt))
        for i in range(max_tokens):
            logits = logits + (suppress + begin_suppress if i == 0 else suppress)
            tok: int = int(logits.argmax(dim=0).item())
            if tok == eos:
                return
            yield tok
            logits = self.decoder(Tensor([[tok]], dtype=dtype.int64))
