# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Shape and wiring checks for the vision tower and the multimodal text path.

Numerical agreement with transformers was verified separately against a reference dump. What is guarded
here is everything that can drift without torch around: parameter names the converter relies on, the
placeholder splice, and the rotary cursor that generation after an image depends on.
"""

from magnetron import Tensor, context, dtype

context.stop_grad_recorder()
context.set_default_device('cpu')
context.set_default_dtype(dtype.float32)

from magnetron_models.models.qwen3_5 import Config as DenseConfig, Qwen35Model  # noqa: E402
from magnetron_models.models.qwen3_5_moe import Config, Qwen35MoeModel  # noqa: E402
from magnetron_models.models.qwen3_5.vision import PreprocessedImage, VisionConfig, VisionModel  # noqa: E402
from magnetron_models.models.qwen3_5.vision.image import preprocess_pixels  # noqa: E402
from magnetron_models.models.qwen3_5.vision.multimodal import scatter_image_embeds  # noqa: E402

TINY_VISION = VisionConfig(
    depth=2, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=64, num_position_embeddings=2304, attn_query_chunk=8
)


def _tiny_text_config(**overrides: object) -> Config:
    base = dict(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        full_attention_interval=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
        max_position_embeddings=1024,
        vision=TINY_VISION,
        # Remapped into the tiny vocab, the real ids sit past 248000.
        image_token_id=500,
        video_token_id=501,
        vision_start_token_id=502,
        vision_end_token_id=503,
    )
    base.update(overrides)
    return Config(**base)


def _tiny_dense_config(**overrides: object) -> DenseConfig:
    base = dict(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        full_attention_interval=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        tie_word_embeddings=False,
        max_position_embeddings=1024,
        vision=TINY_VISION,
        image_token_id=500,
        video_token_id=501,
        vision_start_token_id=502,
        vision_end_token_id=503,
    )
    base.update(overrides)
    return DenseConfig(**base)


def _randomize(model: object) -> None:
    for _, param in model.named_parameters():
        param.data = Tensor.normal(*param.shape, mean=0.0, std=0.02)


def test_parameter_names_match_the_checkpoint_layout() -> None:
    names = {name for name, _ in VisionModel(TINY_VISION).named_parameters()}
    # These are the checkpoint's model.visual.* keys with the prefix stripped, which is all the converter does.
    assert 'patch_embed.proj.weight' in names
    assert 'patch_embed.proj.bias' in names
    assert 'pos_embed.weight' in names
    assert 'merger.norm.weight' in names
    assert 'merger.linear_fc1.bias' in names
    assert 'merger.linear_fc2.weight' in names
    for block in range(TINY_VISION.depth):
        for suffix in ('norm1.weight', 'norm2.bias', 'attn.qkv.weight', 'attn.proj.bias', 'mlp.linear_fc1.weight', 'mlp.linear_fc2.bias'):
            assert f'blocks.{block}.{suffix}' in names


def test_patch_embed_folds_the_conv_kernel() -> None:
    # Conv3d(in=3, out=hidden, kernel=stride=[2, 16, 16]) has exactly this many weights per output channel.
    weight = dict(VisionModel(TINY_VISION).named_parameters())['patch_embed.proj.weight']
    assert tuple(weight.shape) == (TINY_VISION.hidden_size, 3 * 2 * 16 * 16)
    assert TINY_VISION.patch_dim == 3 * 2 * 16 * 16


def test_tower_emits_one_token_per_merge_block() -> None:
    model = VisionModel(TINY_VISION)
    _randomize(model)
    grids = [(1, 4, 6), (1, 2, 2)]
    patches = sum(t * h * w for t, h, w in grids)
    out = model.forward(Tensor.normal(patches, TINY_VISION.patch_dim, mean=0.0, std=1.0), grids)
    assert tuple(out.shape) == (sum(t * (h // 2) * (w // 2) for t, h, w in grids), TINY_VISION.out_hidden_size)


def test_query_chunking_does_not_change_the_result() -> None:
    import dataclasses

    grids = [(1, 4, 6)]
    pixels = Tensor.normal(24, TINY_VISION.patch_dim, mean=0.0, std=1.0)
    weights = None
    outputs = []
    # A budget small enough to force one query row per block must agree with attending the segment whole.
    for budget in (1 << 24, 4096, 64):
        cfg = dataclasses.replace(TINY_VISION, attn_score_budget=budget)
        model = VisionModel(cfg)
        if weights is None:
            _randomize(model)
            weights = {name: param for name, param in model.named_parameters()}
        else:
            for name, param in model.named_parameters():
                param.data = weights[name]
        outputs.append(model.forward(pixels, grids).reshape(-1).tolist())
    for other in outputs[1:]:
        assert max(abs(a - b) for a, b in zip(outputs[0], other)) < 1e-5


def test_query_block_keeps_the_score_matrix_within_budget() -> None:
    cfg = TINY_VISION
    for patches in (16, 280, 4096, 16384, 65536):
        block = cfg.query_block(patches)
        assert block >= 1
        assert block <= cfg.attn_query_chunk
        # Peak memory tracks this product, so it must stay bounded no matter how large the image is.
        assert block * patches * cfg.num_heads <= max(cfg.attn_score_budget, patches * cfg.num_heads)


def test_default_ceiling_is_the_checkpoint_ceiling() -> None:
    from magnetron_models.models.qwen3_5.vision.config import CHECKPOINT_MAX_PIXELS

    assert VisionConfig().max_pixels == CHECKPOINT_MAX_PIXELS


def test_preprocess_produces_the_patch_matrix() -> None:
    cfg = TINY_VISION
    height, width = 32, 64  # Already a multiple of patch_size * merge_size, so no resize is involved.
    raw = bytes((x * 7 + y * 3) % 256 for y in range(height) for x in range(width) for _ in range(3))
    pixels = preprocess_pixels(raw, height, width, cfg)
    assert tuple(pixels.shape) == ((height // 16) * (width // 16), cfg.patch_dim)


def test_image_tokens_land_on_the_placeholder_rows() -> None:
    cfg = _tiny_text_config()
    model = Qwen35MoeModel(cfg)
    _randomize(model)
    grid = (1, 4, 6)
    image = PreprocessedImage(
        pixel_values=Tensor.normal(grid[1] * grid[2], cfg.vision.patch_dim, mean=0.0, std=1.0),
        grid_thw=grid,
        merge_size=cfg.vision.spatial_merge_size,
    )
    assert image.num_tokens == 6
    token_ids = [1, 2, cfg.vision_start_token_id] + [cfg.image_token_id] * 6 + [cfg.vision_end_token_id, 3]
    positions, embeds, rows = model._multimodal_inputs(token_ids, [image])
    assert tuple(positions.shape) == (3, 1, len(token_ids))
    assert tuple(embeds.shape) == (6, cfg.hidden_size)
    assert rows.tolist() == [3, 4, 5, 6, 7, 8]

    spliced = scatter_image_embeds(model.embed_tokens(Tensor(token_ids, dtype=dtype.int64).reshape(1, -1)), embeds, rows)
    flat = spliced.reshape(len(token_ids), cfg.hidden_size).tolist()
    want = embeds.tolist()
    for offset, row in enumerate(rows.tolist()):
        for a, b in zip(flat[row], want[offset]):
            assert abs(a - b) < 1e-5
    # Text rows must survive the splice untouched.
    text_row = model.embed_tokens(Tensor([1], dtype=dtype.int64).reshape(1, 1)).reshape(-1).tolist()
    for a, b in zip(flat[0], text_row):
        assert abs(a - b) < 1e-5


def test_rope_cursor_compresses_image_spans() -> None:
    cfg = _tiny_text_config()
    model = Qwen35MoeModel(cfg)
    grid = (1, 4, 6)
    image = PreprocessedImage(
        pixel_values=Tensor.zeros(grid[1] * grid[2], cfg.vision.patch_dim),
        grid_thw=grid,
        merge_size=cfg.vision.spatial_merge_size,
    )
    _randomize(model)
    token_ids = [1, cfg.vision_start_token_id] + [cfg.image_token_id] * 6 + [cfg.vision_end_token_id, 2]
    model._multimodal_inputs(token_ids, [image])
    # 10 prompt tokens, but the 6 image tokens only occupy max(4, 6) // 2 = 3 rotary positions.
    assert model.rope_pos == 10 - 6 + 3
    before = model.rope_pos
    assert model._decode_positions().reshape(-1).tolist() == [float(before)] * 3
    assert model.rope_pos == before + 1


def test_text_only_config_builds_no_vision_tower() -> None:
    model = Qwen35MoeModel(_tiny_text_config(vision=None))
    assert model.visual is None
    assert not any(name.startswith('visual.') for name, _ in model.named_parameters())
    positions, embeds, rows = model._multimodal_inputs([1, 2, 3, 4], [])
    assert embeds is None and rows is None
    assert positions.reshape(-1).tolist() == [0.0, 1.0, 2.0, 3.0] * 3
    assert model.rope_pos == 4


def test_mrope_section_must_tile_the_rotary_dim() -> None:
    try:
        _tiny_text_config(mrope_section=(1, 1, 1))
    except ValueError as e:
        assert 'does not tile' in str(e)
    else:
        raise AssertionError('a section split that does not cover the rotary dim must raise')


def test_vision_out_hidden_size_is_pinned_to_the_text_width() -> None:
    import dataclasses

    cfg = _tiny_text_config(hidden_size=128, vision=dataclasses.replace(TINY_VISION, out_hidden_size=999))
    assert cfg.vision.out_hidden_size == 128


def test_dense_and_moe_share_the_same_vision_parameters() -> None:
    dense = {name for name, _ in Qwen35Model(_tiny_dense_config()).named_parameters() if name.startswith('visual.')}
    moe = {name for name, _ in Qwen35MoeModel(_tiny_text_config()).named_parameters() if name.startswith('visual.')}
    assert dense == moe and dense, 'both towers must expose the identical checkpoint key set'


def test_dense_model_accepts_images() -> None:
    cfg = _tiny_dense_config()
    model = Qwen35Model(cfg)
    _randomize(model)
    grid = (1, 4, 6)
    image = PreprocessedImage(
        pixel_values=Tensor.normal(grid[1] * grid[2], cfg.vision.patch_dim, mean=0.0, std=1.0),
        grid_thw=grid,
        merge_size=cfg.vision.spatial_merge_size,
    )
    token_ids = [1, cfg.vision_start_token_id] + [cfg.image_token_id] * image.num_tokens + [cfg.vision_end_token_id, 2]
    positions, embeds, rows = model._multimodal_inputs(token_ids, [image])
    assert tuple(embeds.shape) == (image.num_tokens, cfg.hidden_size)
    logits = model(Tensor(token_ids, dtype=dtype.int64).reshape(1, -1), positions, embeds, rows)
    assert tuple(logits.shape) == (1, len(token_ids), cfg.vocab_size)
    # 10 prompt tokens, but the 6 image tokens only occupy max(4, 6) // 2 = 3 rotary positions.
    assert model.rope_pos == len(token_ids) - image.num_tokens + 3


def test_dense_text_only_config_builds_no_vision_tower() -> None:
    model = Qwen35Model(_tiny_dense_config(vision=None))
    assert model.visual is None
    assert not any(name.startswith('visual.') for name, _ in model.named_parameters())
    positions, embeds, rows = model._multimodal_inputs([1, 2, 3], [])
    assert embeds is None and rows is None
    assert positions.reshape(-1).tolist() == [0.0, 1.0, 2.0] * 3


def test_dense_vision_out_hidden_size_is_pinned_to_the_text_width() -> None:
    import dataclasses

    cfg = _tiny_dense_config(hidden_size=128, vision=dataclasses.replace(TINY_VISION, out_hidden_size=999))
    assert cfg.vision.out_hidden_size == 128
