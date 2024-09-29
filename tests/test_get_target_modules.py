import pytest

from aria.config import AriaModelConfig
from aria.lora.utils import get_lora_target_modules


@pytest.fixture
def named_modules():
    return [
        "vision_tower.vision_model.embeddings.patch_embedding",
        "vision_tower.vision_model.embeddings.position_embedding",
        "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj",
        "vision_tower.vision_model.encoder.layers.0.layer_norm1",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc2",
        "vision_tower.vision_model.encoder.layers.0.layer_norm2",
        "multi_modal_projector.query",
        "multi_modal_projector.cross_attn.q_proj",
        "multi_modal_projector.cross_attn.k_proj",
        "multi_modal_projector.cross_attn.v_proj",
        "multi_modal_projector.cross_attn.multihead_attn.in_proj_weight",
        "multi_modal_projector.cross_attn.multihead_attn.out_proj",
        "multi_modal_projector.cross_attn.linear",
        "multi_modal_projector.cross_attn.layer_norm",
        "multi_modal_projector.cross_attn.ln_kv",
        "multi_modal_projector.ln_ffn",
        "multi_modal_projector.ffn.linear_in",
        "multi_modal_projector.ffn.linear_out",
        "language_model.model.embed_tokens",
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.self_attn.k_proj",
        "language_model.model.layers.0.self_attn.v_proj",
        "language_model.model.layers.0.self_attn.o_proj",
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.model.layers.0.mlp.down_proj",
        "language_model.model.layers.0.input_layernorm",
        "language_model.model.layers.0.post_attention_layernorm",
        "language_model.model.norm",
        "language_model.lm_head",
    ]


def test_freeze_vit(named_modules):
    config = AriaModelConfig(
        freeze_vit=True,
        freeze_projector=False,
        freeze_llm=False,
        lora_target_modules=[
            "fc2",
            "linear_out",
            "lm_head",
            "q_proj",
            "linear_in",
            "linear",
            "o_proj",
            "up_proj",
            "fc1",
            "k_proj",
            "down_proj",
            "v_proj",
            "out_proj",
            "gate_proj",
        ],
    )
    target_modules = get_lora_target_modules(named_modules, config)
    expected = [
        "multi_modal_projector.cross_attn.q_proj",
        "multi_modal_projector.cross_attn.k_proj",
        "multi_modal_projector.cross_attn.v_proj",
        "multi_modal_projector.cross_attn.multihead_attn.out_proj",
        "multi_modal_projector.cross_attn.linear",
        "multi_modal_projector.ffn.linear_in",
        "multi_modal_projector.ffn.linear_out",
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.self_attn.k_proj",
        "language_model.model.layers.0.self_attn.v_proj",
        "language_model.model.layers.0.self_attn.o_proj",
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.model.layers.0.mlp.down_proj",
        "language_model.lm_head",
    ]
    assert target_modules == expected
    assert "vision_tower" not in target_modules


def test_freeze_projector(named_modules):
    config = AriaModelConfig(
        freeze_vit=False,
        freeze_projector=True,
        freeze_llm=False,
        lora_target_modules=[
            "fc2",
            "linear_out",
            "lm_head",
            "q_proj",
            "linear_in",
            "linear",
            "o_proj",
            "up_proj",
            "fc1",
            "k_proj",
            "down_proj",
            "v_proj",
            "out_proj",
            "gate_proj",
        ],
    )
    target_modules = get_lora_target_modules(named_modules, config)
    expected = [
        "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc2",
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.self_attn.k_proj",
        "language_model.model.layers.0.self_attn.v_proj",
        "language_model.model.layers.0.self_attn.o_proj",
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.model.layers.0.mlp.down_proj",
        "language_model.lm_head",
    ]
    assert target_modules == expected
    assert "multi_modal_projector" not in target_modules


def test_freeze_llm(named_modules):
    config = AriaModelConfig(
        freeze_vit=False,
        freeze_projector=False,
        freeze_llm=True,
        lora_target_modules=[
            "fc2",
            "linear_out",
            "lm_head",
            "q_proj",
            "linear_in",
            "linear",
            "o_proj",
            "up_proj",
            "fc1",
            "k_proj",
            "down_proj",
            "v_proj",
            "out_proj",
            "gate_proj",
        ],
    )
    target_modules = get_lora_target_modules(named_modules, config)
    expected = [
        "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc2",
        "multi_modal_projector.cross_attn.q_proj",
        "multi_modal_projector.cross_attn.k_proj",
        "multi_modal_projector.cross_attn.v_proj",
        "multi_modal_projector.cross_attn.multihead_attn.out_proj",
        "multi_modal_projector.cross_attn.linear",
        "multi_modal_projector.ffn.linear_in",
        "multi_modal_projector.ffn.linear_out",
    ]
    assert target_modules == expected
    assert "language_model" not in target_modules
