# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""Draft-config builder for a Nemotron DSpark target.

REFERENCE SCAFFOLD — validate before production use.

DeepSpec builds the draft model from the *target* config. The Qwen3 path simply deep-copies
the target config (it is already a standard transformer config). A Nemotron-H target config is
a hybrid Mamba-Transformer config and is **not** drop-in usable by the Qwen3 draft model, so we
instead construct a clean Qwen3-style draft config and copy the standard transformer dimensions
across explicitly. The draft is a small RoPE transformer that consumes the target's hidden
states; it is architecture-agnostic with respect to the target, which is why reusing the Qwen3
draft block is a sound starting point. Matching a Nemotron-native attention block is a possible
optimization, not a requirement.

The field mapping below is the part most likely to need adjustment for a specific checkpoint —
inspect your target's `config.json` and confirm the attribute names before a long run.
"""
import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids

# The draft reuses DeepSpec's Qwen3 DSpark transformer block, so the draft config is a Qwen3Config.
from transformers import Qwen3Config


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"

# Standard transformer fields we need to read off the Nemotron (text) config.
_REQUIRED_TEXT_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rms_norm_eps",
)


def get_nemotron_text_config(target_config):
    """Return the transformer text-config carrying the standard dims.

    Nemotron-H exposes the transformer dimensions at the top level; some packagings nest them
    under ``.text_config``. Prefer a nested text_config if present, else use the config itself.
    """
    text_config = getattr(target_config, "text_config", None) or target_config
    return copy.deepcopy(text_config)


def build_draft_config(target_config, model_args):
    text_config = get_nemotron_text_config(target_config)

    missing = [f for f in _REQUIRED_TEXT_FIELDS if not hasattr(text_config, f)]
    assert not missing, (
        "Nemotron target text config is missing standard transformer fields "
        f"{missing!r}. Map these explicitly for your checkpoint before training."
    )

    num_target_layers = int(text_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers

    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when confidence_head_alpha > 0."
        )

    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    head_dim = getattr(
        text_config,
        "head_dim",
        int(text_config.hidden_size) // int(text_config.num_attention_heads),
    )

    # Construct a clean draft config from the target's standard transformer dimensions.
    #
    # DeepSpec's own Qwen3 path simply deep-copies the (HF Qwen3) target config and bolts the
    # DSpark fields onto it. The Qwen3 DSpark draft block reads only standard transformer
    # attributes (head_dim, num_attention_heads, num_key_value_heads, attention_dropout,
    # attention_bias, rms_norm_eps, layer_types, ...). A Nemotron-H config is not a Qwen3 config,
    # so rather than deep-copy it we build an equivalent Qwen3Config carrying the same standard
    # fields; HF defaults supply the rest (e.g. attention_dropout=0.0, attention_bias=False).
    # If your Nemotron checkpoint differs on any of these (notably hidden_act or rope settings),
    # set them here explicitly.
    draft_config = Qwen3Config(
        vocab_size=int(text_config.vocab_size),
        hidden_size=int(text_config.hidden_size),
        intermediate_size=int(text_config.intermediate_size),
        num_hidden_layers=num_draft_layers,
        num_attention_heads=int(text_config.num_attention_heads),
        num_key_value_heads=int(text_config.num_key_value_heads),
        head_dim=int(head_dim),
        hidden_act=str(getattr(text_config, "hidden_act", "silu")),
        max_position_embeddings=int(text_config.max_position_embeddings),
        rms_norm_eps=float(text_config.rms_norm_eps),
        # Fallback only if the target config omits rope_theta — verify against your checkpoint.
        rope_theta=float(getattr(text_config, "rope_theta", 1_000_000.0)),
        tie_word_embeddings=False,
    )

    # DSpark draft fields (mirrors deepspec/modeling/dspark/qwen3/config.py).
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.target_model_type = str(getattr(target_config, "model_type", "nemotron_h"))
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(model_args.confidence_head_with_markov)
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = ["build_draft_config", "get_nemotron_text_config"]
