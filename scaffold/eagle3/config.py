# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""Draft-config builder for a Nemotron EAGLE-3 target.

REFERENCE SCAFFOLD — validate before production use.

Same idea as the DSpark variant (../modeling/config.py): DeepSpec's EAGLE-3 path deep-copies the
(Qwen3/Gemma) target config, which a hybrid Nemotron-H config can't stand in for, so we build a
clean Qwen3-style draft config and map the standard transformer dimensions across explicitly. The
EAGLE-3-specific knobs (ttt_length, step_loss_decay, draft_num_hidden_layers) replace DSpark's
block/markov/confidence fields. The draft inherits the target's embeddings and frozen LM head at
train time (see ../eagle3_trainer.py), so vocab_size and hidden_size must match the target — which
they do, because we read them off the target config here.
"""
import copy

from deepspec.modeling.eagle3.common import validate_eagle3_target_layer_ids

# The draft reuses DeepSpec's Qwen3 EAGLE-3 transformer block, so the draft config is a Qwen3Config.
from transformers import Qwen3Config


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"

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


def build_draft_config(*, target_config, model_args):
    text_config = get_nemotron_text_config(target_config)

    missing = [f for f in _REQUIRED_TEXT_FIELDS if not hasattr(text_config, f)]
    assert not missing, (
        "Nemotron target text config is missing standard transformer fields "
        f"{missing!r}. Map these explicitly for your checkpoint before training."
    )

    num_target_layers = int(text_config.num_hidden_layers)
    target_layer_ids = validate_eagle3_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    ttt_length = int(model_args.ttt_length)
    assert ttt_length >= 1, f"ttt_length must be >= 1, got {ttt_length}"
    step_loss_decay = float(model_args.step_loss_decay)
    assert step_loss_decay > 0.0, f"step_loss_decay must be > 0.0, got {step_loss_decay}"
    draft_num_hidden_layers = int(model_args.draft_num_hidden_layers)
    assert draft_num_hidden_layers >= 1, (
        f"draft_num_hidden_layers must be >= 1, got {draft_num_hidden_layers}"
    )

    head_dim = getattr(
        text_config,
        "head_dim",
        int(text_config.hidden_size) // int(text_config.num_attention_heads),
    )

    # Clean draft config from the target's standard transformer dimensions (see the note in
    # ../modeling/config.py on why this is equivalent to DeepSpec's deep-copy path).
    draft_config = Qwen3Config(
        vocab_size=int(text_config.vocab_size),
        hidden_size=int(text_config.hidden_size),
        intermediate_size=int(text_config.intermediate_size),
        num_hidden_layers=draft_num_hidden_layers,
        num_attention_heads=int(text_config.num_attention_heads),
        num_key_value_heads=int(text_config.num_key_value_heads),
        head_dim=int(head_dim),
        hidden_act=str(getattr(text_config, "hidden_act", "silu")),
        max_position_embeddings=int(text_config.max_position_embeddings),
        rms_norm_eps=float(text_config.rms_norm_eps),
        rope_theta=float(getattr(text_config, "rope_theta", 1_000_000.0)),
        tie_word_embeddings=False,
    )

    # EAGLE-3 draft fields (mirrors deepspec/modeling/eagle3/qwen3/config.py).
    draft_config.architectures = ["Qwen3Eagle3Model"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = draft_num_hidden_layers
    draft_config.layer_types = ["full_attention"] * draft_num_hidden_layers
    draft_config.target_model_name_or_path = str(model_args.target_model_name_or_path)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.ttt_length = ttt_length
    draft_config.step_loss_decay = step_loss_decay
    draft_config.draft_num_hidden_layers = draft_num_hidden_layers
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    return draft_config


__all__ = ["build_draft_config", "get_nemotron_text_config"]
