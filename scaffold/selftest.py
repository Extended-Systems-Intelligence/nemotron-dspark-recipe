# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""No-GPU smoke test for the Nemotron DSpark extension.

Run from a DeepSpec checkout that has the extension applied (see INTEGRATION.md):

    python scaffold/selftest.py

It checks, without any GPU or weights, that:
  - the extension modules import,
  - the 'nemotron' chat template registers,
  - build_draft_config produces a usable draft config from a small fake target config.

This catches the most common integration breakage (import/registration/field-mapping). It does
NOT validate training correctness, layer-tap quality, or the chat-template token strings.
"""
import sys
import types


def main() -> int:
    ok = True

    # 1. Template registration.
    try:
        from deepspec.data.parser import TEMPLATE_REGISTRY

        import template  # noqa: F401  (registers "nemotron" on import)

        TEMPLATE_REGISTRY.get("nemotron")
        print("[ok] 'nemotron' chat template registered")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] chat template registration: {exc!r}")

    # 2. Draft-config build from a minimal fake Nemotron-like target config.
    try:
        from modeling.config import build_draft_config

        # DeepSpec's build_draft_config reads model_args both by attribute (model_args.x) and by
        # membership ("x" in model_args), so emulate its dotted-dict, not a SimpleNamespace.
        class DotDict(dict):
            __getattr__ = dict.__getitem__

        fake_target = types.SimpleNamespace(
            model_type="nemotron_h",
            vocab_size=131072,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=56,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1e-5,
            max_position_embeddings=131072,
            rope_theta=1_000_000.0,
            hidden_act="silu",
        )
        model_args = DotDict(
            num_draft_layers=5,
            block_size=7,
            target_layer_ids=[8, 20, 32, 44, 52],
            mask_token_id=4,
            num_anchors=512,
            confidence_head_alpha=1.0,
            confidence_head_with_markov=True,
            markov_rank=256,
            markov_head_type="vanilla",
        )

        draft_config = build_draft_config(target_config=fake_target, model_args=model_args)
        assert draft_config.num_hidden_layers == 5
        assert draft_config.architectures == ["Qwen3DSparkModel"]
        assert list(draft_config.target_layer_ids) == [8, 20, 32, 44, 52]
        print("[ok] build_draft_config produced a draft config")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] build_draft_config: {exc!r}")

    # 3. EAGLE-3 draft-config build (the sibling drafter).
    try:
        from eagle3.config import build_draft_config as build_eagle3_draft_config

        class _DotDict(dict):
            __getattr__ = dict.__getitem__

        e3_target = types.SimpleNamespace(
            model_type="nemotron_h",
            vocab_size=131072,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=56,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1e-5,
            max_position_embeddings=131072,
            rope_theta=1_000_000.0,
            hidden_act="silu",
        )
        e3_args = _DotDict(
            target_model_name_or_path="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            target_layer_ids=[2, 14, 26, 38, 50],
            ttt_length=7,
            step_loss_decay=0.8,
            draft_num_hidden_layers=1,
        )
        e3_cfg = build_eagle3_draft_config(target_config=e3_target, model_args=e3_args)
        assert e3_cfg.architectures == ["Qwen3Eagle3Model"]
        assert e3_cfg.num_hidden_layers == 1
        print("[ok] EAGLE-3 build_draft_config produced a draft config")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] EAGLE-3 build_draft_config: {exc!r}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
