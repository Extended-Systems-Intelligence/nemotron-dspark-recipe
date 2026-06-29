# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""NemotronEagle3Trainer — an EAGLE-3 trainer for a Nemotron target.

REFERENCE SCAFFOLD — validate before production use.

Mirrors ``Gemma4Eagle3Trainer``: subclass the Qwen3 EAGLE-3 trainer and override only the
draft-model construction. Inherits ``build_models`` (which loads the target to harvest its
embeddings + frozen LM head), ``run_batch`` (the EAGLE-3 TTT loss), and the cache collator.

NEMOTRON-H CAVEAT — the target load. EAGLE-3's ``build_models`` loads the FULL target on CPU via
``AutoModelForCausalLM.from_pretrained`` purely to copy out its input embeddings and output head
(it freezes them into the draft; it does NOT generate). Two things to know for Nemotron-H:
  1. Depending on your transformers version / checkpoint, the Auto load may need
     ``trust_remote_code=True``. If DeepSpec's upstream ``build_models`` doesn't pass it, add it
     there (one line), or override ``build_models`` here to do so.
  2. It needs enough CPU RAM to hold the full target in the training precision (bf16).
Because the target is used only for embedding/head extraction, the broken Nemotron-H
generate/cache path (see ../NEMOTRON-H-NOTES.md) does NOT affect this step.

Add ``NemotronEagle3Trainer`` to ``deepspec/trainer/__init__.py``'s exports (see ./INTEGRATION.md).
"""
from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model
from deepspec.trainer.eagle3_trainer import Qwen3Eagle3Trainer

from .eagle3.config import build_draft_config as build_nemotron_eagle3_config


class NemotronEagle3Trainer(Qwen3Eagle3Trainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_nemotron_eagle3_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3Eagle3Model(draft_config)
