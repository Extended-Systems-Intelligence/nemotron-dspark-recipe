# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""NemotronDSparkTrainer — a DSpark trainer for a Nemotron target.

REFERENCE SCAFFOLD — validate before production use.

Mirrors ``Gemma4DSparkTrainer``: subclass the Qwen3 DSpark trainer and override only the
draft-model construction. Inherits the cache collator, the training step (``run_batch``, which
consumes ``target_hidden_states`` / ``target_last_hidden_states`` / ``loss_mask``), and the
DSpark loss.

Add ``NemotronDSparkTrainer`` to ``deepspec/trainer/__init__.py``'s exports after dropping this
in (see ../INTEGRATION.md).
"""
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer

from .modeling.config import build_draft_config as build_nemotron_draft_config


class NemotronDSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_nemotron_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)
