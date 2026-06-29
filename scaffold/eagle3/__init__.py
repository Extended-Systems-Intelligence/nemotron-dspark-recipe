# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""Nemotron EAGLE-3 draft modeling.

The draft is a small transformer over the target's hidden states, so we reuse DeepSpec's Qwen3
EAGLE-3 draft model. The Nemotron-specific work lives in ``config.build_draft_config`` (dimension
mapping). The trainer inherits the target's embeddings + frozen LM head (see ../eagle3_trainer.py).
"""
from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model as NemotronEagle3Model

from .config import build_draft_config, get_nemotron_text_config

__all__ = ["NemotronEagle3Model", "build_draft_config", "get_nemotron_text_config"]
