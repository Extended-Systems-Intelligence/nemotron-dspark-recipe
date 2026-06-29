# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""Nemotron DSpark draft modeling.

The draft is a standard RoPE transformer over the target's hidden states, so we reuse
DeepSpec's Qwen3 DSpark draft model rather than re-implementing one. The Nemotron-specific work
lives in ``config.build_draft_config`` (dimension mapping). If a Nemotron-native draft block is
later shown to improve accepted length, replace the alias below with a dedicated model class.
"""
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel as NemotronDSparkModel

from .config import build_draft_config, get_nemotron_text_config

__all__ = ["NemotronDSparkModel", "build_draft_config", "get_nemotron_text_config"]
