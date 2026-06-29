# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""EAGLE-3 training config for a Nemotron-Nano target.

REFERENCE SCAFFOLD — validate before production use.

Modeled on DeepSpec's config/eagle3/eagle3_qwen3_4b.py. Copy this into config/eagle3/ of a
DeepSpec checkout that has the Nemotron extension applied (see ../scaffold/INTEGRATION.md).
Fields marked SET-FOR-YOUR-CHECKPOINT must be adjusted to your specific Nemotron model.

This is the EAGLE-3 sibling of configs/dspark_nemotron_nano.py — same target, different drafter.
Train both and benchmark them against each other ([../BENCHMARKS.md]); the better drafter for a
given target + workload is an empirical question, not a foregone one.
"""
import os

# Available after applying scaffold/INTEGRATION.md (exported from deepspec/trainer/__init__.py).
from deepspec.trainer import NemotronEagle3Trainer


BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "eagle3_ttt7_nemotron_nano"
seed = 0

model = dict(
    # SET-FOR-YOUR-CHECKPOINT: a stock Nemotron-Nano checkpoint id or local path.
    target_model_name_or_path="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    # SET-FOR-YOUR-CHECKPOINT: ~5 taps into the target's num_hidden_layers (Mamba+attention both
    # count). PLACEHOLDER spread; pick a low/mid/high set for YOUR layer count, biased toward the
    # attention positions in the hybrid schedule. See ../NEMOTRON-H-NOTES.md §C.
    target_layer_ids=[2, 14, 26, 38, 50],
    ttt_length=7,
    step_loss_decay=0.8,
    draft_num_hidden_layers=1,
)

train = dict(
    trainer_cls=NemotronEagle3Trainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="nemotron",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
