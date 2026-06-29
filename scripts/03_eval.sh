#!/usr/bin/env bash
# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stage 3: evaluate the trained draft over DeepSpec's speculative-decoding benchmarks.
# Thin wrapper over DeepSpec's eval.py. Run from a DeepSpec checkout with the extension applied.
#
# eval.py spawns one worker per visible GPU; RANK/WORLD_SIZE mean node_rank/node_count.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

# Must match the target the draft was trained against.
target_name_or_path=${target_name_or_path:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B}

# Training writes checkpoints under ~/checkpoints/<project_name>/<exp_name>/step_*.
draft_name_or_path=${draft_name_or_path:-${HOME}/checkpoints/deepspec/dspark_block8_nemotron_nano/step_latest}

python eval.py \
    --target_name_or_path "${target_name_or_path}" \
    --draft_name_or_path "${draft_name_or_path}"
