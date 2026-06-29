#!/usr/bin/env bash
# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stage 2: train the DSpark draft against the Nemotron target cache.
# Thin wrapper over DeepSpec's train.py. Run from a DeepSpec checkout with the extension applied.
#
# DeepSpec's launcher overloads RANK/WORLD_SIZE to mean node_rank/node_count; WORLD_SIZE=1 is a
# single-node run. train.py spawns one worker per visible GPU. Default assumes one 8-GPU node.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

config_path=${config_path:-config/dspark/dspark_nemotron_nano.py}
target_cache_dir=${target_cache_dir:-${HOME}/.cache/deepspec/nemotron_nano_target_cache}

# Tune train.local_batch_size to your GPU memory, e.g. --opts "train.local_batch_size=4".
python train.py \
    --config "${config_path}" \
    --opts "data.target_cache_path=${target_cache_dir}"
