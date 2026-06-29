#!/usr/bin/env bash
# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stage 1: data prep + target-cache for a Nemotron DSpark target.
# Thin wrapper over DeepSpec's scripts/data/* with the Nemotron config/target.
# Run from the root of a DeepSpec checkout with the Nemotron extension applied.
#
# READ FIRST: ../GUIDE.md (esp. the cache storage warning) and ../NEMOTRON-H-NOTES.md.
set -euo pipefail

# SET-FOR-YOUR-CHECKPOINT: a stock Nemotron target.
model_path=${model_path:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B}
config_path=${config_path:-config/dspark/dspark_nemotron_nano.py}

dataset_name=${dataset_name:-mlabonne/open-perfectblend}
test_size=${test_size:-0.05}
train_split_path=train_datasets/perfectblend_train.jsonl
eval_data_dir=eval_datasets
train_data_path=train_datasets/nemotron_nano/perfectblend_train_regen.jsonl
cache_dir=${target_cache_dir:-${HOME}/.cache/deepspec/nemotron_nano_target_cache}

server_host=${server_host:-127.0.0.1}
num_workers=${num_workers:-8}
start_port=${start_port:-30000}
concurrency=${concurrency:-32}
temperature=${temperature:-0.7}
top_p=${top_p:-0.8}
top_k=${top_k:-20}
min_p=${min_p:-0}
max_tokens=${max_tokens:-4096}

server_addresses=()
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    server_addresses+=("${server_host}:$((start_port + worker_id))")
done

echo "Step 1/3: download + split ${dataset_name}"
python scripts/data/download_and_split.py \
    --dataset-name "${dataset_name}" \
    --test-size "${test_size}" \
    --train-output-path "${train_split_path}" \
    --test-output-dir "${eval_data_dir}" \
    --skip-existing

mkdir -p "$(dirname "${train_data_path}")"

echo "Step 2/3: regenerate answers with the Nemotron target"
echo "Serve the target first (OpenAI/SGLang endpoint). Match the thinking mode you will serve."
python scripts/data/generate_train_data.py \
    --model "${model_path}" \
    --server-address "${server_addresses[@]}" \
    --concurrency "${concurrency}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --top-k "${top_k}" \
    --min-p "${min_p}" \
    --max-tokens "${max_tokens}" \
    --disable-thinking \
    --resume \
    --input-file-path "${train_split_path}" \
    --output-file-path "${train_data_path}"

echo "Step 3/3: build the target cache (LARGE on disk — see GUIDE.md)"
echo "Nemotron-H: ensure the target forward runs with use_cache=False (see NEMOTRON-H-NOTES.md)."
python scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size 16
