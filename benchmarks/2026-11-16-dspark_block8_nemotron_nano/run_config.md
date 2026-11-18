# Run configuration — 2026-11-16 proof run

Reproducibility record for the DSpark-vs-EAGLE-3 head-to-head summarized in
[`../../BENCHMARKS.md`](../../BENCHMARKS.md). Values are transcribed from the committed configs
and the "What produced these numbers" section; fill any `_` from the actual job before citing.

## Target

- **Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` class, **stock** (no adapters, no modifications)
- **Precision:** bf16
- **Chat template:** `nemotron`

## Drafts (both trained from the same cache, equal budget)

| | DSpark | EAGLE-3 |
|---|---|---|
| Config | [`configs/dspark_nemotron_nano.py`](../../configs/dspark_nemotron_nano.py) | [`configs/eagle3_nemotron_nano.py`](../../configs/eagle3_nemotron_nano.py) |
| `exp_name` | `dspark_block8_nemotron_nano` | `eagle3_ttt7_nemotron_nano` |
| Draft layers | `num_draft_layers=5` | `draft_num_hidden_layers=1` |
| Block / TTT | `block_size=7`* | `ttt_length=7` |
| `target_layer_ids` | `[8, 20, 32, 44, 52]` | `[2, 14, 26, 38, 50]` |
| Seed | 42 | 0 |

\* The experiment/checkpoint name says `block8`, but the committed config has `block_size=7`
(both drafts propose 7 tokens per verification step). Confirm which the trained checkpoint used
and reconcile the name if needed.

## Training (shared)

- **Budget:** 74 optimizer steps / 2 epochs (proof run — ~1–2% of a production budget)
- **`global_batch_size`:** 512, **`local_batch_size`:** 1
- **LR:** 6.0e-4, **warmup_ratio:** 0.04, **precision:** bf16, **sharding:** `no_shard`
- **Hardware:** 2× DGX Spark (GB10, 128 GB unified)
- **Required GB10 fixes:** flex→dense-sdpa-mask path + step-time-decay mitigations — see
  [`../../GB10-TRAINING-NOTES.md`](../../GB10-TRAINING-NOTES.md) §A.

## Cache

- **Samples:** ~19K target-generation samples
- **Captured:** hidden states at the tapped `target_layer_ids` + last hidden state
- **`max_length`:** 4096

## Eval

- **Harness:** DeepSpec acceptance eval (`scripts/03_eval.sh` → `eval.py`)
- **Tasks (9):** gsm8k, math500, aime25, humaneval, mbpp, livecodebench, mt-bench, alpaca,
  arena-hard-v2
- **Prompts/task:** 10–15 · **max new tokens:** 128 · **mode:** thinking-off, warmed
- **Metric read:** acceptance + accepted-length from the eval's speculative stats
- **Required Nemotron-H harness fixes:** `logits_to_keep` slice + hybrid decode cache — see
  [`../../GB10-TRAINING-NOTES.md`](../../GB10-TRAINING-NOTES.md) §D. Small sample counts are
  deliberate (the O(n²) decode-cache fallback makes a full-scale eval take days).

## To fill from the actual job (not in the committed docs)

- Exact `eval.py` revision / DeepSpec commit · GPU count & `CUDA_VISIBLE_DEVICES`
- Draft checkpoint step used for each draft (`step_*`)
- Wall-clock per task · any per-task prompt-count variation within 10–15
