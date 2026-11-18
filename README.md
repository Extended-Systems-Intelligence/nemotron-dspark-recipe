# Nemotron + DSpark: a speculative-decoding training recipe

A community recipe, reference scaffold, and field notes for training a **DSpark-class
speculative-decoding draft model** against **NVIDIA Nemotron** targets using DeepSeek's
open-source [DeepSpec](https://github.com/deepseek-ai/DeepSpec) framework.

> **Status: recipe validated end-to-end — proof-of-recipe results, not a production checkpoint.**
> DeepSpec ships first-class support for the **Qwen3** and **Gemma** target families only.
> Nemotron is **not** supported upstream. This repository provides the missing extension
> points (chat template, draft-config builder, trainer class, and a worked config) plus a
> step-by-step recipe. The recipe now trains a Nemotron draft end-to-end on 2×GB10 and we
> have a first head-to-head acceptance measurement (see [`BENCHMARKS.md`](./BENCHMARKS.md)):
> **DSpark beats EAGLE-3 on every task at equal budget.** These are short, deliberately-small
> **proof drafts** trained to settle the recipe and the architecture question — not production
> models. Treat the released artifacts as a validated starting point, not a finished checkpoint.

## Why this exists

Speculative decoding accelerates autoregressive generation: a small **draft** model proposes
tokens and the large **target** model verifies them in batches, so output is **lossless**
with respect to the target while latency drops. DSpark adds a confidence-scheduled,
semi-autoregressive draft that improves accepted length over EAGLE-3 and MTP-1 baselines.

Nemotron is a strong open-weight target family under a permissive license, but wiring a
modern draft trainer to it is non-obvious — especially for the **Nemotron-H hybrid
Mamba-Transformer** checkpoints, where several stock Hugging Face code paths do not behave
the way EAGLE/DSpark data-prep expects. This repo captures what it actually takes.

## What's here

| Path | What it is |
| --- | --- |
| [`GUIDE.md`](./GUIDE.md) | End-to-end recipe: data prep → target-cache → draft training → eval → serving, with NVIDIA-cloud notes. |
| [`NEMOTRON-H-NOTES.md`](./NEMOTRON-H-NOTES.md) | Field notes on the Nemotron-H hybrid-architecture landmines that bite the data-prep / cache step, and the plumbing workarounds — including the `attn_implementation="eager"` requirement and a worked example (layer taps, template, ids) for `Nemotron-3-Nano-30B-A3B`. |
| [`TWO-SPARK-NOTES.md`](./TWO-SPARK-NOTES.md) | Running the data-gen step on one or two **GB10 DGX Sparks**: parallel target serving behind a least-connections load balancer, GB10 serving gotchas, and a 2×GB10 sizing note. |
| [`GB10-TRAINING-NOTES.md`](./GB10-TRAINING-NOTES.md) | The **draft-training** step on GB10: the `flex_attention`→dense-`sdpa`-mask fix (required, and a silent-corruption trap for EAGLE-3 `ttt>1`), step-time decay on unified memory and how to stop it, frequent checkpointing + per-rank saves for 2-box training, and two eval-harness fixes for a Nemotron-H target. |
| [`scaffold/`](./scaffold/) | The DeepSpec extension for a Nemotron target: chat template, draft-config builders, `NemotronDSparkTrainer` **and `NemotronEagle3Trainer`**, and `INTEGRATION.md`. |
| [`configs/`](./configs/) | Worked configs for a Nemotron-Nano target — `dspark_nemotron_nano.py` and `eagle3_nemotron_nano.py` — modeled on DeepSpec's own configs. |

**Two drafters, same target.** DeepSpec ships both DSpark and EAGLE-3; this recipe wires both for
Nemotron so you can train either and benchmark them head-to-head ([`BENCHMARKS.md`](./BENCHMARKS.md)).
DSpark adds a semi-autoregressive draft + load-aware verification; EAGLE-3 is the established
autoregressive baseline. Which one wins on your target and workload is an empirical question.
| [`scripts/`](./scripts/) | Thin wrappers over DeepSpec's pipeline with the Nemotron target and NVIDIA-cloud-friendly defaults. |
| [`BENCHMARKS.md`](./BENCHMARKS.md) | First measured results — the DSpark-vs-EAGLE-3 acceptance head-to-head (proof run) — plus the reporting format and auto-generated tables for a full production run (serving tok/s, long-context, concurrency). |
| [`benchmarks/`](./benchmarks/) | The results harness: schema, workload mapping, the collect/render scripts that turn an eval run into the committed `BENCHMARKS.md` tables, and the proof runs as committed artifacts. See [`benchmarks/README.md`](./benchmarks/README.md). |

## Quickstart

```bash
# 0. From your clone of THIS recipe repo, remember where it is — the stage-1-3 wrapper
#    scripts run from inside the DeepSpec checkout, but the stage-4 reporting scripts and
#    the committed results live here.
RECIPE=$(pwd)

# 1. Get DeepSpec and install it
git clone https://github.com/deepseek-ai/DeepSpec && cd DeepSpec
python -m pip install -r requirements.txt

# 2. Drop in the Nemotron extension (see scaffold/INTEGRATION.md for the exact paths)
#    - register the "nemotron" chat template
#    - add deepspec/modeling/dspark/nemotron/
#    - add NemotronDSparkTrainer to deepspec/trainer/
#    - copy configs/dspark_nemotron_nano.py into config/dspark/

# 3. Run the stages from the DeepSpec checkout (read GUIDE.md first — storage warning)
bash "$RECIPE"/scripts/01_prepare_data.sh
bash "$RECIPE"/scripts/02_train.sh
bash "$RECIPE"/scripts/03_eval.sh | tee eval_table.txt   # DeepSpec eval prints a TABLE, not JSON

# 4. Record the run in the recipe repo, where results are committed (see benchmarks/README.md).
#    DeepSpec's eval.py prints a human-readable table (and accept_length to TensorBoard) — it does
#    not emit JSON — so turn its per-task acceptance / accepted-length into a small JSON report,
#    then collect + render:
cd "$RECIPE"
python scripts/collect_results.py --print-template > my_run.json   # fill in config + any serving numbers
# write report.json: {"humaneval": {"accepted_length": 3.1, "acceptance": 0.78}, ...}
bash scripts/04_report.sh my_run.json report.json
```

Read [`GUIDE.md`](./GUIDE.md) before running anything — the target-cache stage produces a
**very large** on-disk cache and the Nemotron-H target needs the workarounds in
[`NEMOTRON-H-NOTES.md`](./NEMOTRON-H-NOTES.md). If you train on **GB10 / DGX Spark**, read
[`GB10-TRAINING-NOTES.md`](./GB10-TRAINING-NOTES.md) first — the training step needs the
`flex_attention`→dense-`sdpa`-mask fix or it won't run at usable speed.

## Licensing & attribution

This repository is published by **Extended Systems Intelligence Corporation (XSI)** under the
[Apache License 2.0](./LICENSE). It builds on and references third-party work — DeepSpec (MIT),
SpecForge (Apache-2.0), DFlash (MIT), and NVIDIA Nemotron (NVIDIA Nemotron Open Model License).
See [`NOTICE`](./NOTICE) for the full attribution, including the required Nemotron license notice.

Nemotron models are *"Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License."*

## Contributing

Issues and PRs welcome — particularly benchmark results, corrected `target_layer_ids` for
specific Nemotron checkpoints, and fixes to the draft-config dimension mapping. If you train a
checkpoint with this recipe, please open an issue with your accepted-length numbers and setup,
or contribute a run via the results harness ([`benchmarks/README.md`](./benchmarks/README.md)).

## Discussion

Questions, results, and general discussion happen in the NVIDIA Developer Forums thread where this
recipe was first shared (DGX Spark Projects):
**[Open-source recipe + scaffold: training a DSpark-class speculative-decoding draft for Nemotron](https://forums.developer.nvidia.com/t/open-source-recipe-scaffold-training-a-dspark-class-speculative-decoding-draft-for-nemotron/374876)**.
If you train a checkpoint or hit a GB10-specific snag, that thread and this repo's issue tracker are
the best places to compare notes.
