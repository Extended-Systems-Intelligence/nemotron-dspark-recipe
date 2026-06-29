# Nemotron + DSpark: a speculative-decoding training recipe

A community recipe, reference scaffold, and field notes for training a **DSpark-class
speculative-decoding draft model** against **NVIDIA Nemotron** targets using DeepSeek's
open-source [DeepSpec](https://github.com/deepseek-ai/DeepSpec) framework.

> **Status: reference implementation — validate before production use.**
> DeepSpec ships first-class support for the **Qwen3** and **Gemma** target families only.
> Nemotron is **not** supported upstream. This repository provides the missing extension
> points (chat template, draft-config builder, trainer class, and a worked config) plus a
> step-by-step recipe for running the pipeline on NVIDIA cloud GPUs. The scaffold is written
> against DeepSpec's real interfaces but has not been benchmarked end-to-end; treat the
> released artifacts as a starting point, not a finished checkpoint.

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
| [`NEMOTRON-H-NOTES.md`](./NEMOTRON-H-NOTES.md) | Field notes on the Nemotron-H hybrid-architecture landmines that bite the data-prep / cache step, and the plumbing workarounds. |
| [`scaffold/`](./scaffold/) | The DeepSpec extension for a Nemotron target: chat template, draft-config builder, `NemotronDSparkTrainer`, and `INTEGRATION.md`. |
| [`configs/dspark_nemotron_nano.py`](./configs/dspark_nemotron_nano.py) | A worked DSpark config for a Nemotron-Nano target, modeled on DeepSpec's own configs. |
| [`scripts/`](./scripts/) | Thin wrappers over DeepSpec's pipeline with the Nemotron target and NVIDIA-cloud-friendly defaults. |

## Quickstart

```bash
# 1. Get DeepSpec and install it
git clone https://github.com/deepseek-ai/DeepSpec && cd DeepSpec
python -m pip install -r requirements.txt

# 2. Drop in the Nemotron extension (see scaffold/INTEGRATION.md for the exact paths)
#    - register the "nemotron" chat template
#    - add deepspec/modeling/dspark/nemotron/
#    - add NemotronDSparkTrainer to deepspec/trainer/
#    - copy configs/dspark_nemotron_nano.py into config/dspark/

# 3. Run the three stages (read GUIDE.md first — especially the storage warning)
bash scripts/01_prepare_data.sh
bash scripts/02_train.sh
bash scripts/03_eval.sh
```

Read [`GUIDE.md`](./GUIDE.md) before running anything — the target-cache stage produces a
**very large** on-disk cache and the Nemotron-H target needs the workarounds in
[`NEMOTRON-H-NOTES.md`](./NEMOTRON-H-NOTES.md).

## Licensing & attribution

This repository is published by **Extended Systems Intelligence Corporation (XSI)** under the
[Apache License 2.0](./LICENSE). It builds on and references third-party work — DeepSpec (MIT),
SpecForge (Apache-2.0), DFlash (MIT), and NVIDIA Nemotron (NVIDIA Nemotron Open Model License).
See [`NOTICE`](./NOTICE) for the full attribution, including the required Nemotron license notice.

Nemotron models are *"Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License."*

## Contributing

Issues and PRs welcome — particularly benchmark results, corrected `target_layer_ids` for
specific Nemotron checkpoints, and fixes to the draft-config dimension mapping. If you train a
checkpoint with this recipe, please open an issue with your accepted-length numbers and setup.
