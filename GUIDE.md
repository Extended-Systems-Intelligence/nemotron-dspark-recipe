# Training a DSpark-class draft model for NVIDIA Nemotron

A step-by-step recipe for using DeepSeek's [DeepSpec](https://github.com/deepseek-ai/DeepSpec)
to train a speculative-decoding draft model against a **Nemotron** target, with notes for
running on NVIDIA cloud GPUs.

This guide assumes familiarity with speculative decoding at a high level. The one thing to
keep in mind throughout: the **draft model is a small transformer that consumes the target's
hidden states** — its *own* architecture does not have to match the target's. So even though
Nemotron-H is a hybrid Mamba-Transformer, the draft you train is an ordinary transformer. The
friction is **not** the draft's structure; it is extracting clean hidden states from the hybrid
*target* during the cache stage, and making sure the draft's input dimensions and normalization
line up with the hidden states you tap. That alignment is **not** automatic for a hybrid target
— validate it rather than assuming it. Keep that distinction front of mind — it explains most
of what follows.

---

## 0. Prerequisites

- A DeepSpec checkout with the Nemotron extension from [`scaffold/`](./scaffold/) applied
  (see [`scaffold/INTEGRATION.md`](./scaffold/INTEGRATION.md)).
- A **stock** Nemotron target checkpoint from Hugging Face (e.g. a Nemotron-Nano variant).
  Train the public artifact against a stock checkpoint, not a privately fine-tuned one — a
  draft and its target cache encode the target's output distribution.
- GPUs: DeepSpec's default configs assume **one node with 8 GPUs**. Draft models are small;
  the expensive resources are (a) serving the target during data-gen and (b) the target-cache
  on disk (see §2).
- An inference engine to serve the target during data-gen. DeepSpec ships an SGLang launcher;
  any OpenAI-compatible server that can host the target works.

### Running on NVIDIA cloud

If you do not have a local 8-GPU box, NVIDIA DGX Cloud / DGX Cloud Lepton provides multi-GPU
H100 nodes with the NeMo framework and Base Command Platform preinstalled. The pipeline maps
cleanly:

- **Target serving (data-gen):** stand the target up as an OpenAI/SGLang endpoint on one node.
- **Draft training:** an 8×H100 node runs `train.py` directly; `train.sh` spawns one worker
  per visible GPU. Note DeepSpec's launcher overloads `RANK`/`WORLD_SIZE` to mean
  *node_rank*/*node_count*, so `WORLD_SIZE=1` is a single-node run regardless of GPU count.
- **Cache storage:** provision a large scratch volume before §2 — see the storage warning.

---

## 1. Data preparation — prompts and regenerated answers

DeepSpec's `scripts/data/prepare_data.sh` runs three sub-steps. The recipe is unchanged for
Nemotron except the target model and chat template.

1. **Download and split prompts.** The reference set is
   [`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend)
   with a 5% held-out test split.
2. **Regenerate answers with the target.** Launch the target as a server, then run
   `generate_train_data.py`. The released DeepSpec checkpoints were trained in **non-thinking
   mode** (`--disable-thinking`). Nemotron reasoning checkpoints think by default — match the
   mode you intend to *serve*, and if you serve with reasoning on, plan to fine-tune the draft
   for that regime (DeepSpec's own README makes the same point).
3. **Build the target cache** (next section).

> **Reasoning-mode trap:** if you regenerate with thinking on but a tight `max_tokens`, the
> budget is consumed by the reasoning trace and you get truncated/empty answers. Give generous
> `max_tokens` or disable thinking for the data-gen pass.

---

## 2. Target-cache generation — the expensive, Nemotron-sensitive step

`prepare_target_cache.py` runs the **target** model over the regenerated data and stores the
hidden states the draft will learn from: the multi-layer feature taps named by
`target_layer_ids`, plus the last hidden state. This is where Nemotron-H's hybrid architecture
matters.

> **Storage warning:** the cache is large — DeepSpec cites **~38 TB** for the default
> `Qwen/Qwen3-4B` setting. Size scales with dataset size, sequence length, hidden dimension,
> and the number of tapped layers. Provision scratch accordingly, or reduce the dataset /
> `max_length` / number of taps for a smaller proof-of-concept run.

**What `target_layer_ids` means.** DSpark fuses features from several depths of the target
(EAGLE-3 style). DeepSpec's `gemma-4-12B` config taps `[5, 17, 29, 41, 46]` — a spread from
early to late layers. For Nemotron you must choose indices valid for *your* checkpoint's layer
count, and for **Nemotron-H** you should be deliberate about the hybrid layout (see
[`NEMOTRON-H-NOTES.md`](./NEMOTRON-H-NOTES.md)): `output_hidden_states` returns a hidden state
per decoder layer including the Mamba (SSM) layers, so pick taps that give you a good
low/mid/high spread and that land on or adjacent to the attention layers, which carry the
richest cross-token signal.

**The Nemotron-H landmine.** The cache step runs the target forward. On the hybrid checkpoints,
the stock generate/cache path is unreliable; run the forward pass with `use_cache=False` and
request hidden states explicitly. The exact failure modes and the working forward path are in
[`NEMOTRON-H-NOTES.md`](./NEMOTRON-H-NOTES.md). This is the single most likely place to lose a
day, so read those notes before launching a multi-hour cache job.

---

## 3. Draft training

Point `train.py` at the Nemotron config and the cache you just built:

```bash
python train.py \
    --config config/dspark/dspark_nemotron_nano.py \
    --opts "data.target_cache_path=${target_cache_dir}"
```

The trainer (`NemotronDSparkTrainer`) builds a Nemotron-shaped draft config from the target's
dimensions and trains it against the cached features. The training step consumes
`target_hidden_states`, `target_last_hidden_states`, and `loss_mask`, and optimizes the DSpark
loss (a weighted combination of cross-entropy, L1 on the predicted features, a decay across the
draft block, and the confidence head). The defaults in
[`configs/dspark_nemotron_nano.py`](./configs/dspark_nemotron_nano.py) mirror DeepSpec's own
DSpark configs; tune `local_batch_size` to your GPU memory (`--opts "train.local_batch_size=4"`).

Checkpoints land in `~/checkpoints/<project_name>/<exp_name>/step_*`.

---

## 4. Evaluation

`scripts/eval/eval.sh` runs the trained draft over DeepSpec's nine speculative-decoding
benchmarks (gsm8k, math500, aime25, humaneval, mbpp, livecodebench, mt-bench, alpaca,
arena-hard-v2). Set:

- `target_name_or_path` — the **same** Nemotron checkpoint the draft was trained against,
- `draft_name_or_path` — your trained draft checkpoint.

Read the result as **accepted length** (mean tokens accepted per verification step) first;
wall-clock speedup follows from accepted length and your serving setup. Compare against the
Eagle3 baseline you can train from the same repo for an apples-to-apples number on Nemotron.

> **Don't quote cross-setup numbers.** Per DeepSpec's own caveat, acceptance numbers are only
> comparable when the training setup matches. Report your config alongside any number you cite.

---

## 5. Serving

Once you have a draft, serve it with the target via an engine that supports speculative
decoding (e.g. TensorRT-LLM's EAGLE-3 / draft-target paths, or SGLang). NVIDIA already ships an
EAGLE-3 speculative-decoding path for Nemotron-Nano in TensorRT-LLM, which is a useful
reference for the serving wiring even though the draft you trained here is DSpark-class.
Measure tokens/sec before and after on your target hardware; lossless verification means
quality is unchanged, so the only thing to validate at serving time is the speedup.

---

## Appendix: how a new target family plugs into DeepSpec

For reference, supporting a new target in DeepSpec touches four extension points (all provided
for Nemotron in [`scaffold/`](./scaffold/)):

1. **Chat template** — registered in `deepspec/data/parser.py`'s `TEMPLATE_REGISTRY`.
2. **Draft modeling** — `deepspec/modeling/dspark/<family>/` with a `build_draft_config` that
   derives the draft transformer's dimensions from the target config.
3. **Trainer class** — `<Family>DSparkTrainer(Qwen3DSparkTrainer)` overriding `_build_draft_model`.
4. **Config** — `config/dspark/dspark_<family>_<size>.py` wiring the above together.
