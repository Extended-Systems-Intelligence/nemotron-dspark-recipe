# Benchmarks

**Status: first measured results — proof-of-recipe, not production.** The recipe now trains a
Nemotron draft end-to-end on 2×GB10 and we have a head-to-head acceptance measurement. Read the
banner below before reading the numbers: these are **small, deliberately-short proof drafts**, and
what they establish is that the recipe *works* and *which draft architecture wins at equal budget* —
not a production-grade draft.

The serving-throughput tables further down are the reporting format for a full run and are **not yet
filled** — we haven't run a production-scale training job or an end-to-end serving benchmark. We're
not going to put invented numbers in them.

---

## ⚠️ Read this before the numbers

Both drafts were trained for **74 optimizer steps (2 epochs)** over a **~19K-sample** target-hidden-
state cache — a proof run to validate the recipe and settle the architecture question, not a trained
model. Accepted length lands near **1.1–1.25**, which is low by design: a production draft targets
**~2.5–4**. So:

- These numbers answer **"does the recipe train end-to-end, and which architecture accepts more?"** —
  not "is this draft good yet?" (It isn't; it's on the order of 1–2% of a production training budget.)
- The comparison is at **equal, small budget**. Whether the winner keeps its lead at full training is
  a separate question — architectures can scale differently. Confirm with a scaling sweep before
  committing a large run to one architecture.
- Sample sizes are small (10–15 prompts per task). Trust the **direction** (it's consistent across all
  nine task types), not the third decimal.

---

## Head-to-head: DSpark vs EAGLE-3 draft (proof run)

Same Nemotron-Nano target, same 19K cache, same 74-step / 2-epoch budget, same eval config. Metric is
**mean accepted length** — tokens committed per target verification (`1 + mean accepted draft tokens`;
higher is better; both drafts propose 7 tokens per step).

| Task | DSpark | EAGLE-3 | DSpark advantage |
|---|---|---|---|
| gsm8k | **1.247** | 1.140 | +0.107 |
| math500 | **1.251** | 1.143 | +0.108 |
| aime25 | **1.238** | 1.131 | +0.107 |
| humaneval | **1.237** | 1.147 | +0.090 |
| mbpp | **1.24** | 1.171 | +0.07 |
| livecodebench | **1.24** | 1.128 | +0.11 |
| mt-bench | **1.26** | 1.161 | +0.10 |
| alpaca | **1.24** | 1.169 | +0.07 |
| arena-hard-v2 | **1.25** | 1.123 | +0.13 |
| **mean** | **≈1.245** | **≈1.146** | **+0.10 (~9%)** |

**DSpark wins every task.** The driver is first-token acceptance: DSpark's draft correctly predicts
the target's next token **~21%** of the time versus EAGLE-3's **~11%** — nearly 2×. Nine independent
task types all agreeing is what makes the direction trustworthy despite the small samples.

**Takeaway:** for a Nemotron-Nano target at this budget, the **semi-autoregressive DSpark draft** is
the stronger architecture. That's the decision to carry into a full training run.

---

## What produced these numbers

- **Target:** `nvidia/NVIDIA-Nemotron-3-Nano` class, bf16, stock (no adapters, no modifications).
- **Drafts:** one DSpark draft (`block_size=8`, `draft_num_hidden_layers` per config) and one EAGLE-3
  draft (`ttt_length=7`, 1 draft layer), each trained from the same cache with the configs under
  [`configs/`](configs/).
- **Training:** 74 steps / 2 epochs, `global_batch_size=512`, on 2×GB10 (the flex→dense-sdpa-mask fix
  and the step-time-decay mitigations from [`GB10-TRAINING-NOTES.md`](GB10-TRAINING-NOTES.md) are
  required to get here).
- **Cache:** ~19K target-generation samples, hidden states at the tapped `target_layer_ids`.
- **Eval:** DeepSpec acceptance eval, nine task types (gsm8k, math500, aime25, humaneval, mbpp,
  livecodebench, mt-bench, alpaca, arena-hard-v2), 10–15 prompts each, 128 max new tokens. Acceptance
  and accepted-length read from the eval's speculative stats. See
  [`GB10-TRAINING-NOTES.md` §D](GB10-TRAINING-NOTES.md) for the two eval-harness fixes this needs on a
  Nemotron-H target.

---

## Serving throughput (format for a full run — not yet measured)

Everything below is the reporting shape a **production** run should fill: end-to-end decode tok/s with
and without the draft, at length and under concurrency. A trained draft's serving speedup tracks its
accepted length, so with proof-level acceptance (~1.25) the speedup would be small (~1.25×) and isn't
worth publishing as a headline — that's why these are blank until a real training run lands.

### Run configuration (fill for every result set)

- **Target model / checkpoint:** `nvidia/NVIDIA-Nemotron-3-Nano` (stock)
- **Draft:** trained DSpark draft — checkpoint + the config under `configs/`
- **Hardware:** N× DGX Spark (GB10, 128 GB unified), TP=_, interconnect _
- **Engine:** vLLM / TensorRT-LLM version _
- **KV cache dtype:** _ (e.g. fp8 / nvfp4)
- **Speculative config:** DSpark, `block_size`=_, `num_draft_layers`=_, `target_layer_ids`=_
- **Limits:** `max-model-len`=_, `max-num-seqs`=_, `gpu-memory-utilization`=_
- **Mode:** thinking on/off; warmed runs

### Headline — draft vs no-draft

| Workload | Acceptance | Accepted len | Decode t/s (no draft) | Decode t/s (DSpark) | Speedup |
|---|---|---|---|---|---|
| Code | — | — | — | — | — |
| Mixed | — | — | — | — | — |
| Agentic / tool-calling | — | — | — | — | — |

### Single request (pp_ / tg_)

| Context | Prefill t/s | Decode t/s | TTFT | Acceptance |
|---|---|---|---|---|
| 0 | — | — | — | — |
| 128K | — | — | — | — |
| 256K | — | — | — | — |
| 512K | — | — | — | — |
| 1M | — | — | — | — |

### Concurrency (pp_ / tg_)

| Config | Depth | Prefill t/s | Decode t/s | TTFT |
|---|---|---|---|---|
| c1 | d0 | — | — | — |
| c2 | d0 | — | — | — |
| c4 | d0 | — | — | — |
| c1 | d4K | — | — | — |

Report aggregate and per-stream tok/s. `max-model-len` is a per-request ceiling and `max-num-seqs` a
concurrency cap over a **shared** KV pool — the real limit is `sum(live tokens) ≤ pool size`.

---

## Quality — verify lossless first, then score

Speculative decoding is **lossless with respect to the target**, so the only quality claim to make is
that it matches the no-draft baseline. Verify that, then report task scores for context:

- **Greedy-equivalence check (required):** draft-on output == draft-off output on a fixed prompt set
  under greedy decoding. This must hold — if it doesn't, the spec-decode wiring is wrong.
- **tool-eval-bench:** _ /100 (with category breakdown)
- **IFEval:** instruction-level _% / prompt-level _%

---

## Methodology

- Warmed runs (discard the first), thinking-off unless the row says otherwise.
- Fixed prompt/generation sizes per table (state `pp`/`tg`).
- Acceptance and accepted length read from the serving engine's speculative-decoding stats.
- Long-context rows use a real retrieval/needle probe, not just a token count.
- Drop measured runs under `benchmarks/<date>-<config>/` and summarize them here.

---

## Format credit

The serving-table structure follows the DGX Spark / GB10 community's numbers-first benchmark reporting
convention — e.g. the DeepSeek-V4-Flash-DSpark and Aiden-recipe write-ups on the NVIDIA Developer
Forums. Thanks to that community for setting the bar; the acceptance / accepted-length / head-to-head
extensions are specific to a trained draft.
