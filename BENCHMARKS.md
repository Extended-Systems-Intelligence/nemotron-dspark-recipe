# Benchmark format & targets

<!-- harness:status:start -->
**Status: TEMPLATE — no measured numbers yet.**
<!-- harness:status:end -->

This repo is a reference scaffold; this file defines the reporting format the next phase fills
in once a Nemotron draft is trained. The table shapes follow the DGX Spark / GB10 community's
reporting convention (see [Format credit](#format-credit)) so results are directly comparable to
other Spark recipes.

**The tables below are generated — do not hand-edit them.** Measured runs live under
`benchmarks/<run_id>/results.json`; `scripts/render_benchmarks.py` renders them into the
`<!-- harness:* -->` blocks here. To add a run see
[`benchmarks/README.md`](benchmarks/README.md). The prose around the blocks is yours to edit.

---

## Why our table differs from a drop-in DSpark recipe

A serving recipe that uses a pre-trained drop-in draft can only report end throughput. We train
the draft, so the metrics that show whether the work paid off are **acceptance rate** and the
**speedup over the no-draft baseline** — reported per workload, because a workload-tuned draft
is the whole point. That's the headline table below; the rest mirrors the standard format.

---

## Runs

<!-- harness:runs:start -->
_No measured runs yet — this file is a template. Add one with `scripts/collect_results.py` (see [benchmarks/README.md](benchmarks/README.md)), then `scripts/render_benchmarks.py`._
<!-- harness:runs:end -->

<!-- harness:selected-heading:start -->
### Run _(none — placeholder)_
<!-- harness:selected-heading:end -->

### Run configuration

Numbers are only comparable when the setup matches — every table below belongs to this one run.

<!-- harness:config:start -->
- **Target model / checkpoint:** `—`
- **Draft:** — draft — `—`
- **Hardware:** —, TP=—, interconnect —
- **Engine:** —
- **KV cache dtype:** —
- **Speculative config:** —, `block_size`=—, `num_draft_layers`=—, `target_layer_ids`=—
- **Limits:** `max-model-len`=—, `max-num-seqs`=—, `gpu-memory-utilization`=—
- **Mode:** —; — runs
<!-- harness:config:end -->

---

### Headline — draft vs no-draft (the point of training a draft)

<!-- harness:headline:start -->
| Workload | Acceptance | Accepted len | Decode t/s (no draft) | Decode t/s (DSpark) | Speedup |
|---|---|---|---|---|---|
| Code | — | — | — | — | — |
| Mixed | — | — | — | — | — |
| Agentic / tool-calling | — | — | — | — | — |
<!-- harness:headline:end -->

> Acceptance is the engine's measured draft-acceptance fraction; accepted length is mean tokens
> committed per verify step. Report both — they explain the tok/s, and they're the numbers a
> trained draft can move that a drop-in draft cannot. The **Code** and **Mixed** rows roll up
> DeepSpec's eval benchmarks (see [`benchmarks/workloads.json`](benchmarks/workloads.json)); the
> **Agentic** row stays blank until you add a tool-calling benchmark, since none of DeepSpec's
> nine stock benchmarks is agentic.

---

### Single request (pp_ / tg_)

<!-- harness:single-request:start -->
| Context | Prefill t/s | Decode t/s | TTFT | Acceptance |
|---|---|---|---|---|
| 0 | — | — | — | — |
| 128K | — | — | — | — |
| 256K | — | — | — | — |
| 512K | — | — | — | — |
| 1M | — | — | — | — |
<!-- harness:single-request:end -->

Note the decode-t/s drop from 0 → max context (a small drop is the win — it means the KV path
holds up at length).

---

### Concurrency (pp_ / tg_)

<!-- harness:concurrency:start -->
| Config | Depth | Prefill t/s | Decode t/s | TTFT |
|---|---|---|---|---|
| c1 | d0 | — | — | — |
| c2 | d0 | — | — | — |
| c4 | d0 | — | — | — |
| c1 | d4K | — | — | — |
<!-- harness:concurrency:end -->

Report aggregate and per-stream tok/s. Note `max-model-len` is a per-request ceiling and
`max-num-seqs` a concurrency cap over a **shared** KV pool — the real limit is
`sum(live tokens) ≤ pool size`, not `concurrency × max-model-len`.

---

### Quality — verify lossless first, then score

Speculative decoding is **lossless with respect to the target**, so the only quality claim to
make is that it matches the no-draft baseline. Verify that, then report task scores for context:

<!-- harness:quality:start -->
- **Greedy-equivalence check (required):** — (not run)
- **tool-eval-bench:** — /100
- **IFEval:** instruction-level — / prompt-level —
<!-- harness:quality:end -->

The greedy-equivalence check — draft-on output == draft-off output on a fixed prompt set under
greedy decoding — **must** hold. If it doesn't, the spec-decode wiring is wrong and every speed
number above is meaningless; the renderer flags a failing gate in red.

---

## Methodology

- Warmed runs (discard the first), thinking-off unless the row says otherwise.
- Fixed prompt/generation sizes per table (state `pp`/`tg`).
- Acceptance and accepted length read from the serving engine's speculative-decoding stats.
- Long-context rows use a real retrieval/needle probe, not just a token count, so the number
  reflects correctness at length, not only speed.

---

## Format credit

The table structure here follows the DGX Spark / GB10 community's numbers-first benchmark
reporting convention — e.g. the DeepSeek-V4-Flash-DSpark and Aiden-recipe write-ups on the
NVIDIA Developer Forums. Thanks to that community for setting the bar; the extensions
(acceptance / accepted-length / baseline-vs-draft headline) are specific to a trained draft.
