# Benchmark format & targets

**Status: TEMPLATE — no measured numbers yet.** This repo is a reference scaffold; this file
defines the reporting format the next phase will fill in once a Nemotron draft is trained. The
table shapes follow the DGX Spark / GB10 community's reporting convention (see
[Format credit](#format-credit)) so results are directly comparable to other Spark recipes.

Drop measured runs under `benchmarks/<date>-<config>/` and summarize them here.

---

## Why our table differs from a drop-in DSpark recipe

A serving recipe that uses a pre-trained drop-in draft can only report end throughput. We train
the draft, so the metrics that show whether the work paid off are **acceptance rate** and the
**speedup over the no-draft baseline** — reported per workload, because a workload-tuned draft
is the whole point. That's the headline table below; the rest mirrors the standard format.

---

## Run configuration

Fill this in for every result set — numbers are only comparable when the setup matches.

- **Target model / checkpoint:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` (stock) — _set yours_
- **Draft:** trained DSpark draft — checkpoint + the config under `configs/`
- **Hardware:** N× DGX Spark (GB10, 128 GB unified), TP=_, interconnect _
- **Engine:** vLLM / TensorRT-LLM version _
- **KV cache dtype:** _ (e.g. fp8 / nvfp4)
- **Speculative config:** DSpark, `block_size`=_, `num_draft_layers`=_, `target_layer_ids`=_
- **Limits:** `max-model-len`=_, `max-num-seqs`=_, `gpu-memory-utilization`=_
- **Mode:** thinking on/off; warmed runs

---

## Headline — draft vs no-draft (the point of training a draft)

| Workload | Acceptance | Accepted len | Decode t/s (no draft) | Decode t/s (DSpark) | Speedup |
|---|---|---|---|---|---|
| Code | — | — | — | — | — |
| Mixed | — | — | — | — | — |
| Agentic / tool-calling | — | — | — | — | — |

> Acceptance is the engine's measured draft-acceptance fraction; accepted length is mean tokens
> committed per verify step. Report both — they explain the tok/s, and they're the numbers a
> trained draft can move that a drop-in draft cannot.

---

## Single request (pp_ / tg_)

| Context | Prefill t/s | Decode t/s | TTFT | Acceptance |
|---|---|---|---|---|
| 0 | — | — | — | — |
| 128K | — | — | — | — |
| 256K | — | — | — | — |
| 512K | — | — | — | — |
| 1M | — | — | — | — |

Note the decode-t/s drop from 0 → max context (a small drop is the win — it means the KV path
holds up at length).

---

## Concurrency (pp_ / tg_)

| Config | Depth | Prefill t/s | Decode t/s | TTFT |
|---|---|---|---|---|
| c1 | d0 | — | — | — |
| c2 | d0 | — | — | — |
| c4 | d0 | — | — | — |
| c1 | d4K | — | — | — |

Report aggregate and per-stream tok/s. Note `max-model-len` is a per-request ceiling and
`max-num-seqs` a concurrency cap over a **shared** KV pool — the real limit is
`sum(live tokens) ≤ pool size`, not `concurrency × max-model-len`.

---

## Quality — verify lossless first, then score

Speculative decoding is **lossless with respect to the target**, so the only quality claim to
make is that it matches the no-draft baseline. Verify that, then report task scores for context:

- **Greedy-equivalence check (required):** draft-on output == draft-off output on a fixed prompt
  set under greedy decoding. This must hold — if it doesn't, the spec-decode wiring is wrong.
- **tool-eval-bench:** _ /100 (with category breakdown)
- **IFEval:** instruction-level _% / prompt-level _%

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
