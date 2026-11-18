# benchmarks/ — measured runs and the results harness

This directory holds measured benchmark runs for drafts trained with this recipe, plus the
machinery that turns them into the tables in [`../BENCHMARKS.md`](../BENCHMARKS.md). The two
`2026-07-11-proof-*` runs are the published proof-run measurements, migrated here as committed
artifacts (accepted-length only — no serving throughput yet).

```
benchmarks/
├── schema.json                    # the results.json contract (JSON Schema; the validator reads this)
├── workloads.json                 # which eval benchmarks roll up into Code / Mixed / Agentic
├── 2026-07-11-proof-dspark-*/     # DSpark proof run (real measurements)
├── 2026-07-11-proof-eagle3-*/     # EAGLE-3 proof run (real measurements)
├── <run_id>/results.json          # one measured run (you add these)
└── _example-run/                  # synthetic data for the selftest — '_' dirs are never rendered
```

The harness itself lives in [`../scripts/`](../scripts/):

| Script | Role |
|---|---|
| `results_harness.py` | shared library: validation, DeepSpec adapter, aggregation, rendering |
| `collect_results.py` | eval output + run metadata → `benchmarks/<run_id>/results.json` |
| `render_benchmarks.py` | all runs → the tables in `BENCHMARKS.md` |
| `04_report.sh` | convenience wrapper: collect + render in one step |
| `selftest_results_harness.py` | no-GPU end-to-end test of all of the above |

## Adding a run

1. **Evaluate** your trained draft (`../scripts/03_eval.sh`). DeepSpec's `eval.py` prints a
   **human-readable table** (and writes `accept_length` scalars to TensorBoard if a
   `tensorboard_dir` is set) — it does **not** emit JSON. Transcribe its per-task acceptance and
   accepted-length into a small JSON report, e.g. `report.json`:

   ```json
   {
     "humaneval": {"accepted_length": 3.1, "acceptance": 0.78, "samples": 164},
     "gsm8k":     {"accepted_length": 2.7, "acceptance": 0.61, "samples": 1319}
   }
   ```

2. **Describe the run.** Generate a blank metadata file and fill it in — hardware, engine, the
   speculative config, the greedy-equivalence result, and any serving-throughput numbers you
   measured separately:

   ```bash
   python ../scripts/collect_results.py --print-template > my_run.json
   $EDITOR my_run.json
   ```

3. **Collect** the run record (validated against `schema.json` before it is written):

   ```bash
   python ../scripts/collect_results.py --meta my_run.json --deepspec-eval report.json
   ```

4. **Render** it into `BENCHMARKS.md`:

   ```bash
   python ../scripts/render_benchmarks.py
   ```

   Then commit `benchmarks/<run_id>/results.json` **and** the updated `BENCHMARKS.md`.

Steps 3–4 together are `bash ../scripts/04_report.sh my_run.json report.json`.

## What goes where

- **Acceptance + accepted length.** DeepSpec's `eval.py` prints these as a table (not JSON), so
  you write them into a `{benchmark: {...}}` JSON report and pass it via `--deepspec-eval` (the
  `normalize_deepspec_eval` adapter accepts common field-name aliases and normalizes 0–100
  acceptance to a fraction) or `--acceptance-report` (taken as-is). If a future DeepSpec build
  does emit JSON, feed that directly; **validate the field mapping on your first real run** and
  extend the alias lists in `results_harness.py` if needed. Either way, a report with no
  recognized acceptance/accepted-length metric is rejected rather than silently dropped.
- **Serving throughput** (prefill/decode t/s, TTFT, the headline decode-t/s columns) is *not*
  produced by `eval.py` — it comes from serving the draft on your engine. Put those numbers in
  the `serving` section of the metadata file. Anything you leave out renders as an em-dash.
- **Greedy-equivalence** is the required gate. Record it under `greedy_equivalence`; a run whose
  gate is `checked: true, passed: false` is flagged loudly, because the speed numbers are then
  not trustworthy.

## Workload buckets

`BENCHMARKS.md`'s headline table has three rows — Code, Mixed, Agentic — and
[`workloads.json`](workloads.json) maps DeepSpec's nine benchmarks onto them. **Agentic is empty
by default**: none of the nine stock benchmarks is a tool-calling workload, so that row stays
blank until you add an agentic/tool-eval benchmark key (and list it under `agentic` in
`workloads.json`). Edit `workloads.json` to change the rollup; it's data, not code.

## Honesty rules

- `results.json` is `.json`, not `.jsonl`, so it escapes the repo's `.gitignore` and is meant to
  be committed — a measured run is a first-class artifact.
- Never set `synthetic: false` on numbers you did not measure. The `_example-run/` fixture is
  `synthetic: true` and lives in a `_`-prefixed directory precisely so it can never leak into
  `BENCHMARKS.md`.
- One table = one configuration. `BENCHMARKS.md` shows a single selected run (the most recent, or
  `--run <id>`); the **Runs** index lists them all. Don't merge configs into one table.
