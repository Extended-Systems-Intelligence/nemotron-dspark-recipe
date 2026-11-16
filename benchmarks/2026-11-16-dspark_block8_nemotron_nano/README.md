# 2026-11-16 — DSpark vs EAGLE-3 proof run (Nemotron-Nano, 2×GB10)

The landing zone for one measured benchmark run, following the convention in
[`../../BENCHMARKS.md`](../../BENCHMARKS.md) (*"Drop measured runs under `benchmarks/<date>-<config>/`
and summarize them here"*). The prose summary lives in `BENCHMARKS.md`; the machine-readable data
and the raw evidence live here.

## What's here

| File | What it is |
|---|---|
| [`results.csv`](results.csv) | Per-task mean accepted length for both drafts, as data. Kept in sync with the head-to-head table in `BENCHMARKS.md`. |
| [`run_config.md`](run_config.md) | Reproducibility record — target, both draft configs, training budget, cache, eval harness + required GB10/Nemotron-H fixes. |
| `raw/` | **Raw evidence — drop the untouched job output here.** Empty until the logs are added (see below). |

## Drop raw evidence into `raw/`

The summary is only as trustworthy as the evidence behind it. When you have the job output, add it
verbatim (do not edit the logs):

- `raw/eval.dspark.log` — stdout of `03_eval.sh` for the DSpark draft
- `raw/eval.eagle3.log` — stdout of `03_eval.sh` for the EAGLE-3 draft
- `raw/first_token_acceptance.txt` — the ~21% (DSpark) vs ~11% (EAGLE-3) first-token numbers and
  where they were read from
- anything else that backs a number (per-task prompt lists, engine version banner, `step_*` used)

Then regenerate / verify the table:

```bash
# Render the head-to-head markdown table from results.csv (deterministic, matches BENCHMARKS.md):
python scripts/parse_eval_results.py --csv benchmarks/2026-11-16-dspark_block8_nemotron_nano/results.csv

# Or rebuild results.csv straight from the raw eval logs, then render:
python scripts/parse_eval_results.py \
    --dspark-log benchmarks/2026-11-16-dspark_block8_nemotron_nano/raw/eval.dspark.log \
    --eagle3-log benchmarks/2026-11-16-dspark_block8_nemotron_nano/raw/eval.eagle3.log \
    --write-csv benchmarks/2026-11-16-dspark_block8_nemotron_nano/results.csv
```

See [`scripts/parse_eval_results.py`](../../scripts/parse_eval_results.py) `--help` for the exact
log-line format it expects (and adjust the regex there to match your `eval.py` revision).

## The result, in one line

DSpark wins **every** task — mean accepted length **≈1.245 vs ≈1.146** (**+~9%**), driven by
first-token acceptance **~21% vs ~11%**. This is a **proof run** (74 steps, ~1–2% of a production
budget), so the finding is *"the recipe trains end-to-end and the DSpark architecture accepts more
at equal budget,"* **not** *"this draft is production-ready."* Full context and caveats:
[`../../BENCHMARKS.md`](../../BENCHMARKS.md).
