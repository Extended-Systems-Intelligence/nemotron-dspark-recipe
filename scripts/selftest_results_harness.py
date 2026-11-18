#!/usr/bin/env python3
# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""No-GPU end-to-end test for the benchmark results harness.

    python scripts/selftest_results_harness.py

Exercises, with no weights or GPU:
  - benchmarks/schema.json validates the synthetic example and the committed runs,
    and rejects out-of-range metrics (e.g. acceptance > 1, negative accepted length),
  - the DeepSpec eval adapter normalizes realistic shapes and rejects samples-only,
  - bucket aggregation falls back to a simple mean when sample counts are incomplete,
  - the 0-context serving row requires an exact match (no back-fill from 128K),
  - collect_results.py produces a schema-valid results.json from a fake eval,
  - render_benchmarks.py fills the serving tables + runs index for a serving run,
    uses a drafter-neutral header, is idempotent, and rejects an invalid record,
  - a failing greedy-equivalence gate is flagged.

This validates the harness plumbing only — not training or benchmark correctness.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results_harness as rh  # noqa: E402

PY = sys.executable
COLLECT = os.path.join(rh.HERE, "collect_results.py")
RENDER = os.path.join(rh.HERE, "render_benchmarks.py")
EXAMPLE = os.path.join(rh.BENCHMARKS_DIR, "_example-run", "results.json")

_failures = []


def check(cond: bool, msg: str) -> None:
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _failures.append(msg)


def test_schema_validation() -> None:
    rec = rh.load_json(EXAMPLE)
    check(rh.validate_run(rec) == [], "schema validates the synthetic example")

    # Committed real runs must validate too.
    for name in ("2026-07-11-proof-dspark-block8-nano", "2026-07-11-proof-eagle3-ttt7-nano"):
        p = os.path.join(rh.BENCHMARKS_DIR, name, "results.json")
        if os.path.isfile(p):
            check(rh.validate_run(rh.load_json(p)) == [], f"schema validates committed run {name}")

    # Structural rejection.
    bad = copy.deepcopy(rec)
    bad["schema_version"] = "99"
    bad["config"].pop("drafter")
    check(len(rh.validate_run(bad)) >= 2, "schema rejects a structurally broken record")

    # Numeric-bound rejection: acceptance > 1 and negative accepted_length.
    oob = copy.deepcopy(rec)
    oob["acceptance_eval"]["benchmarks"]["humaneval"]["acceptance"] = 72
    oob["acceptance_eval"]["benchmarks"]["humaneval"]["accepted_length"] = -1
    errs = rh.validate_run(oob)
    check(any("maximum" in e for e in errs), "schema rejects acceptance > 1 (would render 7200%)")
    check(any("minimum" in e for e in errs), "schema rejects negative accepted_length")


def test_deepspec_adapter() -> None:
    d = {"humaneval": {"acceptance_rate": 72.0, "avg_accept_len": 3.4, "num_samples": 164}}
    out = rh.normalize_deepspec_eval(d)
    check(abs(out["humaneval"]["acceptance"] - 0.72) < 1e-9, "adapter normalizes 0-100 acceptance to fraction")
    check(out["humaneval"]["accepted_length"] == 3.4, "adapter reads accepted_length alias")

    lst = {"results": [{"task": "mbpp", "alpha": 0.70, "tau": 3.2, "n": 500}]}
    out = rh.normalize_deepspec_eval(lst)
    check("mbpp" in out and out["mbpp"]["acceptance"] == 0.70, "adapter reads list form + alpha/tau aliases")

    # No recognized metric at all -> raise.
    for bad in ({"humaneval": {"unknown_field": 1}}, {"humaneval": {"num_samples": 164}}):
        raised = False
        try:
            rh.normalize_deepspec_eval(bad)
        except ValueError:
            raised = True
        check(raised, f"adapter raises when no acceptance/length metric is present ({list(bad['humaneval'])})")


def test_weighted_mean_fallback() -> None:
    # One benchmark has a sample count, the other does not: must be a simple mean (0.5),
    # not a weighted mean that drops the unweighted measurement (0.2).
    wl = {"buckets": {"code": ["a", "b"], "mixed": [], "agentic": []},
          "bucket_labels": {"code": "Code", "mixed": "Mixed", "agentic": "Agentic"}}
    rec = {"acceptance_eval": {"benchmarks": {
        "a": {"acceptance": 0.2, "samples": 100},
        "b": {"acceptance": 0.8},
    }}}
    acc = rh.bucket_stats(rec, wl)["code"].acceptance
    check(acc is not None and abs(acc - 0.5) < 1e-9, f"incomplete sample weights fall back to simple mean (got {acc})")


def test_zero_context_exact() -> None:
    rec = {"serving": {"single_request": [
        {"context": 131072, "prefill_tps": 1800.0, "decode_tps": 95.0, "ttft_ms": 900.0, "acceptance": 0.68},
    ]}}
    table = rh.render_single_request(rec)
    zero_row = [ln for ln in table.splitlines() if ln.startswith("| 0 |")][0]
    check(zero_row == "| 0 | — | — | — | — |", f"0-context row stays em-dash when only 128K measured ({zero_row})")
    row_128k = [ln for ln in table.splitlines() if ln.startswith("| 128K |")][0]
    check("95.0" in row_128k, "128K row is filled from the 128K measurement")


def test_collect_cli(tmp: str) -> None:
    eval_json = os.path.join(tmp, "eval.json")
    with open(eval_json, "w") as fh:
        json.dump({
            "humaneval": {"acceptance": 0.72, "accepted_length": 3.4, "samples": 164},
            "gsm8k": {"acceptance": 0.61, "accepted_length": 2.7, "samples": 1319},
        }, fh)
    meta = copy.deepcopy(rh.load_json(EXAMPLE))
    meta.pop("acceptance_eval", None)
    meta["run_id"] = "selftest-run"
    meta_json = os.path.join(tmp, "meta.json")
    with open(meta_json, "w") as fh:
        json.dump(meta, fh)

    bench_dir = os.path.join(tmp, "collect", "benchmarks")
    r = subprocess.run(
        [PY, COLLECT, "--meta", meta_json, "--deepspec-eval", eval_json,
         "--out-dir", bench_dir, "--synthetic"],
        capture_output=True, text=True,
    )
    check(r.returncode == 0, f"collect_results.py exits 0 (stderr={r.stderr.strip()})")
    produced = os.path.join(bench_dir, "selftest-run", "results.json")
    check(os.path.isfile(produced), "collect wrote results.json")
    if os.path.isfile(produced):
        rec = rh.load_json(produced)
        check(rh.validate_run(rec) == [], "collected results.json is schema-valid")
        check(rh.bucket_stats(rec, rh.load_workloads())["code"].acceptance is not None,
              "collected run has a Code-bucket acceptance")


def _setup_render_dir(tmp: str, subdir: str, rec: dict):
    bench_dir = os.path.join(tmp, subdir, "benchmarks")
    os.makedirs(os.path.join(bench_dir, rec["run_id"]), exist_ok=True)
    with open(os.path.join(bench_dir, rec["run_id"], "results.json"), "w") as fh:
        json.dump(rec, fh)
    md = os.path.join(tmp, subdir, "BENCHMARKS.md")
    with open(rh.BENCHMARKS_MD) as fh:
        template = fh.read()
    with open(md, "w") as fh:
        fh.write(template)
    return bench_dir, md


def test_render_populates(tmp: str) -> None:
    rec = rh.load_json(EXAMPLE)  # synthetic, carries serving data
    rec["run_id"] = "example-run"
    bench_dir, md = _setup_render_dir(tmp, "render", rec)

    r = subprocess.run(
        [PY, RENDER, "--benchmarks-md", md, "--benchmarks-dir", bench_dir, "--include-synthetic"],
        capture_output=True, text=True,
    )
    check(r.returncode == 0, f"render_benchmarks.py exits 0 (stderr={r.stderr.strip()})")
    out = open(md).read()
    for needle in ["2.57×", "| Code | 69% | 3.16", "`example-run`", "✅ PASS", "61 /100",
                   "Decode t/s (draft)", "Mean accepted len"]:
        check(needle in out, f"rendered BENCHMARKS.md contains {needle!r}")
    check("Decode t/s (DSpark)" not in out, "headline header is drafter-neutral (no hardcoded DSpark)")
    check("| Agentic / tool-calling | — | — | — | — | — |" in out, "Agentic row stays blank")

    r2 = subprocess.run(
        [PY, RENDER, "--benchmarks-md", md, "--benchmarks-dir", bench_dir, "--include-synthetic", "--check"],
        capture_output=True, text=True,
    )
    check(r2.returncode == 0, "render is idempotent (--check passes after render)")


def test_render_rejects_invalid(tmp: str) -> None:
    rec = rh.load_json(EXAMPLE)
    rec["run_id"] = "broken-run"
    rec["synthetic"] = False
    rec["schema_version"] = "99"          # invalid
    rec["config"].pop("drafter")          # missing required
    bench_dir, md = _setup_render_dir(tmp, "reject", rec)
    r = subprocess.run([PY, RENDER, "--benchmarks-md", md, "--benchmarks-dir", bench_dir, "--check"],
                       capture_output=True, text=True)
    check(r.returncode != 0, "render rejects an invalid committed record with nonzero exit")
    check("broken-run" in r.stderr, "render error names the offending artifact path")


def test_eagle3_config_fields() -> None:
    rec = {"config": {"drafter": "eagle3",
                      "speculative": {"ttt_length": 7, "draft_num_hidden_layers": 1}}}
    out = rh.render_config(rec)
    check("`ttt_length`=7" in out, "EAGLE-3 config renders ttt_length")
    check("`draft_num_hidden_layers`=1" in out, "EAGLE-3 config renders draft_num_hidden_layers")
    check("block_size" not in out, "EAGLE-3 config omits DSpark-only block_size")


def test_stale_serving_reset(tmp: str) -> None:
    # A serving run fills the tables; once it is gone, re-render must reset them to
    # placeholders rather than leave stale published numbers.
    rec = rh.load_json(EXAMPLE)
    rec["run_id"] = "temp-serving"
    bench_dir, md = _setup_render_dir(tmp, "stale", rec)
    subprocess.run([PY, RENDER, "--benchmarks-md", md, "--benchmarks-dir", bench_dir,
                    "--include-synthetic"], capture_output=True, text=True)
    check("108.0" in open(md).read(), "serving run fills the decode-t/s cells")

    os.remove(os.path.join(bench_dir, "temp-serving", "results.json"))
    r = subprocess.run([PY, RENDER, "--benchmarks-md", md, "--benchmarks-dir", bench_dir,
                        "--include-synthetic"], capture_output=True, text=True)
    out = open(md).read()
    check(r.returncode == 0, "re-render after run removal succeeds")
    check("108.0" not in out, "stale serving numbers are cleared when the run is gone")
    check("`block_size`=_" in out, "config block resets to the canonical placeholder")


def test_greedy_fail_flag() -> None:
    rec = rh.load_json(EXAMPLE)
    rec["greedy_equivalence"] = {"checked": True, "passed": False}
    out = rh.render_quality(rec)
    check("FAIL" in out, "failing greedy-equivalence gate is flagged in the quality block")


def test_repo_render_up_to_date() -> None:
    # The committed BENCHMARKS.md must already be rendered from the committed runs.
    r = subprocess.run([PY, RENDER, "--check"], capture_output=True, text=True)
    check(r.returncode == 0, f"committed BENCHMARKS.md is up to date (stderr={r.stderr.strip()})")


def main() -> int:
    print("results-harness selftest\n")
    with tempfile.TemporaryDirectory() as tmp:
        test_schema_validation()
        test_deepspec_adapter()
        test_weighted_mean_fallback()
        test_zero_context_exact()
        test_collect_cli(tmp)
        test_render_populates(tmp)
        test_render_rejects_invalid(tmp)
        test_eagle3_config_fields()
        test_stale_serving_reset(tmp)
        test_greedy_fail_flag()
        test_repo_render_up_to_date()
    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
