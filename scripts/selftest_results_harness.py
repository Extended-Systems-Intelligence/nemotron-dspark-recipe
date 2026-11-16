#!/usr/bin/env python3
# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""No-GPU end-to-end test for the benchmark results harness.

    python scripts/selftest_results_harness.py

Exercises, with no weights or GPU:
  - benchmarks/schema.json validates the synthetic example run,
  - the DeepSpec eval adapter normalizes several realistic output shapes,
  - collect_results.py produces a schema-valid results.json from a fake eval,
  - render_benchmarks.py fills the BENCHMARKS.md tables with the right cells,
  - a failing greedy-equivalence gate is flagged, and rendering is idempotent.

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


def test_schema_validates_example() -> None:
    rec = rh.load_json(EXAMPLE)
    errs = rh.validate_run(rec)
    check(errs == [], f"schema validates the synthetic example (errors={errs})")
    # A deliberately broken record must fail.
    bad = copy.deepcopy(rec)
    bad["schema_version"] = "99"
    bad["config"].pop("drafter")
    errs = rh.validate_run(bad)
    check(len(errs) >= 2, f"schema rejects a broken record (errors={len(errs)})")


def test_deepspec_adapter() -> None:
    # Dict keyed by benchmark, percentage-style acceptance (0-100) must normalize.
    d = {"humaneval": {"acceptance_rate": 72.0, "avg_accept_len": 3.4, "num_samples": 164}}
    out = rh.normalize_deepspec_eval(d)
    check(abs(out["humaneval"]["acceptance"] - 0.72) < 1e-9, "adapter normalizes 0-100 acceptance to fraction")
    check(out["humaneval"]["accepted_length"] == 3.4, "adapter reads accepted_length alias")
    # List-of-records with a wrapper key and name field.
    lst = {"results": [{"task": "mbpp", "alpha": 0.70, "tau": 3.2, "n": 500}]}
    out = rh.normalize_deepspec_eval(lst)
    check("mbpp" in out and out["mbpp"]["acceptance"] == 0.70, "adapter reads list form + alpha/tau aliases")
    # Unrecognized shape must raise, not silently drop.
    raised = False
    try:
        rh.normalize_deepspec_eval({"humaneval": {"unknown_field": 1}})
    except ValueError:
        raised = True
    check(raised, "adapter raises on output with no acceptance/length fields")


def test_collect_cli(tmp: str) -> None:
    # Fake DeepSpec eval output + meta -> results.json via the CLI.
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


def test_render_populates(tmp: str) -> None:
    # Copy the example under a normal (non-underscore) name so discover picks it up.
    bench_dir = os.path.join(tmp, "render", "benchmarks")
    os.makedirs(os.path.join(bench_dir, "example-run"), exist_ok=True)
    rec = rh.load_json(EXAMPLE)
    rec["run_id"] = "example-run"
    with open(os.path.join(bench_dir, "example-run", "results.json"), "w") as fh:
        json.dump(rec, fh)
    md = os.path.join(tmp, "BENCHMARKS.md")
    with open(rh.BENCHMARKS_MD) as fh:
        template = fh.read()
    with open(md, "w") as fh:
        fh.write(template)

    r = subprocess.run(
        [PY, RENDER, "--benchmarks-md", md, "--benchmarks-dir", bench_dir, "--include-synthetic"],
        capture_output=True, text=True,
    )
    check(r.returncode == 0, f"render_benchmarks.py exits 0 (stderr={r.stderr.strip()})")
    out = open(md).read()
    for needle in ["2.57×", "2.05×", "| Code | 69% | 3.16", "`example-run`",
                   "1 measured run", "✅ PASS", "61 /100"]:
        check(needle in out, f"rendered BENCHMARKS.md contains {needle!r}")
    # Agentic row stays blank (no member benchmarks).
    check("| Agentic / tool-calling | — | — | — | — | — |" in out, "Agentic row stays blank")

    # Idempotent: a second render is a no-op.
    r2 = subprocess.run(
        [PY, RENDER, "--benchmarks-md", md, "--benchmarks-dir", bench_dir, "--include-synthetic", "--check"],
        capture_output=True, text=True,
    )
    check(r2.returncode == 0, "render is idempotent (--check passes after render)")


def test_greedy_fail_flag() -> None:
    rec = rh.load_json(EXAMPLE)
    rec["greedy_equivalence"] = {"checked": True, "passed": False}
    out = rh.render_quality(rec)
    check("FAIL" in out, "failing greedy-equivalence gate is flagged in the quality block")


def main() -> int:
    print("results-harness selftest\n")
    with tempfile.TemporaryDirectory() as tmp:
        test_schema_validates_example()
        test_deepspec_adapter()
        test_collect_cli(tmp)
        test_render_populates(tmp)
        test_greedy_fail_flag()
    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
