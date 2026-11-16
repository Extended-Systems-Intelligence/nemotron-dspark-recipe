#!/usr/bin/env python3
# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""Render collected runs into BENCHMARKS.md, between the harness markers.

    python scripts/render_benchmarks.py                 # rewrite BENCHMARKS.md in place
    python scripts/render_benchmarks.py --run <run_id>  # show a specific run in the tables
    python scripts/render_benchmarks.py --check         # CI: non-zero exit if out of date

Idempotent: it only touches text between the <!-- harness:*:start/end --> markers,
leaving all prose alone. With no committed runs it renders placeholder tables and
keeps the TEMPLATE status line, so running it on a fresh repo is a no-op.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results_harness as rh  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks-md", default=rh.BENCHMARKS_MD, help="path to BENCHMARKS.md")
    ap.add_argument("--benchmarks-dir", default=rh.BENCHMARKS_DIR, help="path to benchmarks/ dir")
    ap.add_argument("--run", help="run_id to show in the per-config tables (default: most recent)")
    ap.add_argument("--include-synthetic", action="store_true", help="include runs marked synthetic")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if BENCHMARKS.md is not already up to date")
    args = ap.parse_args()

    workloads = rh.load_json(rh.WORKLOADS_PATH)
    runs = rh.discover_runs(args.benchmarks_dir, include_synthetic=args.include_synthetic)
    if args.run and not any(r.run_id == args.run for r in runs):
        sys.stderr.write(f"run {args.run!r} not found under {args.benchmarks_dir} "
                         "(is it synthetic? pass --include-synthetic)\n")
        return 2

    with open(args.benchmarks_md, "r", encoding="utf-8") as fh:
        current = fh.read()

    rendered = rh.render_benchmarks_md(current, runs, workloads, selected_id=args.run)

    if args.check:
        if rendered != current:
            sys.stderr.write(f"{os.path.basename(args.benchmarks_md)} is out of date; "
                             "run scripts/render_benchmarks.py\n")
            return 1
        print(f"{os.path.basename(args.benchmarks_md)} is up to date")
        return 0

    if rendered == current:
        print(f"{os.path.basename(args.benchmarks_md)} already up to date ({len(runs)} run(s))")
        return 0
    with open(args.benchmarks_md, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    print(f"updated {os.path.basename(args.benchmarks_md)} from {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
