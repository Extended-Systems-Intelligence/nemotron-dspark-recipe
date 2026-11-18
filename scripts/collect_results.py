#!/usr/bin/env python3
# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""Collect one benchmark run into benchmarks/<run_id>/results.json.

Typical use, after `scripts/03_eval.sh` has produced a DeepSpec eval JSON:

    python scripts/collect_results.py \\
        --meta my_run.json \\
        --deepspec-eval eval_output.json

`--meta` is a JSON file you fill in with the run's config + any serving/quality
numbers (everything except the per-benchmark acceptance figures). Get a blank one:

    python scripts/collect_results.py --print-template > my_run.json

The acceptance figures come from DeepSpec's eval output via `--deepspec-eval`
(normalized by results_harness.normalize_deepspec_eval), or from an already-
normalized `{benchmark: {acceptance, accepted_length, samples}}` file via
`--acceptance-report`.

The result is validated against benchmarks/schema.json before it is written;
`.json` (not `.jsonl`) keeps it out of the repo's `.gitignore`, so commit it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results_harness as rh  # noqa: E402

TEMPLATE = {
    "run_id": "2026-01-01-dspark-nano",
    "date": "",
    "synthetic": False,
    "config": {
        "drafter": "dspark",
        "target_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "target_checkpoint": "",
        "draft_checkpoint": "~/checkpoints/deepspec/dspark_block8_nemotron_nano/step_latest",
        "draft_config_file": "configs/dspark_nemotron_nano.py",
        "hardware": "2x DGX Spark (GB10, 128 GB unified)",
        "tp": 1,
        "interconnect": "",
        "engine": "",
        "kv_cache_dtype": "",
        "speculative": {"block_size": 7, "num_draft_layers": 5, "target_layer_ids": []},
        "limits": {"max_model_len": 131072, "max_num_seqs": 32, "gpu_memory_utilization": 0.9},
        "thinking": False,
        "warmed": True,
        "notes": "",
    },
    "greedy_equivalence": {"checked": False, "passed": False, "prompts": 0, "notes": ""},
    "serving": {
        "headline": {
            "code": {"decode_tps_no_draft": None, "decode_tps_draft": None},
            "mixed": {"decode_tps_no_draft": None, "decode_tps_draft": None},
            "agentic": {"decode_tps_no_draft": None, "decode_tps_draft": None},
        },
        "single_request": [],
        "concurrency": [],
    },
    "quality": {
        "tool_eval_bench": {"score": None, "by_category": {}},
        "ifeval": {"instruction_pct": None, "prompt_pct": None},
    },
}


def _prune_nones(obj):
    """Drop keys whose value is None so placeholders don't fail numeric validation."""
    if isinstance(obj, dict):
        return {k: _prune_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_prune_nones(v) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", help="run metadata JSON (see --print-template)")
    ap.add_argument("--deepspec-eval", help="DeepSpec eval.py output JSON (fills acceptance_eval)")
    ap.add_argument("--acceptance-report", help="already-normalized {benchmark: {...}} JSON")
    ap.add_argument("--run-id", help="override run_id (also the output directory name)")
    ap.add_argument("--date", help="override date (YYYY-MM-DD); default today")
    ap.add_argument("--synthetic", action="store_true", help="mark this run as fabricated/example data")
    ap.add_argument("--out-dir", default=rh.BENCHMARKS_DIR, help="benchmarks dir (default: ./benchmarks)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results.json")
    ap.add_argument("--print-template", action="store_true", help="print a blank meta file and exit")
    args = ap.parse_args()

    if args.print_template:
        print(json.dumps(TEMPLATE, indent=2))
        return 0

    if not args.meta:
        ap.error("--meta is required (or use --print-template)")

    record = _prune_nones(rh.load_json(args.meta))
    record["schema_version"] = "1"
    if args.run_id:
        record["run_id"] = args.run_id
    if args.synthetic:
        record["synthetic"] = True
    record["date"] = args.date or record.get("date") or _dt.date.today().isoformat()

    # Acceptance numbers.
    benchmarks = None
    if args.deepspec_eval:
        benchmarks = rh.normalize_deepspec_eval(rh.load_json(args.deepspec_eval))
        source = f"deepspec-eval:{os.path.basename(args.deepspec_eval)}"
    elif args.acceptance_report:
        benchmarks = rh.load_json(args.acceptance_report)
        source = f"acceptance-report:{os.path.basename(args.acceptance_report)}"
    if benchmarks is not None:
        record.setdefault("acceptance_eval", {})
        record["acceptance_eval"]["source"] = source
        record["acceptance_eval"]["benchmarks"] = benchmarks

    errors = rh.validate_run(record)
    if errors:
        sys.stderr.write("results.json failed validation against benchmarks/schema.json:\n")
        for e in errors:
            sys.stderr.write(f"  - {e}\n")
        return 2

    run_id = record["run_id"]
    if run_id.startswith("_") or run_id.startswith("."):
        sys.stderr.write(f"run_id {run_id!r} is reserved (leading '_'/'.' dirs are skipped by the renderer)\n")
        return 2

    out_dir = os.path.join(args.out_dir, run_id)
    out_path = os.path.join(out_dir, "results.json")
    if os.path.exists(out_path) and not args.force:
        sys.stderr.write(f"{out_path} exists; pass --force to overwrite\n")
        return 1
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")

    print(f"wrote {out_path}")
    print("next: python scripts/render_benchmarks.py" + (" --include-synthetic" if record.get("synthetic") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
