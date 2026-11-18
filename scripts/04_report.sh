#!/usr/bin/env bash
# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stage 4: turn an eval run into committed benchmark results.
# Collects one run into benchmarks/<run_id>/results.json and re-renders BENCHMARKS.md.
# Run this from the recipe repo root (not the DeepSpec checkout).
#
# Usage:
#   bash scripts/04_report.sh <meta.json> <report.json> [extra collect_results.py args...]
#
#   <meta.json>    run metadata (see: scripts/collect_results.py --print-template)
#   <report.json>  a JSON acceptance report, keyed by benchmark:
#                  {"humaneval": {"accepted_length": 3.1, "acceptance": 0.78, "samples": 164}, ...}
#
# NOTE: DeepSpec's eval.py prints a human-readable table (and accept_length to TensorBoard) --
# it does NOT emit JSON. Transcribe its per-task acceptance / accepted-length into <report.json>
# first. The collector's --deepspec-eval adapter accepts that report (and common field aliases);
# if your DeepSpec build does emit JSON, pass that file instead.
#
# Example:
#   python scripts/collect_results.py --print-template > my_run.json && $EDITOR my_run.json
#   $EDITOR report.json                               # per-task numbers from the eval table
#   bash scripts/04_report.sh my_run.json report.json
#
# Then commit benchmarks/<run_id>/results.json and the updated BENCHMARKS.md.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

meta=$1
eval_json=$2
shift 2

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python "${here}/collect_results.py" --meta "${meta}" --deepspec-eval "${eval_json}" "$@"
python "${here}/render_benchmarks.py"

echo
echo "Done. Review BENCHMARKS.md, then commit it together with benchmarks/<run_id>/results.json."
