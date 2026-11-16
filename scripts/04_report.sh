#!/usr/bin/env bash
# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stage 4: turn an eval run into committed benchmark results.
# Collects one run into benchmarks/<run_id>/results.json and re-renders BENCHMARKS.md.
#
# Usage:
#   bash scripts/04_report.sh <meta.json> <deepspec_eval.json> [extra collect_results.py args...]
#
#   <meta.json>           run metadata (see: scripts/collect_results.py --print-template)
#   <deepspec_eval.json>  DeepSpec eval.py output captured from scripts/03_eval.sh
#
# Example:
#   python scripts/collect_results.py --print-template > my_run.json && $EDITOR my_run.json
#   bash scripts/03_eval.sh > eval_output.json        # capture your eval JSON
#   bash scripts/04_report.sh my_run.json eval_output.json
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
