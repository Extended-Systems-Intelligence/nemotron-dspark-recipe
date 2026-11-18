# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shared library for the Nemotron DSpark benchmark results harness.

Stdlib only (Python 3.9+). Three jobs:

  1. Load + validate a run record (benchmarks/<run>/results.json) against
     benchmarks/schema.json, which stays the single source of truth.
  2. Normalize DeepSpec eval.py output into the acceptance_eval section.
  3. Render run records into the Markdown tables in BENCHMARKS.md, between
     idempotent ``<!-- harness:<name>:start/end -->`` markers.

The CLIs are scripts/collect_results.py and scripts/render_benchmarks.py.
A no-GPU end-to-end check is scripts/selftest_results_harness.py.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
BENCHMARKS_DIR = os.path.join(REPO_ROOT, "benchmarks")
SCHEMA_PATH = os.path.join(BENCHMARKS_DIR, "schema.json")
WORKLOADS_PATH = os.path.join(BENCHMARKS_DIR, "workloads.json")
BENCHMARKS_MD = os.path.join(REPO_ROOT, "BENCHMARKS.md")

EM_DASH = "—"

# Canonical table shapes, kept in lockstep with BENCHMARKS.md.
SINGLE_REQUEST_ROWS: List[Tuple[str, int]] = [
    ("0", 0),
    ("128K", 131072),
    ("256K", 262144),
    ("512K", 524288),
    ("1M", 1048576),
]
CONCURRENCY_ROWS: List[Tuple[str, int, str]] = [
    ("c1 / d0", 1, "d0"),
    ("c2 / d0", 2, "d0"),
    ("c4 / d0", 4, "d0"),
    ("c1 / d4K", 1, "d4K"),
]


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema validator (the subset benchmarks/schema.json uses).
# --------------------------------------------------------------------------- #
def _type_ok(value: Any, typ: str) -> bool:
    if typ == "object":
        return isinstance(value, dict)
    if typ == "array":
        return isinstance(value, list)
    if typ == "string":
        return isinstance(value, str)
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ == "null":
        return value is None
    return True


def _validate(value: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if "type" in schema:
        types = schema["type"]
        types = [types] if isinstance(types, str) else types
        if not any(_type_ok(value, t) for t in types):
            errors.append(f"{path}: expected type {schema['type']}, got {type(value).__name__}")
            return  # further checks assume the type held
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {value} >= exclusiveMaximum {schema['exclusiveMaximum']}")
    if "pattern" in schema and isinstance(value, str):
        if re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: {value!r} does not match /{schema['pattern']}/")
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required key '{req}'")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                _validate(value[key], sub, f"{path}.{key}", errors)
        ap = schema.get("additionalProperties", True)
        if isinstance(ap, dict):
            for key, val in value.items():
                if key not in props:
                    _validate(val, ap, f"{path}.{key}", errors)
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{i}]", errors)


def validate_run(record: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return a list of human-readable validation errors (empty == valid)."""
    if schema is None:
        schema = load_json(SCHEMA_PATH)
    errors: List[str] = []
    _validate(record, schema, "$", errors)
    return errors


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_workloads() -> Dict[str, Any]:
    return load_json(WORKLOADS_PATH)


@dataclass
class Run:
    record: Dict[str, Any]
    path: str

    @property
    def run_id(self) -> str:
        return str(self.record.get("run_id", os.path.basename(os.path.dirname(self.path))))

    @property
    def date(self) -> str:
        return str(self.record.get("date", ""))

    @property
    def synthetic(self) -> bool:
        return bool(self.record.get("synthetic", False))


def discover_runs(benchmarks_dir: str = BENCHMARKS_DIR, include_synthetic: bool = False,
                  validate: bool = False) -> List[Run]:
    """Load every benchmarks/<run>/results.json. Dirs whose name starts with
    '_' or '.' are skipped (reserved for examples/scratch).

    With validate=True, every discovered record is checked against the schema and
    a ValueError is raised (listing the artifact path and each error) if any record
    is invalid — so a hand-edited or directly contributed results.json cannot reach
    the renderer, and `render --check` cannot pass on invalid committed data."""
    runs: List[Run] = []
    if not os.path.isdir(benchmarks_dir):
        return runs
    schema = load_json(SCHEMA_PATH) if validate else None
    problems: List[str] = []
    for name in sorted(os.listdir(benchmarks_dir)):
        if name.startswith("_") or name.startswith("."):
            continue
        results = os.path.join(benchmarks_dir, name, "results.json")
        if not os.path.isfile(results):
            continue
        run = Run(record=load_json(results), path=results)
        if validate:
            errs = validate_run(run.record, schema)
            if errs:
                rel = os.path.relpath(results, REPO_ROOT)
                problems.extend(f"{rel}: {e}" for e in errs)
        if run.synthetic and not include_synthetic:
            continue
        runs.append(run)
    if problems:
        raise ValueError("invalid run record(s) found:\n  - " + "\n  - ".join(problems))
    runs.sort(key=lambda r: (r.date, r.run_id))
    return runs


def mean_metric(record: Dict[str, Any], key: str) -> Optional[float]:
    """Simple mean of a per-benchmark acceptance_eval metric across all benchmarks."""
    benches = (record.get("acceptance_eval") or {}).get("benchmarks") or {}
    vals = [float(b[key]) for b in benches.values()
            if isinstance(b, dict) and b.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def has_serving(record: Dict[str, Any]) -> bool:
    """True if the run carries any serving-throughput measurement."""
    s = record.get("serving") or {}
    if not isinstance(s, dict):
        return False
    if s.get("single_request") or s.get("concurrency"):
        return True
    headline = s.get("headline") or {}
    for v in headline.values() if isinstance(headline, dict) else []:
        if isinstance(v, dict) and (v.get("decode_tps_no_draft") is not None
                                    or v.get("decode_tps_draft") is not None):
            return True
    return False


# --------------------------------------------------------------------------- #
# DeepSpec eval.py adapter
# --------------------------------------------------------------------------- #
# DeepSpec's eval.py reports per-benchmark acceptance + accepted length. Its exact
# JSON field names vary by version, so we accept a set of common aliases and fail
# loudly (not silently) if none are found. Validate the mapping against your
# DeepSpec checkout on the first real run; override with an explicit normalized
# report if your version differs.
_ACCEPTANCE_KEYS = ("acceptance", "accept_rate", "acceptance_rate", "accept_ratio", "alpha")
_ACCLEN_KEYS = ("accepted_length", "avg_accept_len", "mean_accepted_tokens",
                "accept_length", "tau", "avg_accepted_length")
_SAMPLE_KEYS = ("samples", "num_samples", "n", "count")
_NAME_KEYS = ("benchmark", "task", "name", "dataset")


def _first(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _coerce_benchmark_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    acc = _first(raw, _ACCEPTANCE_KEYS)
    if acc is not None:
        acc = float(acc)
        # Accept percentages expressed as 0-100 and normalize to a fraction.
        out["acceptance"] = acc / 100.0 if acc > 1.0 else acc
    acclen = _first(raw, _ACCLEN_KEYS)
    if acclen is not None:
        out["accepted_length"] = float(acclen)
    samples = _first(raw, _SAMPLE_KEYS)
    if samples is not None:
        out["samples"] = int(samples)
    return out


def _has_metric(entry: Dict[str, Any]) -> bool:
    """A benchmark entry counts only if it carries acceptance or accepted_length
    (a samples count alone is not a measurement)."""
    return entry.get("acceptance") is not None or entry.get("accepted_length") is not None


def normalize_deepspec_eval(data: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize parsed DeepSpec eval output into {benchmark: {acceptance, accepted_length, samples}}.

    Accepts either a dict keyed by benchmark name, or a list of per-benchmark records
    each carrying a name field. Raises ValueError if no acceptance/accepted-length
    fields can be found at all (so a format mismatch is never silently dropped)."""
    benchmarks: Dict[str, Dict[str, Any]] = {}

    # Some DeepSpec versions wrap results under a top-level key.
    if isinstance(data, dict):
        for wrapper in ("results", "benchmarks", "per_benchmark", "eval"):
            if wrapper in data and isinstance(data[wrapper], (dict, list)):
                data = data[wrapper]
                break

    if isinstance(data, dict):
        for name, raw in data.items():
            if isinstance(raw, dict):
                entry = _coerce_benchmark_entry(raw)
                if _has_metric(entry):
                    benchmarks[str(name)] = entry
    elif isinstance(data, list):
        for raw in data:
            if not isinstance(raw, dict):
                continue
            name = _first(raw, _NAME_KEYS)
            if name is None:
                continue
            entry = _coerce_benchmark_entry(raw)
            if _has_metric(entry):
                benchmarks[str(name)] = entry
    else:
        raise ValueError("DeepSpec eval output must be a JSON object or array")

    if not benchmarks:
        raise ValueError(
            "no acceptance/accepted-length fields found in the DeepSpec eval output; "
            f"looked for acceptance keys {_ACCEPTANCE_KEYS} and length keys {_ACCLEN_KEYS}. "
            "Pass an already-normalized report, or extend the alias lists in results_harness.py."
        )
    return benchmarks


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
@dataclass
class BucketStat:
    acceptance: Optional[float] = None
    accepted_length: Optional[float] = None
    members: List[str] = field(default_factory=list)


def _weighted_mean(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """pairs of (value, weight). Sample-weighted only when EVERY pair has a
    positive weight; otherwise fall back to a simple mean, so a benchmark missing
    its sample count is never silently dropped from the average."""
    if not pairs:
        return None
    if all(w > 0 for _, w in pairs):
        total_w = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / total_w
    return sum(v for v, _ in pairs) / len(pairs)


def bucket_stats(record: Dict[str, Any], workloads: Dict[str, Any]) -> Dict[str, BucketStat]:
    """Roll per-benchmark acceptance_eval numbers up into the workload buckets."""
    benches = (record.get("acceptance_eval") or {}).get("benchmarks") or {}
    buckets = workloads["buckets"]
    out: Dict[str, BucketStat] = {}
    for bucket, members in buckets.items():
        acc_pairs: List[Tuple[float, float]] = []
        len_pairs: List[Tuple[float, float]] = []
        present: List[str] = []
        for m in members:
            b = benches.get(m)
            if not isinstance(b, dict):
                continue
            present.append(m)
            w = float(b.get("samples") or 0.0)
            if "acceptance" in b and b["acceptance"] is not None:
                acc_pairs.append((float(b["acceptance"]), w))
            if "accepted_length" in b and b["accepted_length"] is not None:
                len_pairs.append((float(b["accepted_length"]), w))
        out[bucket] = BucketStat(
            acceptance=_weighted_mean(acc_pairs),
            accepted_length=_weighted_mean(len_pairs),
            members=present,
        )
    return out


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def fmt_pct(x: Optional[float]) -> str:
    return EM_DASH if x is None else f"{x * 100:.0f}%"


def fmt_num(x: Optional[float], decimals: int = 1) -> str:
    return EM_DASH if x is None else f"{x:.{decimals}f}"


def fmt_speedup(no_draft: Optional[float], draft: Optional[float]) -> str:
    if not no_draft or draft is None:
        return EM_DASH
    return f"{draft / no_draft:.2f}×"


def _g(d: Any, *keys: str) -> Any:
    """Nested get that tolerates missing dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# --------------------------------------------------------------------------- #
# Table renderers (return the rows that go between the markers)
# --------------------------------------------------------------------------- #
def render_headline(record: Dict[str, Any], workloads: Dict[str, Any]) -> str:
    stats = bucket_stats(record, workloads)
    serving_headline = _g(record, "serving", "headline") or {}
    labels = workloads["bucket_labels"]
    lines = [
        "| Workload | Acceptance | Accepted len | Decode t/s (no draft) | Decode t/s (draft) | Speedup |",
        "|---|---|---|---|---|---|",
    ]
    for bucket in workloads["buckets"]:
        st = stats.get(bucket, BucketStat())
        sv = serving_headline.get(bucket, {}) if isinstance(serving_headline, dict) else {}
        nd = sv.get("decode_tps_no_draft")
        dr = sv.get("decode_tps_draft")
        lines.append(
            f"| {labels.get(bucket, bucket)} | {fmt_pct(st.acceptance)} | "
            f"{fmt_num(st.accepted_length, 2)} | {fmt_num(nd)} | {fmt_num(dr)} | "
            f"{fmt_speedup(nd, dr)} |"
        )
    return "\n".join(lines)


def _match_single_row(rows: List[Dict[str, Any]], target_tokens: int) -> Optional[Dict[str, Any]]:
    best = None
    best_delta = None
    for r in rows:
        ctx = r.get("context")
        if ctx is None:
            continue
        # The 0-context baseline row requires an exact zero match, so a run that
        # measured only 128K never back-fills the unmeasured 0 row (which would
        # publish a fake baseline and hide the real decode-rate drop). Non-zero
        # rows accept the nearest measurement within 5%.
        if target_tokens == 0:
            if int(ctx) != 0:
                continue
            delta = 0
        else:
            delta = abs(int(ctx) - target_tokens)
            if delta > 0.05 * target_tokens:
                continue
        if best_delta is None or delta < best_delta:
            best, best_delta = r, delta
    return best


def render_single_request(record: Dict[str, Any]) -> str:
    rows = _g(record, "serving", "single_request") or []
    lines = [
        "| Context | Prefill t/s | Decode t/s | TTFT | Acceptance |",
        "|---|---|---|---|---|",
    ]
    for label, tokens in SINGLE_REQUEST_ROWS:
        r = _match_single_row(rows, tokens) or {}
        ttft = r.get("ttft_ms")
        ttft_s = EM_DASH if ttft is None else f"{ttft:.0f} ms"
        lines.append(
            f"| {label} | {fmt_num(r.get('prefill_tps'))} | {fmt_num(r.get('decode_tps'))} | "
            f"{ttft_s} | {fmt_pct(r.get('acceptance'))} |"
        )
    return "\n".join(lines)


def render_concurrency(record: Dict[str, Any]) -> str:
    rows = _g(record, "serving", "concurrency") or []
    by_key = {(int(r.get("concurrency", -1)), str(r.get("depth", ""))): r for r in rows}
    lines = [
        "| Config | Depth | Prefill t/s | Decode t/s | TTFT |",
        "|---|---|---|---|---|",
    ]
    for label, conc, depth in CONCURRENCY_ROWS:
        r = by_key.get((conc, depth), {})
        ttft = r.get("ttft_ms")
        ttft_s = EM_DASH if ttft is None else f"{ttft:.0f} ms"
        cfg, dep = label.split(" / ")
        lines.append(
            f"| {cfg} | {dep} | {fmt_num(r.get('prefill_tps'))} | "
            f"{fmt_num(r.get('decode_tps'))} | {ttft_s} |"
        )
    return "\n".join(lines)


def render_quality(record: Dict[str, Any]) -> str:
    ge = record.get("greedy_equivalence") or {}
    if not ge.get("checked"):
        # Keep the descriptive placeholder until a run actually records the gate.
        ge_line = ("- **Greedy-equivalence check (required):** draft-on output == draft-off output "
                   "on a fixed prompt set\n  under greedy decoding. This must hold — if it doesn't, "
                   "the spec-decode wiring is wrong.")
    elif ge.get("passed"):
        n = ge.get("prompts")
        suffix = f" over {n} prompts" if n else ""
        ge_line = f"- **Greedy-equivalence check (required):** ✅ PASS{suffix}"
    else:
        ge_line = ("- **Greedy-equivalence check (required):** ❌ **FAIL — spec-decode wiring is "
                   "wrong; the throughput numbers above are not trustworthy.**")

    tool = _g(record, "quality", "tool_eval_bench", "score")
    tool_line = (f"- **tool-eval-bench:** {EM_DASH if tool is None else f'{tool:.0f}'} /100"
                 + (" (with category breakdown)" if tool is None else ""))
    cats = _g(record, "quality", "tool_eval_bench", "by_category")
    if tool is not None and isinstance(cats, dict) and cats:
        parts = ", ".join(f"{k} {v:.0f}" for k, v in cats.items())
        tool_line += f" ({parts})"

    inst = _g(record, "quality", "ifeval", "instruction_pct")
    prompt = _g(record, "quality", "ifeval", "prompt_pct")
    if_line = (f"- **IFEval:** instruction-level {fmt_pct(inst)} / prompt-level {fmt_pct(prompt)}")
    return "\n".join([ge_line, tool_line, if_line])


# Canonical placeholder for the serving run-configuration block when no run is
# selected. Kept verbatim in BENCHMARKS.md so render is a no-op in the no-run state.
CONFIG_PLACEHOLDER = "\n".join([
    "- **Target model / checkpoint:** `nvidia/NVIDIA-Nemotron-3-Nano` (stock)",
    "- **Draft:** trained DSpark draft — checkpoint + the config under `configs/`",
    "- **Hardware:** N× DGX Spark (GB10, 128 GB unified), TP=_, interconnect _",
    "- **Engine:** vLLM / TensorRT-LLM version _",
    "- **KV cache dtype:** _ (e.g. fp8 / nvfp4)",
    "- **Speculative config:** DSpark, `block_size`=_, `num_draft_layers`=_, `target_layer_ids`=_",
    "- **Limits:** `max-model-len`=_, `max-num-seqs`=_, `gpu-memory-utilization`=_",
    "- **Mode:** thinking on/off; warmed runs",
])


def render_config(record: Dict[str, Any]) -> str:
    c = record.get("config") or {}
    if not c:
        return CONFIG_PLACEHOLDER  # no run selected -> canonical placeholder
    spec = c.get("speculative", {}) or {}
    lim = c.get("limits", {}) or {}

    def val(x: Any) -> str:
        return EM_DASH if x in (None, "") else str(x)

    tli = spec.get("target_layer_ids")
    tli_s = EM_DASH if not tli else "[" + ", ".join(str(i) for i in tli) + "]"
    mode = c.get("thinking")
    mode_s = EM_DASH if mode is None else ("thinking on" if mode else "thinking off")
    warmed = c.get("warmed")
    warmed_s = "warmed" if warmed else ("cold" if warmed is False else EM_DASH)
    gmu = lim.get("gpu_memory_utilization")

    # The DSpark and EAGLE-3 drafters have different speculative parameters; show each
    # drafter's own fields so an EAGLE-3 run keeps its ttt_length / draft layers.
    if c.get("drafter") == "eagle3":
        spec_line = ("- **Speculative config:** eagle3, "
                     f"`ttt_length`={val(spec.get('ttt_length'))}, "
                     f"`draft_num_hidden_layers`={val(spec.get('draft_num_hidden_layers'))}, "
                     f"`target_layer_ids`={tli_s}")
    else:
        spec_line = (f"- **Speculative config:** {val(c.get('drafter'))}, "
                     f"`block_size`={val(spec.get('block_size'))}, "
                     f"`num_draft_layers`={val(spec.get('num_draft_layers'))}, "
                     f"`target_layer_ids`={tli_s}")

    return "\n".join([
        f"- **Target model / checkpoint:** `{val(c.get('target_model'))}`"
        + (f" ({c['target_checkpoint']})" if c.get("target_checkpoint") else ""),
        f"- **Draft:** {val(c.get('drafter'))} draft — `{val(c.get('draft_checkpoint'))}`"
        + (f" · config `{c['draft_config_file']}`" if c.get("draft_config_file") else ""),
        f"- **Hardware:** {val(c.get('hardware'))}, TP={val(c.get('tp'))}, "
        f"interconnect {val(c.get('interconnect'))}",
        f"- **Engine:** {val(c.get('engine'))}",
        f"- **KV cache dtype:** {val(c.get('kv_cache_dtype'))}",
        spec_line,
        f"- **Limits:** `max-model-len`={val(lim.get('max_model_len'))}, "
        f"`max-num-seqs`={val(lim.get('max_num_seqs'))}, "
        f"`gpu-memory-utilization`={val(gmu)}",
        f"- **Mode:** {mode_s}; {warmed_s} runs",
    ])


def render_runs_index(runs: List[Run], workloads: Dict[str, Any], selected_id: Optional[str]) -> str:
    if not runs:
        return "_No measured runs yet._"
    lines = [
        "| Run | Date | Drafter | Mean accepted len | Mean acceptance | Serving | Greedy-eq |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        rec = run.record
        sel = " ← serving tables below" if run.run_id == selected_id else ""
        drafter = _g(rec, "config", "drafter") or EM_DASH
        acclen = fmt_num(mean_metric(rec, "accepted_length"), 3)
        acc = fmt_pct(mean_metric(rec, "acceptance"))
        serving = "✅" if has_serving(rec) else EM_DASH
        ge = rec.get("greedy_equivalence") or {}
        ge_s = EM_DASH if not ge.get("checked") else ("✅" if ge.get("passed") else "❌")
        lines.append(
            f"| `{run.run_id}`{sel} | {run.date} | {drafter} | {acclen} | {acc} | {serving} | {ge_s} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Marker injection
# --------------------------------------------------------------------------- #
def _marker_pair(name: str) -> Tuple[str, str]:
    return f"<!-- harness:{name}:start -->", f"<!-- harness:{name}:end -->"


def inject_block(markdown: str, name: str, body: str) -> str:
    """Replace the content between the named start/end markers. Raises if absent."""
    start, end = _marker_pair(name)
    si = markdown.find(start)
    ei = markdown.find(end)
    if si == -1 or ei == -1 or ei < si:
        raise ValueError(f"markers for '{name}' not found (need {start} ... {end})")
    before = markdown[: si + len(start)]
    after = markdown[ei:]
    return f"{before}\n{body}\n{after}"


def _has_markers(markdown: str, name: str) -> bool:
    start, end = _marker_pair(name)
    return start in markdown and end in markdown


def render_benchmarks_md(markdown: str, runs: List[Run], workloads: Dict[str, Any],
                         selected_id: Optional[str] = None) -> str:
    """Return BENCHMARKS.md content with each present harness block re-rendered.

    Only blocks whose `<!-- harness:<name>:start/end -->` markers exist in the file
    are touched; hand-written prose (including a hand-authored status/proof section)
    is left alone. The serving tables and run configuration render the *selected*
    run — an explicit `selected_id`, else the most recent run that actually carries
    serving measurements, else none (placeholders). The runs index lists every run,
    so acceptance-only runs (e.g. the migrated proof runs) still appear there even
    when no run has serving numbers yet."""
    selected = None
    if runs:
        if selected_id:
            selected = next((r for r in runs if r.run_id == selected_id), None)
        if selected is None:
            serving_runs = [r for r in runs if has_serving(r.record)]
            selected = serving_runs[-1] if serving_runs else None

    sel_record = selected.record if selected else {"config": {}}
    sel_id = selected.run_id if selected else None

    blocks = {
        "status": lambda: ("**Status: TEMPLATE — no measured numbers yet.**" if not runs
                           else f"**Status: {len(runs)} measured run(s).**"),
        "runs": lambda: render_runs_index(runs, workloads, sel_id),
        "selected-heading": lambda: (f"### Run `{sel_id}`" if selected
                                     else "### Run _(none — placeholder)_"),
        "config": lambda: render_config(sel_record),
        "headline": lambda: render_headline(sel_record, workloads),
        "single-request": lambda: render_single_request(sel_record),
        "concurrency": lambda: render_concurrency(sel_record),
        "quality": lambda: render_quality(sel_record),
    }
    # Every present block is always re-rendered. When no run is selected the
    # serving blocks render their canonical placeholders (config -> CONFIG_PLACEHOLDER,
    # tables -> em-dashes, quality -> the descriptive gate text), so a run that is
    # deleted, marked synthetic, or loses its serving data can never leave stale
    # published numbers behind that `--check` would then accept.
    out = markdown
    for name, render in blocks.items():
        if _has_markers(out, name):
            out = inject_block(out, name, render())
    return out
