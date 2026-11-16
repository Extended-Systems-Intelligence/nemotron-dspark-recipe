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


def discover_runs(benchmarks_dir: str = BENCHMARKS_DIR, include_synthetic: bool = False) -> List[Run]:
    """Load every benchmarks/<run>/results.json. Dirs whose name starts with
    '_' or '.' are skipped (reserved for examples/scratch)."""
    runs: List[Run] = []
    if not os.path.isdir(benchmarks_dir):
        return runs
    for name in sorted(os.listdir(benchmarks_dir)):
        if name.startswith("_") or name.startswith("."):
            continue
        results = os.path.join(benchmarks_dir, name, "results.json")
        if not os.path.isfile(results):
            continue
        run = Run(record=load_json(results), path=results)
        if run.synthetic and not include_synthetic:
            continue
        runs.append(run)
    runs.sort(key=lambda r: (r.date, r.run_id))
    return runs


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
                if entry:
                    benchmarks[str(name)] = entry
    elif isinstance(data, list):
        for raw in data:
            if not isinstance(raw, dict):
                continue
            name = _first(raw, _NAME_KEYS)
            if name is None:
                continue
            entry = _coerce_benchmark_entry(raw)
            if entry:
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
    """pairs of (value, weight). Equal weights if all weights are 0."""
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return sum(v for v, _ in pairs) / len(pairs)
    return sum(v * w for v, w in pairs) / total_w


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
        "| Workload | Acceptance | Accepted len | Decode t/s (no draft) | Decode t/s (DSpark) | Speedup |",
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
        # exact, or within 5% for the non-zero rows
        if target_tokens == 0:
            delta = abs(int(ctx))
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
        ge_line = f"- **Greedy-equivalence check (required):** {EM_DASH} (not run)"
    elif ge.get("passed"):
        n = ge.get("prompts")
        suffix = f" over {n} prompts" if n else ""
        ge_line = f"- **Greedy-equivalence check (required):** ✅ PASS{suffix}"
    else:
        ge_line = ("- **Greedy-equivalence check (required):** ❌ **FAIL — spec-decode wiring is "
                   "wrong; the throughput numbers above are not trustworthy.**")

    tool = _g(record, "quality", "tool_eval_bench", "score")
    tool_line = f"- **tool-eval-bench:** {EM_DASH if tool is None else f'{tool:.0f}'} /100"
    cats = _g(record, "quality", "tool_eval_bench", "by_category")
    if isinstance(cats, dict) and cats:
        parts = ", ".join(f"{k} {v:.0f}" for k, v in cats.items())
        tool_line += f" ({parts})"

    inst = _g(record, "quality", "ifeval", "instruction_pct")
    prompt = _g(record, "quality", "ifeval", "prompt_pct")
    if_line = (f"- **IFEval:** instruction-level {fmt_pct(inst)} / prompt-level {fmt_pct(prompt)}")
    return "\n".join([ge_line, tool_line, if_line])


def render_config(record: Dict[str, Any]) -> str:
    c = record.get("config", {})
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
    return "\n".join([
        f"- **Target model / checkpoint:** `{val(c.get('target_model'))}`"
        + (f" ({c['target_checkpoint']})" if c.get("target_checkpoint") else ""),
        f"- **Draft:** {val(c.get('drafter'))} draft — `{val(c.get('draft_checkpoint'))}`"
        + (f" · config `{c['draft_config_file']}`" if c.get("draft_config_file") else ""),
        f"- **Hardware:** {val(c.get('hardware'))}, TP={val(c.get('tp'))}, "
        f"interconnect {val(c.get('interconnect'))}",
        f"- **Engine:** {val(c.get('engine'))}",
        f"- **KV cache dtype:** {val(c.get('kv_cache_dtype'))}",
        f"- **Speculative config:** {val(c.get('drafter'))}, "
        f"`block_size`={val(spec.get('block_size'))}, "
        f"`num_draft_layers`={val(spec.get('num_draft_layers'))}, "
        f"`target_layer_ids`={tli_s}",
        f"- **Limits:** `max-model-len`={val(lim.get('max_model_len'))}, "
        f"`max-num-seqs`={val(lim.get('max_num_seqs'))}, "
        f"`gpu-memory-utilization`={val(gmu)}",
        f"- **Mode:** {mode_s}; {warmed_s} runs",
    ])


def render_runs_index(runs: List[Run], workloads: Dict[str, Any], selected_id: Optional[str]) -> str:
    if not runs:
        return ("_No measured runs yet — this file is a template. Add one with "
                "`scripts/collect_results.py` (see [benchmarks/README.md](benchmarks/README.md)), "
                "then `scripts/render_benchmarks.py`._")
    lines = [
        "| Run | Date | Drafter | Code speedup | Mixed speedup | Greedy-eq |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        rec = run.record
        sel = " ← shown below" if run.run_id == selected_id else ""
        drafter = _g(rec, "config", "drafter") or EM_DASH
        sh = _g(rec, "serving", "headline") or {}
        code_sp = fmt_speedup(_g(sh, "code", "decode_tps_no_draft"), _g(sh, "code", "decode_tps_draft"))
        mixed_sp = fmt_speedup(_g(sh, "mixed", "decode_tps_no_draft"), _g(sh, "mixed", "decode_tps_draft"))
        ge = rec.get("greedy_equivalence") or {}
        ge_s = EM_DASH if not ge.get("checked") else ("✅" if ge.get("passed") else "❌")
        lines.append(
            f"| `{run.run_id}`{sel} | {run.date} | {drafter} | {code_sp} | {mixed_sp} | {ge_s} |"
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


def render_benchmarks_md(markdown: str, runs: List[Run], workloads: Dict[str, Any],
                         selected_id: Optional[str] = None) -> str:
    """Return BENCHMARKS.md content with every harness block re-rendered.

    Selected run (for the per-config tables) = the one whose run_id == selected_id,
    else the most recent run. With no runs, blocks render as placeholders."""
    selected = None
    if runs:
        if selected_id:
            selected = next((r for r in runs if r.run_id == selected_id), None)
        selected = selected or runs[-1]

    sel_record = selected.record if selected else {"config": {}}
    sel_id = selected.run_id if selected else None

    status = ("**Status: TEMPLATE — no measured numbers yet.**"
              if not runs else
              f"**Status: {len(runs)} measured run(s).** Tables below show run "
              f"`{sel_id}`; re-render with `--run <id>` to show another.")

    blocks = {
        "status": status,
        "runs": render_runs_index(runs, workloads, sel_id),
        "selected-heading": (f"### Run `{sel_id}`" if selected
                             else "### Run _(none — placeholder)_"),
        "config": render_config(sel_record),
        "headline": render_headline(sel_record, workloads),
        "single-request": render_single_request(sel_record),
        "concurrency": render_concurrency(sel_record),
        "quality": render_quality(sel_record),
    }
    out = markdown
    for name, body in blocks.items():
        out = inject_block(out, name, body)
    return out
