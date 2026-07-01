#!/usr/bin/env python3
"""
Aggregate per-rep JSONL output from sweep_tg.sh / sweep_pp.sh into a
paper-grade summary.

Input format (one JSON per line):
    {"engine": ..., "variant": ..., "pp": ..., "tg": ..., "rep": ...,
     "prompt_tokens": ..., "gen_tokens": ...,
     "e2e_ms": ..., ["prefill_ms": ..., "decode_ms": ...]}

Output:
  * CSV per (engine, variant, pp, tg): n, e2e median/mean, request_gen
    throughput from e2e_ms, optional direct decode timing, and derived
    decode timing from a cross-tg fit.
  * Markdown table: one row per cell with median ± IQR/2 shorthand.

Derivation for engines that only report e2e_ms: fit a line
    e2e_ms(tg) = prefill_ms + decode_ms_per_tok * tg
across the tg values available for that (engine, variant, pp). Intercept is
prefill, slope is per-token decode latency. Requires ≥2 distinct tg values;
otherwise derived fields are left blank.

request_gen_tps is generated_tokens / e2e_ms. decode_direct_* is populated
only when an engine emits decode_ms with comparable same-request semantics.
decode_fit_* is derived from e2e_ms across tg values and is labeled as such.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOFLINE_GPU_BANDWIDTH_GBPS = 1792.0
DEFAULT_ROOFLINE_BANDWIDTH_FRACTION = 0.85
DEFAULT_ROOFLINE_PARAM_COUNT = 4.02e9
DEFAULT_ROOFLINE_DTYPE_BYTES = 2.0
DEFAULT_ROOFLINE_LAYERS = 36
DEFAULT_ROOFLINE_KV_HEADS = 8
DEFAULT_ROOFLINE_HEAD_DIM = 128
# RTX 5090 f16 tensor-core SoL used only for the tiny pp=18 prefill term.
# The tg roofline is dominated by decode bandwidth.
DEFAULT_ROOFLINE_PREFILL_TFLOPS = 417.792


def read_model_roofline_defaults(model_dir):
    """Read model size and KV shape from the same HF model used by the sweep."""
    model_dir = Path(model_dir)
    defaults = {}

    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        values = {
            "layers": cfg.get("num_hidden_layers"),
            "kv_heads": cfg.get("num_key_value_heads"),
            "head_dim": cfg.get("head_dim"),
        }
        if values["head_dim"] is None:
            hidden_size = cfg.get("hidden_size")
            attn_heads = cfg.get("num_attention_heads")
            if hidden_size is not None and attn_heads:
                values["head_dim"] = hidden_size // attn_heads
        defaults.update({k: v for k, v in values.items() if v is not None})

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        total_size = index.get("metadata", {}).get("total_size")
        if total_size is not None:
            defaults["weight_bytes"] = float(total_size)

    return defaults


def resolve_roofline_values(args):
    model = {}
    if args.roofline_model_dir is not None:
        model = read_model_roofline_defaults(args.roofline_model_dir)

    dtype_bytes = args.roofline_dtype_bytes
    weight_bytes = args.roofline_weight_bytes
    if weight_bytes is None:
        weight_bytes = model.get("weight_bytes")

    param_count = args.roofline_param_count
    if param_count is None and weight_bytes is not None:
        param_count = weight_bytes / dtype_bytes
    if param_count is None:
        param_count = DEFAULT_ROOFLINE_PARAM_COUNT
    if weight_bytes is None:
        weight_bytes = param_count * dtype_bytes

    layers = args.roofline_layers
    if layers is None:
        layers = model.get("layers", DEFAULT_ROOFLINE_LAYERS)
    kv_heads = args.roofline_kv_heads
    if kv_heads is None:
        kv_heads = model.get("kv_heads", DEFAULT_ROOFLINE_KV_HEADS)
    head_dim = args.roofline_head_dim
    if head_dim is None:
        head_dim = model.get("head_dim", DEFAULT_ROOFLINE_HEAD_DIM)

    effective_bw_gbps = args.roofline_effective_bandwidth_gbps
    if effective_bw_gbps is None:
        effective_bw_gbps = (
            args.roofline_gpu_bandwidth_gbps
            * args.roofline_bandwidth_fraction
        )

    return {
        "effective_bw_gbps": effective_bw_gbps,
        "nominal_bw_gbps": args.roofline_gpu_bandwidth_gbps,
        "bandwidth_fraction": args.roofline_bandwidth_fraction,
        "weight_bytes": weight_bytes,
        "param_count": param_count,
        "dtype_bytes": dtype_bytes,
        "layers": int(layers),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "model_dir": str(args.roofline_model_dir) if args.roofline_model_dir else None,
        "uses_model_defaults": bool(model),
    }


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    # Linear interpolation (same as numpy default).
    idx = q * (len(xs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])


def fit_line(xs, ys):
    """Least-squares slope/intercept of y = intercept + slope*x."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return intercept, slope


def request_roofline(pp, tg, args):
    """Ideal request-level generated-token roofline for batch-1 decode.

    The bound uses alpha=0. Decode is modeled as HBM streaming of FP16 model
    weights plus average KV-cache reads. Prefill is included to align with
    request_gen_tps, but is tiny for the pp=18 tg sweep.
    """
    if not args.request_roofline:
        return {}

    resolved = args.roofline_resolved
    params = resolved["param_count"]
    dtype_bytes = resolved["dtype_bytes"]
    weight_bytes = resolved["weight_bytes"]
    kv_step_bytes = (
        2
        * resolved["layers"]
        * resolved["kv_heads"]
        * resolved["head_dim"]
        * dtype_bytes
    )

    avg_context_tokens = pp + ((tg - 1) / 2.0) if tg > 0 else pp
    avg_decode_payload_bytes = weight_bytes + kv_step_bytes * avg_context_tokens
    decode_bytes = tg * avg_decode_payload_bytes
    effective_bw_bytes_s = resolved["effective_bw_gbps"] * 1e9
    nominal_bw_bytes_s = resolved["nominal_bw_gbps"] * 1e9
    decode_s = (
        decode_bytes / effective_bw_bytes_s
        if effective_bw_bytes_s > 0 else float("nan")
    )
    nominal_decode_s = (
        decode_bytes / nominal_bw_bytes_s
        if nominal_bw_bytes_s > 0 else float("nan")
    )

    prefill_s = 0.0
    if args.roofline_prefill_tflops > 0:
        prefill_s = (2 * params * pp) / (args.roofline_prefill_tflops * 1e12)

    total_s = prefill_s + decode_s
    nominal_total_s = prefill_s + nominal_decode_s
    request_tps = tg / total_s if total_s > 0 else float("nan")
    nominal_request_tps = (
        tg / nominal_total_s if nominal_total_s > 0 else float("nan")
    )
    avg_kv_gb = (
        (decode_bytes / tg - weight_bytes) / 1e9
        if tg > 0 else float("nan")
    )

    return {
        "request_effective_roofline_tps": request_tps,
        "request_nominal_roofline_tps": nominal_request_tps,
        "request_roofline_prefill_ms": prefill_s * 1000.0,
        "request_effective_roofline_decode_ms": decode_s * 1000.0,
        "request_nominal_roofline_decode_ms": nominal_decode_s * 1000.0,
        "request_effective_bandwidth_gbps": resolved["effective_bw_gbps"],
        "request_nominal_bandwidth_gbps": resolved["nominal_bw_gbps"],
        "request_roofline_weight_gb": weight_bytes / 1e9,
        "request_roofline_avg_kv_gb": avg_kv_gb,
        "request_roofline_param_count": params,
    }


def fmt_opt(x, spec):
    if x is None:
        return "—"
    if isinstance(x, float) and x != x:
        return "—"
    return format(x, spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--request-roofline", action="store_true",
                        help="Add an ideal request-level HBM roofline.")
    parser.add_argument("--roofline-model-dir", type=Path, default=None,
                        help="HF model dir used to infer weight bytes and KV shape.")
    parser.add_argument("--roofline-weight-bytes", type=float, default=None,
                        help="Override model weight bytes read per decode token.")
    parser.add_argument("--roofline-gpu-bandwidth-gbps", type=float,
                        default=DEFAULT_ROOFLINE_GPU_BANDWIDTH_GBPS)
    parser.add_argument("--roofline-bandwidth-fraction", type=float,
                        default=DEFAULT_ROOFLINE_BANDWIDTH_FRACTION)
    parser.add_argument("--roofline-effective-bandwidth-gbps", type=float,
                        default=None,
                        help="Override GPU bandwidth * fraction.")
    parser.add_argument("--roofline-param-count", type=float,
                        default=None)
    parser.add_argument("--roofline-dtype-bytes", type=float,
                        default=DEFAULT_ROOFLINE_DTYPE_BYTES)
    parser.add_argument("--roofline-layers", type=int,
                        default=None)
    parser.add_argument("--roofline-kv-heads", type=int,
                        default=None)
    parser.add_argument("--roofline-head-dim", type=int,
                        default=None)
    parser.add_argument("--roofline-prefill-tflops", type=float,
                        default=DEFAULT_ROOFLINE_PREFILL_TFLOPS)
    args = parser.parse_args()
    args.roofline_resolved = resolve_roofline_values(args)

    # Group per (engine, variant, pp, tg).
    cells = defaultdict(list)
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec["engine"], rec["variant"], rec["pp"], rec["tg"])
            cells[key].append(rec)

    # Per-cell summary.
    cell_rows = []
    for (engine, variant, pp, tg), records in sorted(cells.items()):
        e2e = [r["e2e_ms"] for r in records if "e2e_ms" in r]
        if not e2e:
            continue
        prompt_tokens = records[0].get("prompt_tokens", 0)
        gen_tokens = records[0].get("gen_tokens", tg)
        med = statistics.median(e2e)
        mean = statistics.mean(e2e)
        p25 = percentile(e2e, 0.25)
        p75 = percentile(e2e, 0.75)
        iqr = p75 - p25
        total_tok = prompt_tokens + gen_tokens
        valid_e2e = [ms for ms in e2e if ms > 0]
        e2e_tps = total_tok / (med / 1000.0) if med > 0 else float("nan")
        e2e_mean_tps = statistics.mean(
            [total_tok / (ms / 1000.0) for ms in valid_e2e]
        ) if valid_e2e else float("nan")
        request_gen_tps = gen_tokens / (med / 1000.0) if med > 0 else float("nan")
        request_gen_mean_tps = statistics.mean(
            [gen_tokens / (ms / 1000.0) for ms in valid_e2e]
        ) if valid_e2e else float("nan")
        roofline = request_roofline(pp, tg, args)
        request_effective_roofline_tps = roofline.get(
            "request_effective_roofline_tps"
        )
        request_effective_roofline_fraction = None
        if request_effective_roofline_tps and request_effective_roofline_tps > 0:
            request_effective_roofline_fraction = (
                request_gen_tps / request_effective_roofline_tps
            )
        request_nominal_roofline_tps = roofline.get("request_nominal_roofline_tps")
        request_nominal_roofline_fraction = None
        if request_nominal_roofline_tps and request_nominal_roofline_tps > 0:
            request_nominal_roofline_fraction = (
                request_gen_tps / request_nominal_roofline_tps
            )

        # Direct phase timers are optional and kept separate from e2e metrics.
        prefill_direct_med = None
        prefill_direct_tps = None
        decode_direct_med = None
        decode_direct_tps = None
        prefills = [r["prefill_ms"] for r in records if "prefill_ms" in r]
        decodes = [r["decode_ms"] for r in records if "decode_ms" in r]
        if prefills:
            prefill_direct_med = statistics.median(prefills)
            if prefill_direct_med > 0:
                prefill_direct_tps = prompt_tokens / (prefill_direct_med / 1000.0)
        if decodes:
            decode_direct_med = statistics.median(decodes)
            if decode_direct_med > 0:
                decode_direct_tps = gen_tokens / (decode_direct_med / 1000.0)

        cell_rows.append({
            "engine": engine,
            "variant": variant,
            "pp": pp,
            "tg": tg,
            "n": len(e2e),
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "e2e_median_ms": med,
            "e2e_mean_ms": mean,
            "e2e_p25_ms": p25,
            "e2e_p75_ms": p75,
            "e2e_iqr_ms": iqr,
            "e2e_median_tps": e2e_tps,
            "e2e_mean_tps": e2e_mean_tps,
            "request_gen_median_tps": request_gen_tps,
            "request_gen_mean_tps": request_gen_mean_tps,
            "request_nominal_roofline_tps": request_nominal_roofline_tps,
            "request_nominal_roofline_fraction": request_nominal_roofline_fraction,
            "request_effective_roofline_tps": request_effective_roofline_tps,
            "request_effective_roofline_fraction": request_effective_roofline_fraction,
            "request_roofline_prefill_ms": roofline.get("request_roofline_prefill_ms"),
            "request_effective_roofline_decode_ms": roofline.get(
                "request_effective_roofline_decode_ms"
            ),
            "request_nominal_roofline_decode_ms": roofline.get(
                "request_nominal_roofline_decode_ms"
            ),
            "request_effective_bandwidth_gbps": roofline.get(
                "request_effective_bandwidth_gbps"
            ),
            "request_nominal_bandwidth_gbps": roofline.get(
                "request_nominal_bandwidth_gbps"
            ),
            "request_roofline_weight_gb": roofline.get("request_roofline_weight_gb"),
            "request_roofline_avg_kv_gb": roofline.get("request_roofline_avg_kv_gb"),
            "request_roofline_param_count": roofline.get("request_roofline_param_count"),
            "prefill_direct_median_ms": prefill_direct_med,
            "prefill_direct_median_tps": prefill_direct_tps,
            "decode_direct_median_ms": decode_direct_med,
            "decode_direct_median_tps": decode_direct_tps,
        })

    # Derive prefill_ms / decode_ms_per_tok per (engine, variant, pp) via linear fit.
    fit_by_group = {}
    for (engine, variant, pp, _tg), _ in sorted(cells.items()):
        key = (engine, variant, pp)
        if key in fit_by_group:
            continue
        tg_vals = []
        med_vals = []
        for r in cell_rows:
            if (r["engine"], r["variant"], r["pp"]) == key:
                tg_vals.append(r["tg"])
                med_vals.append(r["e2e_median_ms"])
        intercept, slope = fit_line(tg_vals, med_vals)
        decode_fit_tps = (1000.0 / slope) if slope and slope > 0 else float("nan")
        fit_by_group[key] = {
            "prefill_fit_ms": intercept,
            "decode_fit_ms_per_tok": slope,
            "decode_fit_tps": decode_fit_tps,
            "n_tg_points": len(tg_vals),
        }

    for r in cell_rows:
        key = (r["engine"], r["variant"], r["pp"])
        r.update(fit_by_group[key])

    # Write CSV.
    fields = [
        "engine", "variant", "pp", "tg", "n",
        "prompt_tokens", "gen_tokens",
        "e2e_median_ms", "e2e_mean_ms", "e2e_p25_ms", "e2e_p75_ms", "e2e_iqr_ms",
        "request_gen_median_tps", "request_gen_mean_tps",
        "request_nominal_roofline_tps", "request_nominal_roofline_fraction",
        "request_effective_roofline_tps", "request_effective_roofline_fraction",
        "request_roofline_prefill_ms",
        "request_nominal_roofline_decode_ms", "request_effective_roofline_decode_ms",
        "request_nominal_bandwidth_gbps", "request_effective_bandwidth_gbps",
        "request_roofline_weight_gb", "request_roofline_avg_kv_gb",
        "request_roofline_param_count",
        "e2e_median_tps", "e2e_mean_tps",
        "prefill_direct_median_ms", "prefill_direct_median_tps",
        "decode_direct_median_ms", "decode_direct_median_tps",
        "prefill_fit_ms", "decode_fit_ms_per_tok", "decode_fit_tps", "n_tg_points",
    ]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in cell_rows:
            w.writerow(r)

    # Markdown table.
    lines = []
    lines.append("# Paper Sweep — Median + IQR")
    lines.append("")
    n_values = sorted({r["n"] for r in cell_rows})
    if len(n_values) == 1:
        n_summary = f"n reps = {n_values[0]} per cell"
    elif n_values:
        n_summary = "n reps vary by cell; see the n column"
    else:
        n_summary = "n reps = 0 per cell"
    lines.append(f"{n_summary}; "
                 "per-cell tables report medians unless labeled otherwise. "
                 "The benchmark stdout summaries are means, so they are not "
                 "expected to match these medians exactly. `decode_fit_*` "
                 "columns come from a linear fit of e2e_median_ms vs tg for "
                 "each (engine, variant, pp).")
    lines.append("")
    if args.request_roofline:
        resolved = args.roofline_resolved
        bw_gbps = resolved["effective_bw_gbps"]
        nominal_bw_gbps = resolved["nominal_bw_gbps"]
        weight_gb = resolved["weight_bytes"] / 1e9
        kv_step_bytes = (
            2
            * resolved["layers"]
            * resolved["kv_heads"]
            * resolved["head_dim"]
            * resolved["dtype_bytes"]
        )
        lines.append("## Request-Level Upper Bound")
        lines.append("")
        lines.append(
            "The roofline columns use an idealized batch-1 request-level "
            "upper bound with zero software overhead: "
            "`tg / (T_prefill_ideal + T_decode_roof)`. Decode uses "
            "FP16 weight bytes plus average KV-cache bytes. The table "
            "reports both nominal-bandwidth and effective-bandwidth "
            "roof fractions; nominal is the hardware-spec comparison, "
            "while effective is just the configured bandwidth fraction "
            "or explicit bandwidth override. "
            "Prefill uses `2 * P * pp / FLOPs_peak` only to match "
            "the request-level metric boundary."
        )
        lines.append("")
        if args.roofline_effective_bandwidth_gbps is None:
            bw_source = (
                f"{args.roofline_bandwidth_fraction:.0%} of "
                f"{args.roofline_gpu_bandwidth_gbps:.0f} GB/s"
            )
        else:
            bw_source = (
                f"explicit override; nominal={args.roofline_gpu_bandwidth_gbps:.0f} GB/s"
            )
        model_source = (
            f", model={resolved['model_dir']}"
            if resolved["uses_model_defaults"] else ""
        )
        lines.append(
            f"Parameters: `BW_nominal={nominal_bw_gbps:.1f} GB/s`, "
            f"`BW_eff={bw_gbps:.1f} GB/s` ({bw_source}), "
            f"`W={weight_gb:.2f} GB`, "
            f"`KV_step={kv_step_bytes / 1e6:.3f} MB/context-token`, "
            f"`prefill_peak={args.roofline_prefill_tflops:.1f} TFLOP/s`, "
            f"`alpha=0`{model_source}."
        )
        lines.append("")
    lines.append("## Request Generation Throughput")
    lines.append("")
    if args.request_roofline:
        lines.append("| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | nominal roof | % nominal | effective roof | % effective | total_tps over e2e (median) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in cell_rows:
        prefix = (
            f"| {r['engine']} | {r['variant']} | {r['pp']} | {r['tg']} "
            f"| {r['n']} "
            f"| {r['e2e_median_ms']:.2f} ± {r['e2e_iqr_ms']/2:.2f} "
            f"| {r['request_gen_median_tps']:.1f} "
        )
        if args.request_roofline:
            nominal_roof = fmt_opt(r["request_nominal_roofline_tps"], ".1f")
            nominal_pct = (
                fmt_opt(r["request_nominal_roofline_fraction"] * 100.0, ".1f") + "%"
                if r["request_nominal_roofline_fraction"] is not None else "—"
            )
            effective_roof = fmt_opt(r["request_effective_roofline_tps"], ".1f")
            effective_pct = (
                fmt_opt(r["request_effective_roofline_fraction"] * 100.0, ".1f") + "%"
                if r["request_effective_roofline_fraction"] is not None else "—"
            )
            lines.append(
                f"{prefix}| {nominal_roof} | {nominal_pct} "
                f"| {effective_roof} | {effective_pct} "
                f"| {r['e2e_median_tps']:.1f} |"
            )
        else:
            lines.append(f"{prefix}| {r['e2e_median_tps']:.1f} |")
    lines.append("")
    lines.append("## Derived Decode From Cross-TG Fit")
    lines.append("")
    lines.append("| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    seen = set()
    for r in cell_rows:
        key = (r["engine"], r["variant"], r["pp"])
        if key in seen:
            continue
        seen.add(key)
        slope = r["decode_fit_ms_per_tok"]
        tps_fit = r["decode_fit_tps"]
        prefill = r["prefill_fit_ms"]
        prefill_s = f"{prefill:.2f}" if prefill == prefill else "n/a"
        slope_s = f"{slope:.4f}" if slope == slope else "n/a"
        tps_s = f"{tps_fit:.1f}" if tps_fit == tps_fit else "n/a"
        lines.append(
            f"| {r['engine']} | {r['variant']} | {r['pp']} "
            f"| {r['n_tg_points']} | {prefill_s} | {slope_s} | {tps_s} |"
        )
    lines.append("")
    lines.append("## Direct Phase Timings")
    lines.append("")
    lines.append("Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.")
    lines.append("")
    lines.append("| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in cell_rows:
        pre = (
            f"{r['prefill_direct_median_ms']:.2f}"
            if r["prefill_direct_median_ms"] is not None else "—"
        )
        pre_tps = (
            f"{r['prefill_direct_median_tps']:.1f}"
            if r["prefill_direct_median_tps"] is not None else "—"
        )
        dec = (
            f"{r['decode_direct_median_ms']:.2f}"
            if r["decode_direct_median_ms"] is not None else "—"
        )
        dec_tps = (
            f"{r['decode_direct_median_tps']:.1f}"
            if r["decode_direct_median_tps"] is not None else "—"
        )
        if pre == "—" and dec == "—":
            continue
        lines.append(
            f"| {r['engine']} | {r['variant']} | {r['pp']} | {r['tg']} "
            f"| {pre} | {pre_tps} | {dec} | {dec_tps} |"
        )

    args.markdown.write_text("\n".join(lines) + "\n")

    print(f"wrote {args.csv}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
