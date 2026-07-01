#!/usr/bin/env python3
"""Combine grout + PyTorch prefill sweeps into one table + plot + README.

Reads every per-rep JSONL under ./raw/ (grout's run.jsonl and the
sweep_pp_pytorch.sh outputs), computes median prefill latency and prefill
throughput per (engine, variant, pp), and regenerates:

    prefill_comparison.png   prefill tok/s and ms vs pp
    README.md                methodology + tables + embedded plot

Pure prefill is compared on both sides: grout's prefill_ms (prompt_elapsed,
the single prompt step_seq) vs PyTorch's single use_cache=False forward
(CUDA-event timed). Identical prompts (same make_prompts.py). RTX 5090 /
sm_120, Qwen3-4B fp16.

Run:  python3 benchmarks/pytorch/combine_prefill.py
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import tempfile
from pathlib import Path

# Headless plotting + a writable config dir (avoids ~/.config/matplotlib issues).
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

# Display order + pretty labels + plot styling for each (engine, variant).
SERIES = [
    (("grout", "default"),               "grout (cuTile, tuned sm120)",     "#1f77b4", "o", "-"),
    (("transformers", "eager-sdpa"),     "PyTorch eager (SDPA)",            "#d62728", "s", "--"),
    (("transformers", "compiled-default"),      "PyTorch compile (default)",       "#2ca02c", "^", "--"),
    (("transformers", "compiled-max-autotune"), "PyTorch compile (max-autotune)",  "#ff7f0e", "D", "--"),
]


def load_cells() -> dict:
    """(engine, variant, pp) -> {'ms': [...], 'toks': int}."""
    cells: dict = {}
    for path in sorted(glob.glob(str(RAW / "**" / "*.jsonl"), recursive=True)):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "prefill_ms" not in r:
                continue
            key = (r["engine"], r["variant"], int(r["pp"]))
            cell = cells.setdefault(key, {"ms": [], "toks": int(r["prompt_tokens"])})
            cell["ms"].append(float(r["prefill_ms"]))
    return cells


def main() -> None:
    cells = load_cells()
    pps = sorted({pp for (_e, _v, pp) in cells})

    # Resolve present series (in display order) -> {pp: (median_ms, tps, n)}.
    present = []
    for key, label, color, marker, ls in SERIES:
        per_pp = {}
        for pp in pps:
            c = cells.get((key[0], key[1], pp))
            if not c or not c["ms"]:
                continue
            med = statistics.median(c["ms"])
            per_pp[pp] = (med, c["toks"] / (med / 1000.0), len(c["ms"]))
        if per_pp:
            present.append((label, color, marker, ls, per_pp))

    grout = next((s for s in present if s[0].startswith("grout")), None)
    eager = next((s for s in present if "eager" in s[0]), None)

    # ---- plot -------------------------------------------------------------
    fig, (ax_tps, ax_ms) = plt.subplots(1, 2, figsize=(13, 5))
    for label, color, marker, ls, per_pp in present:
        xs = sorted(per_pp)
        ax_tps.plot(xs, [per_pp[p][1] for p in xs], marker=marker, ls=ls,
                    color=color, label=label)
        ax_ms.plot(xs, [per_pp[p][0] for p in xs], marker=marker, ls=ls,
                   color=color, label=label)
    ax_tps.set(xscale="log", xlabel="prompt length (pp, tokens)",
               ylabel="prefill throughput (tok/s)",
               title="Prefill throughput (higher is better)")
    ax_ms.set(xscale="log", yscale="log", xlabel="prompt length (pp, tokens)",
              ylabel="prefill latency (ms)",
              title="Prefill latency (lower is better)")
    for ax in (ax_tps, ax_ms):
        ax.set_xticks(pps)
        ax.set_xticklabels([str(p) for p in pps])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Qwen3-4B fp16 prefill — grout vs PyTorch\nNVIDIA RTX 5090 (sm_120)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = HERE / "prefill_comparison.png"
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")

    # ---- markdown helpers -------------------------------------------------
    def fmt_tps(v):
        return f"{v:,.0f}" if v is not None else "—"

    def row(getter):
        return [getter(per_pp.get(pp)) for (_l, _c, _m, _ls, per_pp) in present]

    def tps_table():
        hdr = "| pp | " + " | ".join(l for l, *_ in present) + " | grout vs eager |"
        sep = "|---:|" + "|".join(["---:"] * len(present)) + "|---:|"
        lines = [hdr, sep]
        for pp in pps:
            cells_ = [fmt_tps(per_pp[pp][1]) if pp in per_pp else "—"
                      for (_l, _c, _m, _ls, per_pp) in present]
            spd = "—"
            if grout and eager and pp in grout[4] and pp in eager[4]:
                spd = f"{grout[4][pp][1] / eager[4][pp][1]:.2f}×"
            lines.append(f"| {pp} | " + " | ".join(cells_) + f" | {spd} |")
        return "\n".join(lines)

    def ms_table():
        hdr = "| pp | " + " | ".join(l for l, *_ in present) + " |"
        sep = "|---:|" + "|".join(["---:"] * len(present)) + "|"
        lines = [hdr, sep]
        for pp in pps:
            cells_ = [f"{per_pp[pp][0]:.2f}" if pp in per_pp else "—"
                      for (_l, _c, _m, _ls, per_pp) in present]
            lines.append(f"| {pp} | " + " | ".join(cells_) + " |")
        return "\n".join(lines)

    # Headline: grout speedup vs eager across pp.
    speedups = [grout[4][pp][1] / eager[4][pp][1]
                for pp in pps if grout and eager and pp in grout[4] and pp in eager[4]]
    headline = (f"grout prefill is **{min(speedups):.2f}×–{max(speedups):.2f}× faster "
                f"than PyTorch eager** across pp=18..8192" if speedups else "")
    n_note = "n reps per cell: " + ", ".join(
        f"{l}={statistics.median([v[2] for v in per_pp.values()]):.0f}"
        for (l, _c, _m, _ls, per_pp) in present)

    readme = f"""# Qwen3-4B prefill — grout vs PyTorch (RTX 5090 / sm_120)

Prefill-only comparison for [cutile-rs issue #171](https://github.com/NVlabs/cutile-rs/issues/171):
grout (cuTile, tuned sm_120 profile) vs PyTorch `transformers`, eager and
`torch.compile`. {headline}.

## Method

- **GPU / model**: RTX 5090 (Blackwell sm_120), Qwen3-4B fp16.
- **Pure prefill on both sides**: grout's `prefill_ms` (`prompt_elapsed`, the
  single prompt `step_seq`); PyTorch a single `use_cache=False` forward, CUDA-event
  timed. No decode in either number.
- **PyTorch variants**: eager SDPA (flash-backed), `torch.compile` mode=`default`,
  and mode=`max-autotune`. fp16, no chunking.
- **Identical prompts** generated by `make_prompts.py` (exact token counts).
- Medians over measured reps ({n_note}). tok/s = prompt_tokens / median prefill ms.
- grout tuned profile via `benchmarks/sweep_pp_sm120.sh`; PyTorch via
  `benchmarks/sweep_pp_pytorch.sh`.

## Prefill throughput (tok/s, higher is better)

{tps_table()}

## Prefill latency (median ms, lower is better)

{ms_table()}

## Plot

![prefill comparison](prefill_comparison.png)

## Notes

- On the tuned 5090, grout prefill beats PyTorch eager at every pp — the opposite
  of the RTX 5070 Ti result in the issue, i.e. that gap was tuning/hardware, not
  kernel quality.
- `torch.compile` helps PyTorch at short prompts but **regresses at long context**
  (pp=8192): inductor's fused LM-head matmul over the full prompt is far slower
  than eager's cuBLAS call there, so compiled falls behind eager (and further
  behind grout) at pp=8192.
- grout's own prefill throughput peaks near pp=2048 and dips at pp=8192 — the
  early edge of the long-context `MatMulSlice` scaling tracked in the issue.

## Reproduce

```bash
# grout (writes benchmarks/results/sweep/<ts>/run.jsonl)
./benchmarks/sweep_pp_sm120.sh
# PyTorch eager + compiled (writes benchmarks/results/sweep_pytorch/<ts>/run.jsonl)
./benchmarks/sweep_pp_pytorch.sh
COMPILE_MODE=max-autotune ./benchmarks/sweep_pp_pytorch.sh
# drop the run.jsonl files under benchmarks/pytorch/raw/ then:
python3 benchmarks/pytorch/combine_prefill.py
```

Raw per-rep data for this report lives in [`raw/`](raw/).
"""
    (HERE / "README.md").write_text(readme)
    print(f"wrote {HERE / 'README.md'}")


if __name__ == "__main__":
    main()
