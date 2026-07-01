#!/usr/bin/env bash
# PyTorch (transformers) PREFILL pp sweep — the baseline from cutile-rs issue #171.
#
# Mirrors sweep_pp_sm120.sh's pp set and prompt generation so the numbers line up
# with the grout sweep on the same box. PyTorch-only; no grout build, no GPU
# clock locking. Runs transformers prefill (fp16, eager SDPA, single forward).
#
# Usage:
#   ./benchmarks/sweep_pp_pytorch.sh
#   MODEL_HF=/path/to/Qwen3-4B ./benchmarks/sweep_pp_pytorch.sh
#   SWEEP_PP_VALUES="128 512" ./benchmarks/sweep_pp_pytorch.sh   # smoke subset
#   BENCH_PYTHON=/path/to/.venv/bin/python ./benchmarks/sweep_pp_pytorch.sh
#
# Needs a Python with torch + transformers. Defaults to ../bench_envs/vllm_env
# (the same env make_prompts.py uses in sweep_pp.sh), else $BENCH_PYTHON, else python3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"
# Match the canonical 5090 pp sweep so rows align with sweep_pp_sm120.sh.
: "${SWEEP_PP_VALUES:=18 128 512 2048 8192}"
REPS="${REPS:-5}"
WARMUP_REPS="${WARMUP_REPS:-3}"
ATTN="${ATTN:-sdpa}"
# Variants: eager SDPA and torch.compile (the standard PyTorch speedup).
RUN_EAGER="${RUN_EAGER:-1}"
RUN_COMPILED="${RUN_COMPILED:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"   # default | reduce-overhead | max-autotune[-no-cudagraphs]

# Resolve a python interpreter with torch + transformers.
pick_python() {
    if [[ -n "${BENCH_PYTHON:-}" ]]; then echo "$BENCH_PYTHON"; return; fi
    local cand="$GROUT_DIR/../bench_envs/vllm_env/bin/python"
    if [[ -x "$cand" ]]; then echo "$cand"; return; fi
    echo "python3"
}
PY="$(pick_python)"

if ! "$PY" -c "import torch, transformers" 2>/dev/null; then
    echo "error: '$PY' lacks torch/transformers. Set BENCH_PYTHON to a venv that has them." >&2
    exit 1
fi

read -r -a PP_VALUES <<< "$SWEEP_PP_VALUES"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$SCRIPT_DIR/results/sweep_pytorch/$TS"
PROMPTS_DIR="$OUT_DIR/prompts"
JSONL="$OUT_DIR/run.jsonl"
mkdir -p "$PROMPTS_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo " PyTorch (transformers) prefill scan  (pp ∈ {${PP_VALUES[*]}})"
echo " model=$MODEL_HF  reps=$REPS warmup=$WARMUP_REPS attn=$ATTN"
echo " variants: eager=$RUN_EAGER compiled=$RUN_COMPILED (mode=$COMPILE_MODE)"
echo " python=$PY"
echo " out=$OUT_DIR"
echo "═══════════════════════════════════════════════════════════════"

# Identical prompts to the grout sweep (deterministic from the model tokenizer).
"$PY" "$SCRIPT_DIR/make_prompts.py" \
    --model "$MODEL_HF" --out-dir "$PROMPTS_DIR" --pp "${PP_VALUES[@]}"

run_variant() {  # $1 = label, remaining = extra bench_transformers args
    local label="$1"; shift
    echo ""
    echo "--- transformers ($label) pp=${PP} (reps=$REPS) ---"
    "$PY" "$SCRIPT_DIR/bench_transformers.py" \
        --model "$MODEL_HF" \
        --prompt-file "$PROMPTS_DIR/pp_${PP}.txt" \
        --raw-prompt \
        --reps "$REPS" \
        --warmup-reps "$WARMUP_REPS" \
        --attn "$ATTN" \
        --json "$JSONL" \
        --pp-label "$PP" "$@"
}

for PP in "${PP_VALUES[@]}"; do
    if [[ "$RUN_EAGER" == "1" ]]; then
        run_variant "eager"
    fi
    if [[ "$RUN_COMPILED" == "1" ]]; then
        # First warmup rep absorbs the torch.compile cost (untimed).
        run_variant "compiled:$COMPILE_MODE" --compile --compile-mode "$COMPILE_MODE"
    fi
done

# Aggregate into the same format as the grout sweep (best-effort).
if "$PY" "$SCRIPT_DIR/aggregate_sweep.py" \
        --jsonl "$JSONL" \
        --csv "$OUT_DIR/aggregate.csv" \
        --markdown "$OUT_DIR/aggregate.md" 2>/dev/null; then
    echo "Aggregated -> $OUT_DIR/aggregate.md"
fi

# Always print a compact prefill summary (median ms + prompt tok/s per pp).
"$PY" - "$JSONL" <<'PYEOF'
import json, sys, statistics
from collections import defaultdict
rows = defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    rows[(r.get("variant", "?"), r["pp"])].append((r["prompt_tokens"], r["prefill_ms"]))
print(f"\n  {'variant':<22} {'pp':>6} {'prompt_tok':>11} {'median_ms':>11} {'prompt_tok/s':>13}")
print("  " + "-" * 66)
for (variant, pp) in sorted(rows, key=lambda k: (k[0], k[1])):
    toks = rows[(variant, pp)][0][0]
    med = statistics.median(m for _, m in rows[(variant, pp)])
    tps = toks / (med / 1000.0) if med else 0.0
    print(f"  {variant:<22} {pp:>6} {toks:>11} {med:>11.2f} {tps:>13.1f}")
PYEOF

echo ""
echo "Done. Results in: $OUT_DIR"
