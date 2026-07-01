#!/usr/bin/env bash
# Refresh just the grout rows in an existing sweep directory.
# Keeps llama.cpp / sglang / vllm / trt-llm data untouched.
#
# Usage:  ./regen_grout.sh <sweep-dir>  [tg1 tg2 ...]
# Example:
#   ./regen_grout.sh benchmarks/results/sweep/20260420_151333
#   ./regen_grout.sh benchmarks/results/sweep/20260420_151333 36 128 512
#
# Grout is invoked with whatever env you've already exported (e.g.
# GROUT_FMHA_SPLIT_KV=1, GROUT_FMHA_NUM_KV_SPLITS=4). The script does NOT
# set those itself — you run it like:
#   GROUT_FMHA_SPLIT_KV=1 ./regen_grout.sh benchmarks/results/sweep/...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"

SWEEP_DIR="${1:?usage: regen_grout.sh <sweep-dir> [tg values ...]}"
shift || true

JSONL="$SWEEP_DIR/run.jsonl"
if [[ ! -f "$JSONL" ]]; then
    echo "ERROR: $JSONL not found" >&2
    exit 2
fi

# Default tg values = 36 128 512 (paper sweep).
TG_VALUES=("$@")
if [[ ${#TG_VALUES[@]} -eq 0 ]]; then
    TG_VALUES=(36 128 512)
fi

# Same config knobs as sweep_tg.sh.
PP_LABEL=18
BENCH_REPS=10
WARMUP_REPS=3

# Build grout (auto-rebuild if stale).
(cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench 2>&1 | tail -3)

# Strip existing grout records. Keep a backup just in case.
cp "$JSONL" "$JSONL.bak.$(date +%s)"
python3 - "$JSONL" <<'PY'
import json, sys
p = sys.argv[1]
kept = []
dropped = 0
for line in open(p):
    if not line.strip():
        continue
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        continue
    if rec.get("engine") == "grout":
        dropped += 1
        continue
    kept.append(line)
open(p, "w").writelines(kept)
print(f"stripped {dropped} grout records, kept {len(kept)} others")
PY

# Re-run grout at each tg, appending fresh records.
for TG in "${TG_VALUES[@]}"; do
    echo ""
    echo "--- grout pp=${PP_LABEL} tg=${TG}  (env: $(env | grep -E '^GROUT_' | paste -sd, -))"
    (cd "$GROUT_DIR" && ./target/release/grout_bench \
        --model "$MODEL_HF" \
        --prompt "Hello, how are you?" \
        --max-new-tokens "$TG" \
        --reps "$BENCH_REPS" \
        --warmup-reps "$WARMUP_REPS" \
        --json "$JSONL" \
        --variant "${GROUT_VARIANT_LABEL:-default}" \
        --pp-label "$PP_LABEL" \
        --ignore-eos 2>&1) | tail -15
done

# Re-aggregate.
python3 "$SCRIPT_DIR/aggregate_sweep.py" \
    --jsonl "$JSONL" \
    --csv "$SWEEP_DIR/aggregate.csv" \
    --markdown "$SWEEP_DIR/aggregate.md"

echo ""
echo "Done. Updated:"
echo "  $SWEEP_DIR/run.jsonl  (grout rows refreshed)"
echo "  $SWEEP_DIR/aggregate.csv"
echo "  $SWEEP_DIR/aggregate.md"
