#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ITERS="${ITERS:-200}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"
SHAPES="${SHAPES:-all}"
MODEL_SIZE="${MODEL_SIZE:-4b}"
COPIES="${COPIES:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-benchmarks/results/gemv/${STAMP}}"
BIN="target/release/gemv_microbench"

mkdir -p "$OUT_DIR"

cargo build --release --features benchmarks --bin gemv_microbench

CSV="${OUT_DIR}/gemv.csv"
: > "$CSV"

first=1

run_case() {
  local label="$1"
  shift
  local header_args=()
  if [[ "$first" -eq 0 ]]; then
    header_args=(--no-header)
  fi
  first=0

  env \
    -u GROUT_CUBLAS_FAST_ALGO \
    -u GROUT_CUBLAS_COMPUTE16 \
    -u GROUT_CUBLAS_COMPUTE16_MAX_M \
    "$@" \
    "$BIN" \
      --label "$label" \
      --model-size "$MODEL_SIZE" \
      --shape "$SHAPES" \
      --iters "$ITERS" \
      --warmup-iters "$WARMUP_ITERS" \
      --copies "$COPIES" \
      "${header_args[@]}" >> "$CSV"
}

run_case current
run_case compute16_0 GROUT_CUBLAS_COMPUTE16=0
run_case compute16_1 GROUT_CUBLAS_COMPUTE16=1

for algo in default default_tensor_op {0..15}; do
  run_case "algo_${algo}" GROUT_CUBLAS_FAST_ALGO="$algo"
done

echo "wrote ${CSV}"
