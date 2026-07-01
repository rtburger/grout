#!/usr/bin/env bash
# Cross-engine kernel-level profile comparison: grout vs llama.cpp vs SGLang.
#
# Captures nsys traces of each engine running the same Qwen3-4B apples-to-
# apples workload (18-token templated prompt, 36 generated tokens) and
# extracts per-kernel GPU time breakdowns.
#
# Pins each target process to a fixed CPU subset so it doesn't trade places
# with other workloads. The GPU is shared — this script doesn't own it;
# re-run when the GPU is quiet for meaningful kernel timings.
#
# Output layout:
#   $RESULTS_DIR/
#     grout.nsys-rep      full trace (open in Nsight Systems GUI)
#     llama.nsys-rep
#     sglang.nsys-rep
#     grout_kern.csv      per-kernel GPU time (top of list)
#     llama_kern.csv
#     sglang_kern.csv
#     stdout_*.log        captured stdout/stderr of the target
#
# SGLang runs through its Python venv and spawns scheduler subprocesses;
# nsys follows them via --trace-fork-before-exec=true. Traces cover both
# engine init (model load, kernel compile, graph capture) and steady-state
# decode — the per-kernel CSV aggregates across both. For decode-only
# analysis, sort the CSV and ignore one-shot init kernels (compile,
# memcpy_sym, etc.).
#
# Env overrides:
#   MODEL_HF        path to HF model dir    (default: ../hf_models/qwen3_4b)
#   GGUF_PATH       path to GGUF            (default: ../hf_models/qwen3_4b_f16.gguf)
#   LLAMA_CLI       llama-cli binary        (default: ../llama.cpp/build/bin/llama-cli)
#   GROUT_BIN       grout binary            (default: target/release/grout_bench)
#   BENCH_ENVS_DIR  bench venvs root        (default: ../bench_envs)
#   SGLANG_PY       sglang python           (default: $BENCH_ENVS_DIR/sglang_env/bin/python3)
#   BENCH_SGLANG    sglang bench script     (default: benchmarks/bench_sglang.py)
#   RESULTS_DIR     output dir              (default: benchmarks/results/profile)
#   CPU_MASK        taskset mask            (default: 0-7)
#   REPS            measured decode reps    (default: 3)
#   WARMUP_REPS     grout warmup reps       (default: 2)
#   MAX_NEW_TOKENS  tokens to generate      (default: 36)
#   SKIP_GROUT      set to 1 to skip grout
#   SKIP_LLAMA      set to 1 to skip llama
#   SKIP_SGLANG     set to 1 to skip sglang

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"
GGUF_PATH="${GGUF_PATH:-$GROUT_DIR/../hf_models/qwen3_4b_f16.gguf}"
LLAMA_CLI="${LLAMA_CLI:-$GROUT_DIR/../llama.cpp/build/bin/llama-cli}"
GROUT_BIN="${GROUT_BIN:-$GROUT_DIR/target/release/grout_bench}"
BENCH_ENVS_DIR="${BENCH_ENVS_DIR:-$GROUT_DIR/../bench_envs}"
SGLANG_PY="${SGLANG_PY:-$BENCH_ENVS_DIR/sglang_env/bin/python3}"
BENCH_SGLANG="${BENCH_SGLANG:-$SCRIPT_DIR/bench_sglang.py}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/profile}"

CPU_MASK="${CPU_MASK:-0-7}"
REPS="${REPS:-3}"
WARMUP_REPS="${WARMUP_REPS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-36}"

SKIP_GROUT="${SKIP_GROUT:-0}"
SKIP_LLAMA="${SKIP_LLAMA:-0}"
SKIP_SGLANG="${SKIP_SGLANG:-0}"

# Anaconda ships a libstdc++ missing GLIBCXX_3.4.32; sgl-kernel and FlashInfer
# JIT .so files need the system one. Apply only when the file exists.
SYS_STDCXX="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if [[ -f "$SYS_STDCXX" ]]; then
    SGLANG_LD_PRELOAD="$SYS_STDCXX"
else
    SGLANG_LD_PRELOAD=""
fi

# Same 18-token pre-templated Qwen3 chat prompt used by the apples-to-apples
# benches ($'...' is Bash ANSI-C quoting so the \n becomes a real newline).
PROMPT=$'<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

# --- sanity checks ---------------------------------------------------------
command -v nsys >/dev/null || { echo "ERROR: nsys not in PATH"; exit 1; }
command -v taskset >/dev/null || { echo "ERROR: taskset not in PATH"; exit 1; }
[[ -d "$MODEL_HF" ]] || { echo "ERROR: model dir not found: $MODEL_HF"; exit 1; }

if [[ "$SKIP_GROUT" != 1 ]]; then
    # Auto-rebuild grout if the binary is missing or older than any source.
    # Without this, benchmarking can silently run a stale binary that
    # predates source changes — the bug that hid the real impact of the
    # unchecked_accesses + max_divisibility=8 edits on 2026-04-19 and
    # caused a v1 kernel panic to be mistaken for a successful run.
    # Set SKIP_REBUILD=1 to bypass (e.g., for testing the binary directly).
    need_build=0
    if [[ ! -x "$GROUT_BIN" ]]; then
        echo "=== Grout binary missing at $GROUT_BIN — building ==="
        need_build=1
    elif [[ "${SKIP_REBUILD:-0}" != 1 ]]; then
        # Any .rs under src/ or Cargo.{toml,lock} newer than the binary?
        stale_src=$(find "$GROUT_DIR/src" -name '*.rs' -newer "$GROUT_BIN" -print -quit 2>/dev/null)
        if [[ -z "$stale_src" ]]; then
            for f in "$GROUT_DIR/Cargo.toml" "$GROUT_DIR/Cargo.lock"; do
                if [[ -f "$f" && "$f" -nt "$GROUT_BIN" ]]; then
                    stale_src="$f"
                    break
                fi
            done
        fi
        if [[ -n "$stale_src" ]]; then
            echo "=== Grout binary older than $stale_src — rebuilding ==="
            need_build=1
        fi
    fi
    if [[ "$need_build" == 1 ]]; then
        (cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench) || {
            echo "ERROR: cargo build failed — fix the errors or set SKIP_GROUT=1"
            exit 1
        }
    fi
    [[ -x "$GROUT_BIN" ]] || { echo "ERROR: grout binary still not found at $GROUT_BIN after build"; exit 1; }
fi
if [[ "$SKIP_LLAMA" != 1 ]]; then
    [[ -x "$LLAMA_CLI" ]] || { echo "ERROR: llama-cli not found at $LLAMA_CLI"; exit 1; }
    [[ -f "$GGUF_PATH" ]] || { echo "ERROR: GGUF not found: $GGUF_PATH"; exit 1; }
fi
if [[ "$SKIP_SGLANG" != 1 ]]; then
    [[ -x "$SGLANG_PY" ]] || { echo "ERROR: sglang python not found at $SGLANG_PY — create the venv per benchmarks/README.md or set SGLANG_PY/SKIP_SGLANG=1"; exit 1; }
    [[ -f "$BENCH_SGLANG" ]] || { echo "ERROR: sglang bench script not found at $BENCH_SGLANG"; exit 1; }
fi

mkdir -p "$RESULTS_DIR"

# Keep nsys's own temp files under the result directory. Some sandboxed temp
# directories are rejected by Nsight Systems.
export TMPDIR="${TMPDIR:-$RESULTS_DIR/.nsys_tmp}"
mkdir -p "$TMPDIR"

NSYS_FLAGS=(
    --trace=cuda,osrt
    --cuda-graph-trace=node           # expand CUDA graph replays into per-node events
    --cuda-trace-scope=process-tree   # explicit: capture CUDA in descendant processes
    --trace-fork-before-exec=true     # follow scheduler subprocesses (SGLang)
    # --wait defaults to 'primary': finalize trace when the bench's main
    # process exits. Using --wait=all hangs because SGLang scheduler
    # subprocesses get re-parented to init and never exit cleanly.
    --force-overwrite=true
    --stats=false                     # skip slow in-line report; we run `nsys stats` after
)

echo "================================================================"
echo " Profile comparison: grout vs llama.cpp vs SGLang (Qwen3-4B FP16)"
echo "================================================================"
echo "CPU mask   : $CPU_MASK"
echo "Prompt     : 18-token pre-templated"
echo "Gen tokens : $MAX_NEW_TOKENS"
echo "Grout reps : $WARMUP_REPS warmup + $REPS measured"
echo "Engines    : $(
    parts=()
    [[ "$SKIP_GROUT"  != 1 ]] && parts+=(grout)
    [[ "$SKIP_LLAMA"  != 1 ]] && parts+=(llama)
    [[ "$SKIP_SGLANG" != 1 ]] && parts+=(sglang)
    (IFS=' '; echo "${parts[*]:-<none>}")
)"
echo "GPU state  :"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv || true
echo ""

ENGINES=()

# --- grout -----------------------------------------------------------------
# GROUT_CUDA_GRAPH_DECODE=1 is critical: without it, decode falls through to
# the slower StepGraph path (~130 t/s) and the profile captures a different
# op mix than what we actually ship. See src/model.rs:1416 (default false).
if [[ "$SKIP_GROUT" != 1 ]]; then
    echo "=== Profiling grout ==="
    taskset -c "$CPU_MASK" nsys profile \
        "${NSYS_FLAGS[@]}" \
        --output="$RESULTS_DIR/grout" \
        env "GROUT_CUDA_GRAPH_DECODE=1" \
        "$GROUT_BIN" \
            --model "$MODEL_HF" \
            --prompt "Hello, how are you?" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --reps "$REPS" \
            --warmup-reps "$WARMUP_REPS" \
        > "$RESULTS_DIR/stdout_grout.log" 2>&1
    echo "  -> $RESULTS_DIR/grout.nsys-rep"
    ENGINES+=(grout)
fi

# --- llama.cpp via llama-cli ----------------------------------------------
# llama-cli is single-shot: load model, prefill, decode, exit. Simpler to
# profile than llama-server because there's no persistent HTTP stack. The
# per-kernel numbers are the same in both.
if [[ "$SKIP_LLAMA" != 1 ]]; then
    echo "=== Profiling llama.cpp (llama-cli) ==="
    taskset -c "$CPU_MASK" nsys profile \
        "${NSYS_FLAGS[@]}" \
        --output="$RESULTS_DIR/llama" \
        "$LLAMA_CLI" \
            -m "$GGUF_PATH" \
            -ngl 99 -fa 1 \
            -p "$PROMPT" \
            -n "$MAX_NEW_TOKENS" \
            --temp 0 \
            -no-cnv \
            --no-warmup \
        > "$RESULTS_DIR/stdout_llama.log" 2>&1
    echo "  -> $RESULTS_DIR/llama.nsys-rep"
    ENGINES+=(llama)
fi

# --- SGLang via bench_sglang.py -------------------------------------------
# bench_sglang.py drives sgl.Engine with temperature=0 and the same
# 18-token pre-templated prompt as the other engines. --mode default keeps
# the profile to one engine lifecycle (CUDA graphs on, radix cache on).
#
# SGLang runs the model worker in a torch.multiprocessing subprocess.
# Earlier attempts tried to follow it with --trace-fork-before-exec=true +
# --cuda-trace-scope=process-tree, and while nsys saw the processes, CUPTI
# failed to attach to the subprocess's CUDA context so zero kernels were
# captured. The fix: bench_sglang.py --profile-cuda calls
# engine.start_profile(activities=["CUDA_PROFILER"]) which runs
# torch.cuda.cudart().cudaProfilerStart() INSIDE the scheduler subprocess,
# and we match it here with --capture-range=cudaProfilerApi so nsys only
# captures the measured reps — decoupled from subprocess env propagation.
#
# The bash -c wrapper preserves any LD_PRELOAD nsys injects (Anaconda
# libstdc++.so.6 is needed for sgl-kernel).
if [[ "$SKIP_SGLANG" != 1 ]]; then
    echo "=== Profiling SGLang ==="
    sglang_cmd="export LD_PRELOAD=\"${SGLANG_LD_PRELOAD}\${LD_PRELOAD:+:\$LD_PRELOAD}\"; "
    sglang_cmd+="exec \"\$@\""
    taskset -c "$CPU_MASK" nsys profile \
        "${NSYS_FLAGS[@]}" \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop-shutdown \
        --output="$RESULTS_DIR/sglang" \
        bash -c "$sglang_cmd" sglang-wrapper \
            "$SGLANG_PY" "$BENCH_SGLANG" \
                --model "$MODEL_HF" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --reps "$REPS" \
                --mode default \
                --profile-cuda \
        > "$RESULTS_DIR/stdout_sglang.log" 2>&1 || {
            echo "  WARN: SGLang profile exited non-zero; check $RESULTS_DIR/stdout_sglang.log"
        }
    echo "  -> $RESULTS_DIR/sglang.nsys-rep"
    ENGINES+=(sglang)
fi

if [[ ${#ENGINES[@]} -eq 0 ]]; then
    echo "No engines profiled (all SKIP_* set). Nothing to do."
    exit 0
fi

# --- extract kernel-level stats --------------------------------------------
echo ""
echo "=== Extracting kernel stats ==="
for name in "${ENGINES[@]}"; do
    [[ -f "$RESULTS_DIR/${name}.nsys-rep" ]] || {
        echo "  SKIP: $RESULTS_DIR/${name}.nsys-rep missing"
        continue
    }
    # --force-export=true: regenerate sqlite even if one exists older than
    # the .nsys-rep. Without this, nsys stats sees a stale sqlite from a
    # previous run, prints usage, exits non-zero, and leaves the CSV
    # pointing at old data.
    nsys stats \
        --force-export=true \
        --report cuda_gpu_kern_sum \
        --format csv \
        --output "$RESULTS_DIR/${name}_kern" \
        "$RESULTS_DIR/${name}.nsys-rep" \
        > /dev/null 2>&1 || {
            echo "  WARN: stats extraction failed for $name; check $RESULTS_DIR/${name}.nsys-rep manually"
            continue
        }
    # nsys stats writes "<output>_cuda_gpu_kern_sum.csv"; normalize filename
    mv -f "$RESULTS_DIR/${name}_kern_cuda_gpu_kern_sum.csv" "$RESULTS_DIR/${name}_kern.csv" 2>/dev/null || true
    echo "  -> $RESULTS_DIR/${name}_kern.csv"
done

# --- quick side-by-side ----------------------------------------------------
echo ""
echo "=== Top 15 kernels by total GPU time (each engine) ==="
for name in "${ENGINES[@]}"; do
    f="$RESULTS_DIR/${name}_kern.csv"
    [[ -f "$f" ]] || continue
    echo ""
    echo "--- $name ---"
    # Columns: Time(%)  Total Time  Instances  Avg  Med  Min  Max  StdDev  Name
    head -17 "$f" | column -t -s,
done

echo ""
echo "Done."
echo "  Full traces (open in Nsight Systems GUI): $RESULTS_DIR/{$(IFS=,; echo "${ENGINES[*]}")}.nsys-rep"
echo "  Kernel CSVs:                              $RESULTS_DIR/{$(IFS=,; echo "${ENGINES[*]}")}_kern.csv"
