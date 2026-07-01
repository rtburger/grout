#!/usr/bin/env bash
# Legacy one-shot benchmark script. The paper-facing harness is
# sweep_tg_sm{100,120}.sh and sweep_pp_sm{100,120}.sh.
#
# Usage: run_all.sh [GPU_CLOCK_MHZ]
#
# With no args: clocks run at whatever the driver + thermals decide. Expect
# 7-10% variance on RTX 5090 (cold 2910 MHz → warm 2400 MHz throttle).
#
# With an MHz value (e.g. `run_all.sh 2400`): lock both min and max GPU core
# clock to that speed via `sudo nvidia-smi --lock-gpu-clocks`, run the
# benchmarks, then reset clocks on exit. Typical choice is 2400 MHz — the
# sustained clock under load, giving warm-steady-state numbers that match
# what SGLang/vLLM's published benchmarks see. Requires passwordless sudo
# for nvidia-smi, or run this script itself under sudo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LOCK_MHZ="${1:-}"
if [[ -n "$LOCK_MHZ" ]]; then
    if ! [[ "$LOCK_MHZ" =~ ^[0-9]+$ ]]; then
        printf '\033[1;31mERROR: expected integer MHz, got %q\033[0m\n' "$LOCK_MHZ" >&2
        exit 2
    fi
    echo "Locking GPU clock to ${LOCK_MHZ} MHz for this run…"
    sudo nvidia-smi --lock-gpu-clocks="${LOCK_MHZ},${LOCK_MHZ}" >/dev/null
    # Also lock memory clocks if GROUT_LOCK_MEM_MHZ is set (GDDR7 effective).
    if [[ -n "${GROUT_LOCK_MEM_MHZ:-}" ]]; then
        echo "Locking memory clock to ${GROUT_LOCK_MEM_MHZ} MHz…"
        sudo nvidia-smi --lock-memory-clocks="${GROUT_LOCK_MEM_MHZ},${GROUT_LOCK_MEM_MHZ}" >/dev/null
    fi
    cleanup_clocks() {
        echo "Resetting GPU clocks…"
        sudo nvidia-smi --reset-gpu-clocks >/dev/null || true
        if [[ -n "${GROUT_LOCK_MEM_MHZ:-}" ]]; then
            sudo nvidia-smi --reset-memory-clocks >/dev/null || true
        fi
    }
    trap cleanup_clocks EXIT
fi

# ── Configurable paths ──────────────────────────────────────────────────────
MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$GROUT_DIR/../llama.cpp}"
GGUF_PATH="${GGUF_PATH:-$GROUT_DIR/../hf_models/qwen3_4b_f16.gguf}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
BENCH_ENVS_DIR="${BENCH_ENVS_DIR:-$GROUT_DIR/../bench_envs}"

PROMPT="Hello, how are you?"
# Apples-to-apples: all engines generate the SAME number of tokens from
# the SAME (templated) prompt. The bench scripts each default to the
# 18-token pre-templated Qwen3 chat prompt; grout applies the same
# template internally from the raw prompt string below. 36 generated
# tokens is what vLLM / SGLang / llama.cpp / TRT-LLM default to.
MAX_NEW_TOKENS=36
# llama-bench (distinct from bench_llama_cpp.py) sweeps prefill/decode
# sizes for its own scaling table; not used for the head-to-head.
PP_SIZES="18,128,512"
TG_SIZES="36,128"
BENCH_REPS=5
WARMUP_REPS=2

# ── LD_PRELOAD workaround ──────────────────────────────────────────────────
# Anaconda's libstdc++ is often too old (missing GLIBCXX_3.4.32) for
# FlashInfer / sgl-kernel JIT-compiled .so files.  Prefer the system copy.
SYSTEM_LIBSTDCXX="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if [[ -f "$SYSTEM_LIBSTDCXX" ]]; then
    export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$SYSTEM_LIBSTDCXX"
fi

# ── Helpers ─────────────────────────────────────────────────────────────────
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
err()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; }
ts()   { date "+%Y-%m-%dT%H:%M:%S%z"; }

# Resolve the Python interpreter for a given venv.
# Falls back to the system python3 if the venv doesn't exist.
venv_python() {
    local env_name="$1"
    local venv_dir="$BENCH_ENVS_DIR/$env_name"
    if [[ -x "$venv_dir/bin/python3" ]]; then
        echo "$venv_dir/bin/python3"
    else
        echo "python3"
    fi
}

mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary_$(date +%Y%m%d_%H%M%S).txt"

{
echo "═══════════════════════════════════════════════════════════════"
echo " Qwen3-4B FP16  –  Single-GPU Benchmark Suite"
echo " $(ts)"
echo "═══════════════════════════════════════════════════════════════"

# ── GPU info ────────────────────────────────────────────────────────────
log "GPU information"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
           --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
echo ""
nvidia-smi -q -d CLOCK 2>/dev/null | head -30 || true

# ════════════════════════════════════════════════════════════════════════
# 1.  GROUT
# ════════════════════════════════════════════════════════════════════════
log "Benchmark: grout"

if [[ -f "$GROUT_DIR/Cargo.toml" ]]; then
    echo "Building grout (release) …"
    (cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench 2>&1 | tail -3)

    echo ""
    echo "Running grout  (prompt=\"$PROMPT\", max-new-tokens=$MAX_NEW_TOKENS, reps=$BENCH_REPS, warmup=$WARMUP_REPS) …"
    echo "─────────────────────────────────────────────────────────────"
    # Grout chat-templates the raw prompt internally to match the
    # 18-token pre-templated form the other bench scripts use by default.
    (cd "$GROUT_DIR" && cargo run --release --features benchmarks --bin grout_bench -- \
        --model "$MODEL_HF" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --reps "$BENCH_REPS" \
        --warmup-reps "$WARMUP_REPS" 2>&1) \
    | tee "$RESULTS_DIR/grout.txt"
    echo "─────────────────────────────────────────────────────────────"
else
    err "grout Cargo.toml not found at $GROUT_DIR – skipping"
fi

# ════════════════════════════════════════════════════════════════════════
# 2.  LLAMA.CPP  (build if needed, convert model if needed)
# ════════════════════════════════════════════════════════════════════════
log "Benchmark: llama.cpp"

# 2a. Clone + build ──────────────────────────────────────────────────────
if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    echo "Cloning llama.cpp …"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi

LLAMA_BENCH="$LLAMA_CPP_DIR/build/bin/llama-bench"
LLAMA_SERVER="$LLAMA_CPP_DIR/build/bin/llama-server"
if [[ ! -x "$LLAMA_BENCH" || ! -x "$LLAMA_SERVER" ]]; then
    echo "Building llama.cpp (CUDA, Release) …"
    # Detect compute capability (e.g. "12.0" -> "120")
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
                | head -1 | tr -d '.' | head -c3)
    CUDA_ARCH="${CUDA_ARCH:-120}"

    cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5

    cmake --build "$LLAMA_CPP_DIR/build" --config Release \
        -j"$(nproc)" --target llama-bench llama-server 2>&1 | tail -5
fi

# 2b. Convert HF → GGUF (f16) if needed ─────────────────────────────────
if [[ ! -f "$GGUF_PATH" ]]; then
    echo "Converting HF model to GGUF (f16) …"
    echo "  (requires: pip install transformers torch gguf sentencepiece)"
    python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
        "$MODEL_HF" --outtype f16 --outfile "$GGUF_PATH"
fi

# 2c. Run llama-bench ────────────────────────────────────────────────────
echo ""
echo "Running llama-bench  (no flash-attn, pp=$PP_SIZES, tg=$TG_SIZES, reps=$BENCH_REPS) …"
echo "─────────────────────────────────────────────────────────────"
"$LLAMA_BENCH" \
    -m "$GGUF_PATH" -ngl 99 \
    -p "$PP_SIZES" -n "$TG_SIZES" \
    -r "$BENCH_REPS" 2>&1 \
| tee "$RESULTS_DIR/llama_cpp_no_fa.txt"
echo "─────────────────────────────────────────────────────────────"

echo ""
echo "Running llama-bench  (flash-attn ON, pp=$PP_SIZES, tg=$TG_SIZES, reps=$BENCH_REPS) …"
echo "─────────────────────────────────────────────────────────────"
"$LLAMA_BENCH" \
    -m "$GGUF_PATH" -ngl 99 -fa 1 \
    -p "$PP_SIZES" -n "$TG_SIZES" \
    -r "$BENCH_REPS" 2>&1 \
| tee "$RESULTS_DIR/llama_cpp_fa.txt"
echo "─────────────────────────────────────────────────────────────"

# 2d. Apples-to-apples single-request bench via llama-server ─────────────
# Matches bench_vllm.py / bench_sglang.py methodology: same 18-token
# pre-templated Qwen3 chat prompt, 36 tokens forced (ignore_eos), 5 reps +
# 1 warmup, wall-clock elapsed per request.
if [[ -x "$LLAMA_SERVER" ]]; then
    echo ""
    echo "Running bench_llama_cpp.py  (apples-to-apples, fa OFF + ON) …"
    echo "─────────────────────────────────────────────────────────────"
    # No --prompt: use bench_llama_cpp.py's default 18-token templated
    # Qwen3 chat prompt (matches grout's internal templating + vLLM/SGLang/TRT-LLM defaults).
    python3 "$SCRIPT_DIR/bench_llama_cpp.py" \
        --llama-server "$LLAMA_SERVER" \
        --gguf "$GGUF_PATH" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --reps "$BENCH_REPS" \
        --warmup-reps "$WARMUP_REPS" 2>&1 \
    | tee "$RESULTS_DIR/llama_cpp.txt"
    echo "─────────────────────────────────────────────────────────────"
else
    err "llama-server not built at $LLAMA_SERVER – skipping apples-to-apples bench"
fi

# ════════════════════════════════════════════════════════════════════════
# 3.  VLLM  (runs in its own venv)
# ════════════════════════════════════════════════════════════════════════
log "Benchmark: vLLM"

VLLM_PYTHON="$(venv_python vllm_env)"
if "$VLLM_PYTHON" -c "import vllm" 2>/dev/null; then
    echo "Using Python: $VLLM_PYTHON"
    echo "Running vLLM benchmark …"
    echo "─────────────────────────────────────────────────────────────"
    # No --prompt: bench_vllm.py's default is the 18-token templated
    # Qwen3 chat prompt. Passing --prompt "Hello, how are you?" would
    # override that default with a raw 6-token string — not apples-to-apples.
    "$VLLM_PYTHON" "$SCRIPT_DIR/bench_vllm.py" \
        --model "$MODEL_HF" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --reps "$BENCH_REPS" 2>&1 \
    | tee "$RESULTS_DIR/vllm.txt"
    echo "─────────────────────────────────────────────────────────────"
else
    err "vllm not importable – skipping"
    err "  To set up:  python3 -m venv $BENCH_ENVS_DIR/vllm_env && $BENCH_ENVS_DIR/vllm_env/bin/pip install vllm"
fi

# ════════════════════════════════════════════════════════════════════════
# 4.  SGLANG  (runs in its own venv)
# ════════════════════════════════════════════════════════════════════════
log "Benchmark: SGLang"

SGLANG_PYTHON="$(venv_python sglang_env)"
if "$SGLANG_PYTHON" -c "import sglang" 2>/dev/null; then
    echo "Using Python: $SGLANG_PYTHON"
    echo "Running SGLang benchmark …"
    echo "─────────────────────────────────────────────────────────────"
    # Same as vLLM: let bench_sglang.py use its default 18-token templated prompt.
    "$SGLANG_PYTHON" "$SCRIPT_DIR/bench_sglang.py" \
        --model "$MODEL_HF" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --reps "$BENCH_REPS" 2>&1 \
    | tee "$RESULTS_DIR/sglang.txt"
    echo "─────────────────────────────────────────────────────────────"
else
    err "sglang not importable – skipping"
    err "  To set up:  python3 -m venv $BENCH_ENVS_DIR/sglang_env && $BENCH_ENVS_DIR/sglang_env/bin/pip install 'sglang[all]'"
fi

# ════════════════════════════════════════════════════════════════════════
# 5.  TRT-LLM  (runs in its own venv — same pattern as vLLM/SGLang)
# ════════════════════════════════════════════════════════════════════════
log "Benchmark: TRT-LLM"

TRTLLM_PYTHON="$(venv_python trtllm_env)"
TRTLLM_ENV_BIN="$BENCH_ENVS_DIR/trtllm_env/bin"
if "$TRTLLM_PYTHON" -c "import tensorrt_llm" 2>/dev/null; then
    echo "Using Python: $TRTLLM_PYTHON"

    # Both backends route through TRT-LLM's MpiPoolSession on LLM(...) even
    # for single-process/single-GPU use, which calls MPI.COMM_SELF.Spawn
    # and blows up on most systems with "OPAL ERROR: Error in file dpm/dpm.c".
    # The trtllm-llmapi-launch wrapper sets TLLM_SPAWN_PROXY_PROCESS=1 and
    # runs an MGMN leader + comm server so Spawn is replaced with a proxied
    # launch. Prepend the env's bin to PATH so the launcher's own
    # `python3 -m tensorrt_llm.llmapi.mgmn_*_node` calls resolve to the
    # interpreter that has tensorrt_llm installed.
    if [[ ! -x "$TRTLLM_ENV_BIN/trtllm-llmapi-launch" ]]; then
        err "trtllm-llmapi-launch not found at $TRTLLM_ENV_BIN – skipping TRT-LLM"
    else
        # TRT-engine backend only: on Qwen3-4B / RTX 5090 the PyTorch
        # backend was strictly worse (~3.8 t/s decode, ~5.6 t/s e2e) and
        # doubled setup time for a slower number, so it was dropped
        # (pass --backend pytorch to bench_trtllm.py manually if you want
        # to re-measure).
        echo ""
        echo "Running TRT-LLM benchmark (backend=trt) …"
        echo "─────────────────────────────────────────────────────────────"
        PATH="$TRTLLM_ENV_BIN:$PATH" \
            "$TRTLLM_ENV_BIN/trtllm-llmapi-launch" \
            "$TRTLLM_PYTHON" "$SCRIPT_DIR/bench_trtllm.py" \
            --model "$MODEL_HF" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --reps "$BENCH_REPS" \
            --backend trt 2>&1 \
        | tee "$RESULTS_DIR/trtllm.txt"
        echo "─────────────────────────────────────────────────────────────"
    fi
else
    err "tensorrt_llm not importable from $TRTLLM_PYTHON – skipping"
    err "  Expected layout: $BENCH_ENVS_DIR/trtllm_env/bin/python3 with tensorrt-llm installed"
    err "  (A symlink to a conda env works: ln -sfn \$SCRATCH/anaconda3/envs/trtllm $BENCH_ENVS_DIR/trtllm_env)"
fi

# ════════════════════════════════════════════════════════════════════════
# Done
# ════════════════════════════════════════════════════════════════════════
echo ""
log "All benchmarks complete.  Results saved under $RESULTS_DIR/"

} 2>&1 | tee "$SUMMARY"
