# 4070 Baseline — Phase 0

Date: 2026-07-03

## Host/toolchain

- GPU: NVIDIA GeForce RTX 4070, 12 GB, sm_89
- Driver: 595.58.03 — PASS (>= r580)
- `nvidia-smi` reported CUDA runtime capability: 13.2
- Arch `cuda` package: 13.3.1-1
- `/opt/cuda/version.json`: not shipped by the Arch package
- `/opt/cuda/bin/nvcc --version`: CUDA 13.3, V13.3.73
- `tileiras`: `/opt/cuda/bin/tileiras`
- `/opt/cuda/bin/tileiras --version`: CUDA Tile IR assembler 13.3, V13.3.36
- `.cargo/config.toml` pins `CUDA_TOOLKIT_PATH=/opt/cuda` and `CUTILE_TILEIRAS_PATH=/opt/cuda/bin/tileiras`
- Rust: `rustc 1.93.1 (01f6ddf75 2026-02-11)`
- LLVM check: `/usr/bin/llvm-config` is 22.1.3; `/usr/bin/llvm-config-21` is 21.1.8. This crate has no `melior`, `llvm-sys`, or `mlir` entries in `Cargo.lock`.
- grout rev: Phase 0 commit containing this file
- llama.cpp rev: `2d97363`, built with `-DGGML_CUDA=ON`
- pi-rs / pi-ai-candle rev: `d0f7638`

## Commands used

```bash
cargo clean
cargo build --release --bin grout
cargo build --release --features benchmarks --bin grout_bench
cargo test
cargo test -- --ignored

target/release/grout_bench \
  --model ../hf_models/qwen3_4b \
  --prompt 'Write one paragraph about compiler design, continuing until the token limit.' \
  --max-new-tokens 36 --reps 5 --warmup-reps 2 --ignore-eos --quiet

python3 benchmarks/bench_llama_cpp.py \
  --llama-server ../llama.cpp/build/bin/llama-server \
  --gguf <Qwen3 GGUF> --mode fa --max-new-tokens 36 --reps 5 --warmup-reps 2
```

Candle numbers came from the current `pi-ai-candle` provider with `PI_CANDLE_TIMING=1 PI_CANDLE_SYNC_TIMING=1`, 1 warmup + 5 measured generations, greedy sampling, `max_tokens=36`.

## Phase 0 gate verification

- Lib/bin split builds with `cargo build --release --bin grout`.
- Benchmark binary builds with `cargo build --release --features benchmarks --bin grout_bench`.
- Default test gate is CPU-only: `cargo test`.
- GPU ignored-test gate passed with device visible: `cargo test -- --ignored`.
- CLI output check executed the pre-split binary at `23ae5a3` and the Phase 0 binary on `Hello, how are you?`, `--max-new-tokens 8`, `--warmup-reps 0`.
- Generated text matched exactly: `Hello! I'm just a language model`.
- Non-timing stdout matched after model-path normalization; stderr was empty for both runs.

## Grout fp16 Qwen3-4B

Run mode: headless. Driver/toolkit/tileiras versions are listed in Host/toolchain.

Pre-run VRAM for the recorded Grout run: 3050 MiB used, 8781 MiB free.

```text
JIT compile + graph capture...
  done: decode_phase_tps=54.9, request_gen_tps=13.2, prompt_s=0.899
  [warmup] run 1/2: prompt_tokens=25, gen_tokens=36, prefill_ms=20.49, decode_ms=647.88, e2e_ms=668.39, decode_phase_tps=55.6, request_gen_tps=53.9, e2e_tps=91.3
  [warmup] run 2/2: prompt_tokens=25, gen_tokens=36, prefill_ms=20.57, decode_ms=646.53, e2e_ms=667.11, decode_phase_tps=55.7, request_gen_tps=54.0, e2e_tps=91.4

  [timed] run 1/5: prompt_tokens=25, gen_tokens=36, prefill_ms=20.26, decode_ms=653.00, e2e_ms=673.28, decode_phase_tps=55.1, request_gen_tps=53.5, e2e_tps=90.6
  [timed] run 2/5: prompt_tokens=25, gen_tokens=36, prefill_ms=20.08, decode_ms=648.70, e2e_ms=668.78, decode_phase_tps=55.5, request_gen_tps=53.8, e2e_tps=91.2
  [timed] run 3/5: prompt_tokens=25, gen_tokens=36, prefill_ms=20.56, decode_ms=646.56, e2e_ms=667.14, decode_phase_tps=55.7, request_gen_tps=54.0, e2e_tps=91.4
  [timed] run 4/5: prompt_tokens=25, gen_tokens=36, prefill_ms=20.55, decode_ms=654.31, e2e_ms=674.86, decode_phase_tps=55.0, request_gen_tps=53.3, e2e_tps=90.4
  [timed] run 5/5: prompt_tokens=25, gen_tokens=36, prefill_ms=20.15, decode_ms=648.31, e2e_ms=668.46, decode_phase_tps=55.5, request_gen_tps=53.9, e2e_tps=91.3
  [grout] mean over 5 runs: decode_phase_tps=55.4, request_gen_tps=53.7, e2e_tps=91.0, elapsed=0.671s
```

## Throughput baselines

Prompt for llama.cpp: `Write one paragraph about compiler design, continuing until the token limit.`

Prompt for Grout and pi-ai-candle: same user text through each engine's current Qwen chat template.

| engine | model | dtype/quant | ctx | prompt toks | gen toks | prefill ms | decode ms | decode t/s | request gen t/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grout | Qwen3-4B | fp16 | 4096 | 25 | 36 | 20.26 median | 648.70 median | 55.4 mean | 53.7 mean |
| llama.cpp flash-attn | Qwen3-4B | Q4_K_M | 32768 | 13 | 36 | 10.91 median | 241.09 median | 149.1 mean | 141.9 mean |
| llama.cpp flash-attn | Qwen3-8B | Q4_K_M | 16384 | 13 | 36 | 17.73 median | 398.10 median | 90.0 mean | 85.9 mean |
| pi-ai-candle | Qwen3-4B | Q4_K_M | 32768 | 25 | 36 | 13 median | 306 median | 117.7 mean | 107.5 median |
| pi-ai-candle | Qwen3-8B | Q4_K_M | 32768 | 25 | 36 | 19 median | 469 median | 76.8 mean | 71.7 median |

## Roofline notes

Local roofline uses the RTX 4070 memory bandwidth of 504 GB/s.

- Grout fp16 Qwen3-4B: ~8.05 GB weights/token * 55.4 tok/s = ~446 GB/s effective, ~89% of roofline. Ceiling is ~62.6 tok/s, so the fp16 runtime is already near memory-roofline on sm_89.
- llama.cpp Qwen3-4B Q4_K_M: ~2.5 GB weights/token * 149.1 tok/s = ~373 GB/s effective, ~74% of roofline; ceiling is ~202 tok/s.
- llama.cpp Qwen3-8B Q4_K_M: ~5.0 GB weights/token * 90.0 tok/s = ~450 GB/s effective, ~89% of roofline; ceiling is ~101 tok/s.
- A single effective-bandwidth-plus-fixed-overhead model cannot fit the llama.cpp 4B/8B points: `(5.0 - 2.5) GB / (1/90.0 - 1/149.1) s = ~568 GB/s`, above this card's 504 GB/s. The consistent read is shape-dependent efficiency: 4B Q4 shapes leave headroom, 8B Q4 shapes are already near roofline.
- Quantized Grout expectation: Qwen3-4B Q4_K_M honest band 155-180 tok/s, with ~170 as midpoint. The midpoint carries a shape-efficiency caveat: a landing around 158 should trigger GEMV-shape investigation, not automatic failure or success.
- Quantized Grout expectation: Qwen3-8B Q4_K_M plausible band 85-92 tok/s. This is a parity race; demanding materially above llama.cpp's 90.0 tok/s demands near-impossible roofline headroom.
- The old "2x candle" Phase 2 gate is physically impossible on the 4B: 2 * 117.7 = 235.4 tok/s, above the ~202 tok/s 4B Q4_K_M roofline. The corrected Phase 2 gates are per-model in `AGENTS.md`: 4B hard >=135 tok/s, target >=149.1; 8B hard >=84 tok/s, target >=90.0.

## First-run kernel/JIT behavior

- Before re-running with CUDA 13.3, cuTile temp artifacts were cleared from `/tmp`.
- cuTile-rs 0.2.0 uses `CUTILE_TILEIRAS_PATH` if set; otherwise it finds `tileiras` on `PATH`.
- The persistent files emitted by cuTile are temporary UUID `.bc` / `.cubin` files in `env::temp_dir()`; the runtime function cache is thread-local/in-process.
- CUDA 13.2 `tileiras` failed on sm_89 with `error: invalid GPU architecture: 89`.
- CUDA 13.3 `tileiras` compiled and launched the Grout kernels on the 595.58.03 driver.
- llama.cpp used 2 warmups before measured reps. Timed reps were stable after warmup.
- pi-ai-candle first generation after model load was slower; measured reps after 1 warmup were stable.
