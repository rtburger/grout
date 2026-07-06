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

## Phase 1 GGUF fp16-compute gate

Run mode for this GGUF row: desktop/display-attached.

Version block for this row:

- Driver: 595.58.03
- CUDA toolkit: Arch `cuda` package 13.3.1-1; `/opt/cuda/bin/nvcc` CUDA 13.3, V13.3.73
- tileiras: `/opt/cuda/bin/tileiras`, CUDA Tile IR assembler 13.3, V13.3.36

Command:

```bash
target/release/grout_bench \
  --model ../hf_models/qwen3_4b_gguf/Qwen3-4B-Q4_K_M.gguf \
  --prompt 'Write one paragraph about compiler design, continuing until the token limit.' \
  --max-new-tokens 36 --reps 5 --warmup-reps 2 --ignore-eos --quiet
```

Result after GGUF Q4_K_M CPU dequantize-to-fp16 and unchanged engine upload:

```text
  [grout] mean over 5 runs: decode_phase_tps=54.4, request_gen_tps=52.8, e2e_tps=89.4, elapsed=0.682s
```

This is within +/-3% of the Phase 0 fp16 Qwen3-4B baseline (`55.4 tok/s` decode). The ignored 4B GGUF integration test also passed with the same file and adjacent tokenizer.

Tier 2 GGUF integration commands run:

```bash
GROUT_QWEN3_06B_GGUF=../hf_models/qwen3_06b_gguf/Qwen3-0.6B-Q4_K_M.gguf \
cargo test qwen3_06b_gguf_generates_100_greedy_tokens -- --ignored --nocapture

GROUT_QWEN3_4B_Q4_K_M_GGUF=../hf_models/qwen3_4b_gguf/Qwen3-4B-Q4_K_M.gguf \
cargo test qwen3_4b_q4_k_m_gguf_generates_100_greedy_tokens -- --ignored --nocapture
```

Phase 1 closeout notes:

- `src/model.rs` changed only in `Qwen3Engine::load`: instantiate `WeightLoader` before config load, use GGUF metadata config when present, and locate adjacent `tokenizer.json` for `.gguf` files.
- This could not live entirely in `loader.rs` because the engine constructor owns config selection, generation-config loading, tokenizer loading, and the `Qwen3Engine` fields; no forward pass, attention, StepGraph, or CUDA graph code changed.
- Speed-gate inequality: desktop/display-attached result is `54.4 / 55.4 = 98.2%` of the headless Phase 0 baseline. Since desktop overhead is non-negative, loader regression is at most 1.8%, below the 3% gate.
- Tier 1 coverage: default `cargo test` covers Q4_K, Q5_K, Q6_K, Q8_0, F16, and F32 bit-exact f32 dequant vs Candle CPU dequant, plus loud unsupported-type error coverage using a synthetic Q2_K GGUF tensor header.
- Tier 2 coverage: ignored integration tests passed for Qwen3-0.6B Q4_K_M GGUF and Qwen3-4B Q4_K_M GGUF, each generating 100 greedy tokens and passing coherence checks.
- Conditional token-exact 0.6B GGUF-vs-safetensors test status: skipped for lack of local matching F16/BF16 GGUF plus safetensors files. Quantized GGUF output is not compared token-for-token to fp16/bf16 safetensors.
- Dependency delta: `Cargo.toml` adds only `candle-core = { path = "../candle/candle-core" }` under `[dev-dependencies]`. `Cargo.lock` records `candle-core` 0.11.0 and its transitive dev-only dependency graph; no release dependency was added for Phase 1.

## Phase 2 Task 1 standalone quantized GEMV microbench

Run mode for this microbench row: desktop/display-attached.

Version block for this row:

- Driver: 595.58.03
- CUDA toolkit: Arch `cuda` package 13.3.1-1; `/opt/cuda/bin/nvcc` CUDA 13.3, V13.3.73
- tileiras: `/opt/cuda/bin/tileiras`, CUDA Tile IR assembler 13.3, V13.3.36

Command:

```bash
target/release/quant_gemv_microbench \
  --gguf ../hf_models/qwen3_4b_gguf/Qwen3-4B-Q4_K_M.gguf \
  --gguf ../hf_models/qwen3_8b_gguf/Qwen3-8B-Q4_K_M.gguf \
  --iters 20 --warmup-iters 5
```

The harness reads tensor shapes and dtypes from each GGUF tensor table; no model shapes are hardcoded. Bytes for GB/s are `quantized_weight_bytes + fp16_activation_bytes + fp16_output_bytes`. These are standalone benchmark kernels only; the engine still has no quantized resident weights.

| model | tensor kind | tensor | dtype | rows | k | achieved GB/s |
|---|---|---|---:|---:|---:|---:|
| Qwen3-4B-Q4_K_M | attn_q | blk.0.attn_q.weight | Q4_K | 4096 | 2560 | 79.354 |
| Qwen3-4B-Q4_K_M | attn_k | blk.0.attn_k.weight | Q4_K | 1024 | 2560 | 70.964 |
| Qwen3-4B-Q4_K_M | attn_v | blk.0.attn_v.weight | Q6_K | 1024 | 2560 | 90.978 |
| Qwen3-4B-Q4_K_M | attn_output | blk.0.attn_output.weight | Q4_K | 2560 | 4096 | 83.067 |
| Qwen3-4B-Q4_K_M | ffn_gate | blk.0.ffn_gate.weight | Q4_K | 9728 | 2560 | 76.367 |
| Qwen3-4B-Q4_K_M | ffn_up | blk.0.ffn_up.weight | Q4_K | 9728 | 2560 | 80.378 |
| Qwen3-4B-Q4_K_M | ffn_down | blk.0.ffn_down.weight | Q6_K | 2560 | 9728 | 111.245 |
| Qwen3-4B-Q4_K_M | attn_v | blk.4.attn_v.weight | Q4_K | 1024 | 2560 | 71.106 |
| Qwen3-4B-Q4_K_M | ffn_down | blk.4.ffn_down.weight | Q4_K | 2560 | 9728 | 87.454 |
| Qwen3-4B-Q4_K_M | lm_head | token_embd.weight | Q6_K | 151936 | 2560 | 82.989 |
| Qwen3-8B-Q4_K_M | attn_q | blk.0.attn_q.weight | Q4_K | 4096 | 4096 | 93.964 |
| Qwen3-8B-Q4_K_M | attn_k | blk.0.attn_k.weight | Q4_K | 1024 | 4096 | 86.218 |
| Qwen3-8B-Q4_K_M | attn_v | blk.0.attn_v.weight | Q6_K | 1024 | 4096 | 108.754 |
| Qwen3-8B-Q4_K_M | attn_output | blk.0.attn_output.weight | Q4_K | 4096 | 4096 | 91.892 |
| Qwen3-8B-Q4_K_M | ffn_gate | blk.0.ffn_gate.weight | Q4_K | 12288 | 4096 | 96.329 |
| Qwen3-8B-Q4_K_M | ffn_up | blk.0.ffn_up.weight | Q4_K | 12288 | 4096 | 91.221 |
| Qwen3-8B-Q4_K_M | ffn_down | blk.0.ffn_down.weight | Q6_K | 4096 | 12288 | 100.907 |
| Qwen3-8B-Q4_K_M | attn_v | blk.4.attn_v.weight | Q4_K | 1024 | 4096 | 85.863 |
| Qwen3-8B-Q4_K_M | ffn_down | blk.4.ffn_down.weight | Q4_K | 4096 | 12288 | 99.288 |
| Qwen3-8B-Q4_K_M | lm_head | output.weight | Q6_K | 151936 | 4096 | 89.318 |

## First-run kernel/JIT behavior

- Before re-running with CUDA 13.3, cuTile temp artifacts were cleared from `/tmp`.
- cuTile-rs 0.2.0 uses `CUTILE_TILEIRAS_PATH` if set; otherwise it finds `tileiras` on `PATH`.
- The persistent files emitted by cuTile are temporary UUID `.bc` / `.cubin` files in `env::temp_dir()`; the runtime function cache is thread-local/in-process.
- CUDA 13.2 `tileiras` failed on sm_89 with `error: invalid GPU architecture: 89`.
- CUDA 13.3 `tileiras` compiled and launched the Grout kernels on the 595.58.03 driver.
- llama.cpp used 2 warmups before measured reps. Timed reps were stable after warmup.
- pi-ai-candle first generation after model load was slower; measured reps after 1 warmup were stable.

## Phase 2 Q8_0 raw decode-GEMV checkpoint

Run mode for this synthetic Q8_0 checkpoint: desktop/display-attached.

Version block:

- Driver: 595.58.03
- CUDA toolkit: Arch `cuda` package 13.3.1-1; `/opt/cuda/bin/nvcc` CUDA 13.3, V13.3.73
- tileiras: `/opt/cuda/bin/tileiras`, CUDA Tile IR assembler 13.3, V13.3.36
- GPU: NVIDIA GeForce RTX 4070, 12 GB, sm_89, 46 SMs

Context: cuTile 0.2.0 generated launchers hard-code `block_dim=(1,1,1)`, which made the
first Q8_0 8B-shape signal land around 7-8 GB/s. The Q8_0 checkpoint below is the
new multi-row CUDA decode-GEMV backend in `src/kernels.rs` (`q8_0_gemv_r4t64`), measured
with synthetic GGUF-native Q8_0 rows at Qwen3-8B matrix shapes. Bytes are
`quantized_weight_bytes + fp16_activation_bytes + fp16_output_bytes`.

| dtype | rows | k | avg ms | achieved GB/s |
|---|---:|---:|---:|---:|
| Q8_0 | 4096 | 4096 | 0.046 | 391.9 |
| Q8_0 | 1024 | 4096 | 0.014 | 328.2 |
| Q8_0 | 12288 | 4096 | 0.136 | 392.8 |
| Q8_0 | 4096 | 12288 | 0.149 | 359.6 |
| Q8_0 | 151936 | 4096 | 1.632 | 405.4 |

Correctness gate added for this backend:

```bash
cargo test raw_gemv_q8_0_f16_matches_cpu --test kernels -- --ignored --nocapture
```

## Phase 2 Q8_0 SoA tile-native decode-GEMV checkpoint

Run mode for this synthetic Q8_0 checkpoint: desktop/display-attached.

Version block:

- Driver: 595.58.03
- CUDA toolkit: Arch `cuda` package 13.3.1-1; `/opt/cuda/bin/nvcc` CUDA 13.3, V13.3.73
- tileiras: `/opt/cuda/bin/tileiras`, CUDA Tile IR assembler 13.3, V13.3.36
- GPU: NVIDIA GeForce RTX 4070, 12 GB, sm_89, 46 SMs

Context: Q8_0 runtime weights are repacked at GGUF load into SoA device tensors
(`qs: [rows,k] i8`, `scales: [rows,k/32] f16`) for decode GEMV, while native bytes
are retained for unported dequant/embed paths. Decode uses the tile-native/TMA kernel
`gemv_q8_0_soa_f16` with `R=8`, `BK=512`, `block_scales=16`, `LATENCY=1`.
The synthetic microbench sweeps cuTile compile-option occupancy `{1,2,4}` at
Qwen3-8B matrix shapes. Bytes are `qs bytes + scale bytes + fp16_activation_bytes + fp16_output_bytes`.

Command:

```bash
CUDA_TOOLKIT_PATH=/opt/cuda CUTILE_TILEIRAS_PATH=/opt/cuda/bin/tileiras \
  target/release/q8_0_soa_microbench --iters 20 --warmup-iters 5
```

| backend | rows | k | occupancy | avg us | achieved GB/s |
|---|---:|---:|---:|---:|---:|
| Q8_0 SoA tile | 4096 | 4096 | 1 | 28.518 | 625.637 |
| Q8_0 SoA tile | 1024 | 4096 | 1 | 7.373 | 605.833 |
| Q8_0 SoA tile | 12288 | 4096 | 1 | 131.317 | 407.489 |
| Q8_0 SoA tile | 4096 | 12288 | 1 | 135.227 | 395.705 |
| Q8_0 SoA tile | 151936 | 4096 | 1 | 1502.771 | 440.212 |
| Q8_0 SoA tile | 4096 | 4096 | 2 | 19.158 | 931.298 |
| Q8_0 SoA tile | 1024 | 4096 | 2 | 7.629 | 585.503 |
| Q8_0 SoA tile | 12288 | 4096 | 2 | 116.330 | 459.987 |
| Q8_0 SoA tile | 4096 | 12288 | 2 | 152.474 | 350.947 |
| Q8_0 SoA tile | 151936 | 4096 | 2 | 1489.078 | 444.260 |
| Q8_0 SoA tile | 4096 | 4096 | 4 | 21.402 | 833.684 |
| Q8_0 SoA tile | 1024 | 4096 | 4 | 8.194 | 545.144 |
| Q8_0 SoA tile | 12288 | 4096 | 4 | 116.634 | 458.788 |
| Q8_0 SoA tile | 4096 | 12288 | 4 | 136.038 | 393.346 |
| Q8_0 SoA tile | 151936 | 4096 | 4 | 1486.848 | 444.926 |

All swept Q8_0 SoA 8B-shape rows exceed the temporary `>300 GB/s` checkpoint.
The engine uses the occupancy-4 specialization for graph/eager Q8_0 GEMV dispatch.

Correctness gate added for this backend:

```bash
cargo test gemv_q8_0_soa_f16_matches_cpu --test kernels -- --ignored --nocapture
```

Cache residency note for the Q8_0 SoA checkpoint: the 4096 x 4096 row has a
~17 MiB SoA working set (`qs` + scales + activation/output), and the 1024 x
4096 row is smaller still, so both are L2-resident on the RTX 4070's 36 MiB L2.
Treat their >500 GB/s results as cache-assisted. The larger rows — 12288 x
4096, 4096 x 12288, and 151936 x 4096 — exceed L2 and are the DRAM-honest rows
for this checkpoint.

## Release 4B e2e wall-time note

Run mode for this row: desktop/display-attached.

Version block:

- Driver: 595.58.03
- CUDA toolkit: Arch `cuda` package 13.3.1-1; `/opt/cuda/bin/nvcc` CUDA 13.3, V13.3.73
- tileiras: `/opt/cuda/bin/tileiras`, CUDA Tile IR assembler 13.3, V13.3.36
- GPU: NVIDIA GeForce RTX 4070, 12 GB, sm_89, 46 SMs
- Rust: `rustc 1.93.1 (01f6ddf75 2026-02-11)`

Command timed with the Bash `time` keyword around the release CLI:

```bash
TIMEFORMAT='shell_wall_seconds=%R'
time target/release/grout \
  --model ../hf_models/qwen3_4b \
  --prompt 'Hello, how are you?' \
  --max-new-tokens 32 --max-ctx 512 --warmup-reps 0
```

Recorded release wall time: `shell_wall_seconds=20.782`.
The CLI reported `prompt_tokens=18`, `generated_tokens=32`, `prompt_s=1.980`,
`decode_s=8.121`, and `total_s=10.100`. This replaces the earlier debug-build
wall-time observation; it is release-mode GPU execution.
