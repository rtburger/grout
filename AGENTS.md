Operating rules for the agent, verbatim:

1. Never touch attention kernels, StepGraph machinery, or CUDA-graph capture code. Change what feeds them, never how they run.
2. New code goes in new files; edits to upstream files are minimal and mechanical. Every phase = one PR, cargo test green plus the phase gate, or not done.
3. No dependency upgrades. Toolchain pinned in rust-toolchain.toml; cutile-rs rev pinned in Cargo.lock.
4. Supported ggml types: Q4_K, Q5_K, Q6_K, Q8_0, F16, F32. Error loudly on anything else. Port dequant math from llama.cpp/candle — never invent block-decode logic.
5. Hardware truth: 4070, 12 GB, sm_89, 46 SMs. VRAM preflight at load: weights + KV-at-max-ctx + scratch + 700 MB slack vs free VRAM, fail fast with numbers. Context defaults: 16k (Qwen3-8B, R1-distill), 32k (Qwen3-4B, Coder-7B).
6. Non-goals (refuse scope creep): paged KV, prefix cache, KV quantization, speculative decoding, MMQ prefill kernels, batching, server/API, CPU/Metal backends, MoE, multi-GPU, device-side full sampler, async in the engine.
7. /home/rtb/code/agent/candle is read-only reference. Copying requires an attribution header naming the source file.
8. Test split: default `cargo test` is CPU-only. GPU tests are `#[ignore]`d and run with the device visible via `cargo test -- --ignored`.
9. Gate wording: builds are prerequisites, not gate conditions. Any gate that says "unchanged output" means executed generation compared against the reference output; compiling binaries alone never satisfies it.
10. Roofline expectation before Phase 1: Grout fp16 Qwen3-4B at 55.4 tok/s reads ~8.05 GB/token, ~446 GB/s effective, ~89% of the 4070's 504 GB/s roofline. If quantized Grout retains ~85% memory efficiency after ~3.2x smaller weight reads, expected decode is ~170 tok/s on Qwen3-4B Q4_K_M and ~87 tok/s on Qwen3-8B Q4_K_M.
11. Phase 2 hard gate: quantized Grout decode throughput on this 4070 must be >=0.85x llama.cpp and >=1.15x pi-ai-candle. With Phase 0 baselines, that means Qwen3-4B Q4_K_M >=135 tok/s and Qwen3-8B Q4_K_M >=88 tok/s. Target is >=1.0x llama.cpp: Qwen3-4B >=149.1 tok/s, Qwen3-8B >=90.0 tok/s.
12. Benchmark reporting: every future Grout benchmark row must state headless/desktop mode alongside driver, CUDA toolkit, and tileiras versions. Do not compare Grout rows missing those fields.
