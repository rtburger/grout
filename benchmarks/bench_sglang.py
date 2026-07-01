#!/usr/bin/env python3
"""
Benchmark Qwen3-4B FP16 single-request decode performance with SGLang.

Uses SGLang's offline Engine API to measure single-request latency,
matching the methodology used in bench_vllm.py for fair comparison.

Usage:
    python bench_sglang.py --model ../hf_models/qwen3_4b
    python bench_sglang.py --model ../hf_models/qwen3_4b --prompt "Hello, how are you?" --max-new-tokens 36 --reps 5
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import sglang as sgl


def emit_jsonl(path: Optional[Path], record: dict) -> None:
    if path is None:
        return
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# -- helpers -------------------------------------------------------------------

@dataclass
class RunResult:
    prompt_tokens: int
    gen_tokens: int
    elapsed: float
    prefill_ms: Optional[float]  # time-to-first-token from streaming generate()

    @property
    def prompt_tps(self) -> float:
        return self.prompt_tokens / self.elapsed

    @property
    def request_gen_tps(self) -> float:
        return self.gen_tokens / self.elapsed

    @property
    def total_tps(self) -> float:
        return (self.prompt_tokens + self.gen_tokens) / self.elapsed


def bench_one(engine: sgl.Engine, prompt: str, sampling_params: dict) -> RunResult:
    # Use streaming to get time-to-first-token ≈ prefill_ms. sglang
    # doesn't expose a standalone prefill timing in meta_info for the
    # offline Engine, so we time the first chunk manually. The TTFT
    # includes the first decode step (~1-2 ms) on top of prefill —
    # matches vLLM's `first_token_time - first_scheduled_time` metric
    # so the cross-engine comparison stays apples-to-apples.
    start = time.perf_counter()
    first_chunk_t: Optional[float] = None
    final_output = None
    for chunk in engine.generate(prompt, sampling_params, stream=True):
        if first_chunk_t is None:
            first_chunk_t = time.perf_counter()
        final_output = chunk
    elapsed = time.perf_counter() - start

    if final_output is None:
        raise RuntimeError("sglang.Engine.generate returned no chunks")

    prompt_tokens = final_output["meta_info"]["prompt_tokens"]
    gen_tokens = final_output["meta_info"]["completion_tokens"]
    prefill_ms = (
        (first_chunk_t - start) * 1000.0 if first_chunk_t is not None else None
    )

    return RunResult(
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        elapsed=elapsed,
        prefill_ms=prefill_ms,
    )


def run_suite(
    engine: sgl.Engine,
    prompt: str,
    sampling_params: dict,
    reps: int,
    label: str,
    profile_cuda: bool = False,
    warmup_reps: int = 1,
    json_path: Optional[Path] = None,
    variant_name: Optional[str] = None,
    pp_label: int = 0,
    tg_label: int = 0,
) -> list[RunResult]:
    # warmup (discarded)
    for _ in range(max(1, warmup_reps)):
        bench_one(engine, prompt, sampling_params)

    # Bracket measured reps with cudaProfilerStart/Stop inside the scheduler
    # subprocess so nsys (--capture-range=cudaProfilerApi) captures kernels
    # from the actual CUDA context, not the tokenizer front-end process.
    if profile_cuda:
        engine.start_profile(activities=["CUDA_PROFILER"])

    results: list[RunResult] = []
    for i in range(reps):
        r = bench_one(engine, prompt, sampling_params)
        results.append(r)
        prefill_str = (
            f", prefill_ms={r.prefill_ms:.2f}" if r.prefill_ms is not None else ""
        )
        print(
            f"  [{label}] run {i+1}/{reps}: "
            f"prompt_tokens={r.prompt_tokens}, gen_tokens={r.gen_tokens}, "
            f"elapsed={r.elapsed:.3f}s{prefill_str}, "
            f"request_gen_tps={r.request_gen_tps:.1f}, "
            f"total_tps={r.total_tps:.1f}"
        )
        record = {
            "engine": "sglang",
            "variant": variant_name or label,
            "pp": pp_label,
            "tg": tg_label,
            "rep": i,
            "prompt_tokens": r.prompt_tokens,
            "gen_tokens": r.gen_tokens,
            "e2e_ms": r.elapsed * 1000.0,
        }
        if r.prefill_ms is not None:
            record["prefill_ms"] = r.prefill_ms
        emit_jsonl(json_path, record)

    if profile_cuda:
        engine.stop_profile()

    mean_request_gen = sum(r.request_gen_tps for r in results) / len(results)
    mean_total = sum(r.total_tps for r in results) / len(results)
    mean_elapsed = sum(r.elapsed for r in results) / len(results)
    print(
        f"  [{label}] mean over {reps} runs: "
        f"request_gen_tps={mean_request_gen:.1f}, total_tps={mean_total:.1f}, "
        f"elapsed={mean_elapsed:.3f}s"
    )
    return results


# -- main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SGLang single-request FP16 benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to HF model directory")
    # Pre-templated Qwen3 chat prompt (18 tokens) for fair cross-engine comparison.
    default_prompt = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    parser.add_argument("--prompt", type=str, default=default_prompt)
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read prompt text from file. Overrides --prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=36)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--mem-fraction", type=float, default=0.9,
                        help="Fraction of GPU memory to use")
    parser.add_argument("--context-length", type=int, default=0,
                        help="Override SGLang context_length. Leave 0 for model default.")
    parser.add_argument("--mode", choices=["all", "default", "no-radix"], default="all",
                        help="Which engine configuration(s) to run. 'default' "
                             "loads the engine once with CUDA graphs + radix "
                             "cache; useful when profiling so the trace only "
                             "covers a single engine lifecycle.")
    parser.add_argument("--profile-cuda", action="store_true",
                        help="Bracket measured reps with engine.start_profile("
                             "activities=['CUDA_PROFILER']) / stop_profile(). "
                             "Combine with nsys --capture-range=cudaProfilerApi "
                             "so the subprocess where CUDA actually runs is the "
                             "one that toggles CUPTI capture.")
    parser.add_argument("--warmup-reps", type=int, default=3)
    parser.add_argument("--json", type=str, default=None,
                        help="Append JSONL per measured run to this file.")
    parser.add_argument("--pp-label", type=int, default=0,
                        help="Label for `pp` (prompt length) field in JSON.")
    args = parser.parse_args()
    json_path = Path(args.json) if args.json else None
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text()

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
    }

    # -- default mode (CUDA graphs enabled by default in SGLang) ---------------
    if args.mode in ("all", "default"):
        print(f"\n{'='*65}")
        print(f" SGLang benchmark: default (CUDA graphs + RadixAttention)")
        print(f"{'='*65}")
        engine = sgl.Engine(
            model_path=args.model,
            dtype="float16",
            mem_fraction_static=args.mem_fraction,
            **({"context_length": args.context_length} if args.context_length > 0 else {}),
        )
        run_suite(engine, args.prompt, sampling_params, args.reps, "sglang-default",
                  profile_cuda=args.profile_cuda,
                  warmup_reps=args.warmup_reps,
                  json_path=json_path,
                  variant_name="default",
                  pp_label=args.pp_label,
                  tg_label=args.max_new_tokens)

        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    # -- disable radix cache to measure raw decode -----------------------------
    if args.mode in ("all", "no-radix"):
        print(f"\n{'='*65}")
        print(f" SGLang benchmark: disable_radix_cache=True")
        print(f"{'='*65}")
        engine_no_cache = sgl.Engine(
            model_path=args.model,
            dtype="float16",
            mem_fraction_static=args.mem_fraction,
            disable_radix_cache=True,
            **({"context_length": args.context_length} if args.context_length > 0 else {}),
        )
        run_suite(engine_no_cache, args.prompt, sampling_params, args.reps, "sglang-no-radix",
                  profile_cuda=args.profile_cuda,
                  warmup_reps=args.warmup_reps,
                  json_path=json_path,
                  variant_name="no-radix",
                  pp_label=args.pp_label,
                  tg_label=args.max_new_tokens)

        engine_no_cache.shutdown()
        del engine_no_cache
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*65}")
    print(" Done.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
