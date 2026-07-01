#!/usr/bin/env python3
"""
Benchmark Qwen3-4B FP16 single-request decode performance with vLLM.

Measures both eager-mode and CUDA-graph-accelerated inference so the
numbers are directly comparable to grout and llama.cpp results.

Usage:
    python bench_vllm.py --model ../hf_models/qwen3_4b
    python bench_vllm.py --model ../hf_models/qwen3_4b --prompt "Hello, how are you?" --max-new-tokens 36 --reps 5
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
from vllm import LLM, SamplingParams


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
    prefill_ms: Optional[float]  # TTFT-style max_tokens=1 request timing

    @property
    def prompt_tps(self) -> float:
        return self.prompt_tokens / self.elapsed

    @property
    def request_gen_tps(self) -> float:
        return self.gen_tokens / self.elapsed

    @property
    def total_tps(self) -> float:
        return (self.prompt_tokens + self.gen_tokens) / self.elapsed


def bench_one(llm: LLM, prompt: str, sampling_params: SamplingParams) -> RunResult:
    # Measure prefill as TTFT via a separate max_tokens=1 run. vLLM's
    # RequestOutput.metrics isn't populated in the offline LLM.generate
    # path in recent versions, and streaming is only available via the
    # async engine. A dedicated TTFT run is the simplest reliable path —
    # matches sglang's streaming TTFT semantic (includes one decode step
    # on top of pure prefill). Requires prefix caching OFF at the LLM
    # init so the second run doesn't hit cache; sweep_pp.sh already
    # passes --no-prefix-cache for this reason.
    ttft_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        min_tokens=1,
    )
    start = time.perf_counter()
    llm.generate([prompt], ttft_params, use_tqdm=False)
    prefill_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    out = outputs[0]
    return RunResult(
        prompt_tokens=len(out.prompt_token_ids),
        gen_tokens=len(out.outputs[0].token_ids),
        elapsed=elapsed,
        prefill_ms=prefill_ms,
    )


def run_suite(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    reps: int,
    label: str,
    warmup_reps: int = 1,
    json_path: Optional[Path] = None,
    variant_name: Optional[str] = None,
    pp_label: int = 0,
    tg_label: int = 0,
) -> list[RunResult]:
    # warmup (discarded)
    for _ in range(max(1, warmup_reps)):
        bench_one(llm, prompt, sampling_params)

    results: list[RunResult] = []
    for i in range(reps):
        r = bench_one(llm, prompt, sampling_params)
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
            "engine": "vllm",
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
    parser = argparse.ArgumentParser(description="vLLM single-request FP16 benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to HF model directory")
    # Pre-templated Qwen3 chat prompt (18 tokens) for fair cross-engine comparison.
    default_prompt = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    parser.add_argument("--prompt", type=str, default=default_prompt)
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read prompt text from file. Overrides --prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=36)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--warmup-reps", type=int, default=3)
    parser.add_argument("--json", type=str, default=None,
                        help="Append JSONL per measured run to this file.")
    parser.add_argument("--pp-label", type=int, default=0,
                        help="Label for `pp` field in JSON (informational).")
    parser.add_argument("--mode", choices=["all", "eager", "cuda-graph"], default="all",
                        help="Which variant(s) to run.")
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="Disable vLLM's automatic prefix caching. Use for "
                             "apples-to-apples prefill measurements where "
                             "repeated prompts would otherwise hit cache.")
    args = parser.parse_args()
    json_path = Path(args.json) if args.json else None
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text()

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
    )

    # -- eager mode (no CUDA graphs) ------------------------------------------
    if args.mode in ("all", "eager"):
        print(f"\n{'='*65}")
        pfx_label = " (prefix-cache OFF)" if args.no_prefix_cache else ""
        print(f" vLLM benchmark: enforce_eager=True (no CUDA graphs){pfx_label}")
        print(f"{'='*65}")
        llm_eager = LLM(
            model=args.model,
            dtype="float16",
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            enforce_eager=True,
            enable_prefix_caching=not args.no_prefix_cache,
        )
        run_suite(llm_eager, args.prompt, sampling_params, args.reps, "eager",
                  warmup_reps=args.warmup_reps,
                  json_path=json_path,
                  variant_name="eager",
                  pp_label=args.pp_label,
                  tg_label=args.max_new_tokens)

        del llm_eager
        gc.collect()
        torch.cuda.empty_cache()

    # -- CUDA graph mode -------------------------------------------------------
    if args.mode in ("all", "cuda-graph"):
        print(f"\n{'='*65}")
        pfx_label = " (prefix-cache OFF)" if args.no_prefix_cache else ""
        print(f" vLLM benchmark: enforce_eager=False (CUDA graphs){pfx_label}")
        print(f"{'='*65}")
        llm_graph = LLM(
            model=args.model,
            dtype="float16",
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            enforce_eager=False,
            enable_prefix_caching=not args.no_prefix_cache,
        )
        run_suite(llm_graph, args.prompt, sampling_params, args.reps, "cuda-graph",
                  warmup_reps=args.warmup_reps,
                  json_path=json_path,
                  variant_name="cuda-graph",
                  pp_label=args.pp_label,
                  tg_label=args.max_new_tokens)

        del llm_graph
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*65}")
    print(" Done.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
