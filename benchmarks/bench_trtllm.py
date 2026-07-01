#!/usr/bin/env python3
"""
Benchmark Qwen3-4B FP16 single-request decode performance with TensorRT-LLM.

Uses TRT-LLM's Python API to build an engine and measure single-request
latency, matching the methodology used in bench_vllm.py / bench_sglang.py.

Prerequisites:
    - tensorrt_llm installed (pip install tensorrt-llm, or use NGC container)
    - HF model weights at --model path
    - Sufficient GPU memory for FP16 engine build + runtime

Usage:
    python bench_trtllm.py --model ../hf_models/qwen3_4b
    python bench_trtllm.py --model ../hf_models/qwen3_4b --prompt "Hello, how are you?" --max-new-tokens 36 --reps 5
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

    @property
    def prompt_tps(self) -> float:
        return self.prompt_tokens / self.elapsed

    @property
    def request_gen_tps(self) -> float:
        return self.gen_tokens / self.elapsed

    @property
    def total_tps(self) -> float:
        return (self.prompt_tokens + self.gen_tokens) / self.elapsed


def run_suite(
    generate_fn,
    reps: int,
    label: str,
    warmup_reps: int = 1,
    json_path: Optional[Path] = None,
    variant_name: Optional[str] = None,
    pp_label: int = 0,
    tg_label: int = 0,
    engine_name: str = "trt-llm",
) -> list[RunResult]:
    # warmup (discarded)
    for _ in range(max(1, warmup_reps)):
        generate_fn()

    results: list[RunResult] = []
    for i in range(reps):
        r = generate_fn()
        results.append(r)
        print(
            f"  [{label}] run {i+1}/{reps}: "
            f"prompt_tokens={r.prompt_tokens}, gen_tokens={r.gen_tokens}, "
            f"elapsed={r.elapsed:.3f}s, "
            f"request_gen_tps={r.request_gen_tps:.1f}, "
            f"total_tps={r.total_tps:.1f}"
        )
        emit_jsonl(json_path, {
            "engine": engine_name,
            "variant": variant_name or label,
            "pp": pp_label,
            "tg": tg_label,
            "rep": i,
            "prompt_tokens": r.prompt_tokens,
            "gen_tokens": r.gen_tokens,
            "e2e_ms": r.elapsed * 1000.0,
        })

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
    parser = argparse.ArgumentParser(description="TRT-LLM single-request FP16 benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to HF model directory")
    # Pre-templated Qwen3 chat prompt (18 tokens) — matches bench_vllm.py /
    # bench_sglang.py / bench_llama_cpp.py defaults for apples-to-apples.
    default_prompt = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    parser.add_argument("--prompt", type=str, default=default_prompt)
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read prompt text from file. Overrides --prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=36)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--engine-dir", type=str, default=None,
                        help="Path to pre-built TRT-LLM engine dir (trt backend only). "
                             "If not provided, builds one on the fly.")
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--backend",
        choices=["trt", "pytorch"],
        default="trt",
        help="trt (default): ahead-of-time compiled TensorRT engine. "
             "pytorch: TRT-LLM's PyTorch-orchestrated backend. Both must "
             "run under trtllm-llmapi-launch (MPI proxy). On Qwen3-4B / "
             "RTX 5090, pytorch was ~3% slower than trt.",
    )
    parser.add_argument("--warmup-reps", type=int, default=3)
    parser.add_argument("--json", type=str, default=None,
                        help="Append JSONL per measured run to this file.")
    parser.add_argument("--pp-label", type=int, default=0,
                        help="Label for `pp` field in JSON (informational).")
    args = parser.parse_args()
    json_path = Path(args.json) if args.json else None
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text()

    model_dir = Path(args.model)

    # Late imports so the script gives a clean error if tensorrt_llm is absent.
    try:
        import tensorrt_llm
        print(f"tensorrt_llm version: {tensorrt_llm.__version__}")
        from tensorrt_llm import SamplingParams
        if args.backend == "trt":
            # TRT-engine backend: lives under _tensorrt_engine in 1.2+.
            from tensorrt_llm._tensorrt_engine import LLM
            from tensorrt_llm import BuildConfig
        else:
            # PyTorch backend: top-level LLM, no BuildConfig.
            from tensorrt_llm import LLM
            BuildConfig = None
    except ImportError as e:
        print(f"ERROR: Could not import tensorrt_llm: {e}")
        print("Install with:  pip install tensorrt-llm")
        print("Or use the NVIDIA NGC TRT-LLM container.")
        raise SystemExit(1)

    # -- Build or load engine --------------------------------------------------
    engine_dir = args.engine_dir
    if engine_dir is not None:
        engine_dir = Path(engine_dir)

    print(f"\n{'='*65}")
    print(f" TRT-LLM benchmark: Qwen3-4B FP16  (backend={args.backend})")
    print(f"{'='*65}")

    if args.backend == "trt":
        build_config = BuildConfig(
            max_input_len=args.max_input_len,
            max_seq_len=args.max_seq_len,
            max_batch_size=1,
        )
        # Engine-dir caching (for sweep amortization): load if the dir
        # exists and has any contents; else build + save. An empty
        # freshly-mkdir'd directory is treated as "not yet built".
        have_cached_engine = (
            engine_dir is not None
            and engine_dir.exists()
            and any(engine_dir.iterdir())
        )
        if have_cached_engine:
            print(f"Loading pre-built engine from {engine_dir} ...")
            llm = LLM(model=str(engine_dir))
        else:
            print(f"Building TRT-LLM engine from {model_dir} (dtype=float16) ...")
            llm = LLM(
                model=str(model_dir),
                dtype="float16",
                build_config=build_config,
            )
            if engine_dir is not None:
                print(f"Saving engine to {engine_dir} ...")
                engine_dir.mkdir(parents=True, exist_ok=True)
                llm.save(str(engine_dir))
    else:
        # PyTorch backend: no engine build, weights loaded directly.
        print(f"Loading PyTorch-backend LLM from {model_dir} (dtype=float16) ...")
        llm = LLM(
            model=str(model_dir),
            dtype="float16",
            tensor_parallel_size=1,
        )

    # Force exactly `max_new_tokens` generated, ignoring EOS — matches
    # bench_vllm.py (min_tokens==max_tokens), bench_sglang.py (min_new==max_new),
    # and bench_llama_cpp.py (ignore_eos=True) so the decode window is
    # identical across engines.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
    )

    prompt = args.prompt

    # -- Benchmark function ----------------------------------------------------
    def generate_one() -> RunResult:
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        elapsed = time.perf_counter() - start
        out = outputs[0]
        prompt_tokens = len(out.prompt_token_ids)
        gen_tokens = len(out.outputs[0].token_ids)
        return RunResult(
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            elapsed=elapsed,
        )

    run_suite(generate_one, args.reps, "trt-llm",
              warmup_reps=args.warmup_reps,
              json_path=json_path,
              variant_name=args.backend,
              pp_label=args.pp_label,
              tg_label=args.max_new_tokens,
              engine_name="trt-llm")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*65}")
    print(" Done.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
