#!/usr/bin/env python3
"""PyTorch transformers PREFILL baseline, comparable to grout's prefill metric.

For cutile-rs issue #171 we only care about prefill (grout's slow path), so this
times a SINGLE batched forward over the whole prompt -- exactly what grout does
in step_seq for the prompt pass.

Methodology mirrors the issue reporter's bench_transformers.py
(github.com/kitty-eu-org/grout, the source of the issue's bench.md), adapted to
this repo's bench conventions:
  * dtype        : float16 (grout stores f16; bf16 weights are cast to f16)
  * attention    : scaled_dot_product_attention (eager SDPA, no torch.compile)
  * prefill      : single batched forward over the whole prompt (no chunking)
  * timing       : CUDA-event timed with torch.cuda.synchronize; warmup discarded
  * prompt       : read verbatim from --prompt-file (the same files make_prompts.py
                   feeds grout/vLLM), so prompt_tokens match byte-for-byte
  * JSONL schema : engine="transformers", same fields as bench_vllm.py, so
                   aggregate_sweep.py picks it up next to the grout sweep

Usage:
    python bench_transformers.py --model ../hf_models/qwen3_4b \
        --prompt-file prompts/pp_512.txt --pp-label 512 \
        --reps 5 --warmup-reps 3 --json run.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def emit_jsonl(path: Optional[Path], record: dict) -> None:
    if path is None:
        return
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def build_prompt_text(prompt: str, raw: bool) -> str:
    """Reproduce grout's maybe_apply_chat_template literal form so prompt token
    counts match. With --raw-prompt (what the pp sweep uses, since make_prompts.py
    already emits exact-length text) the prompt is used verbatim."""
    if raw or "<|im_start|>" in prompt:
        return prompt
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="transformers PREFILL baseline vs grout")
    p.add_argument("--model", required=True, help="path to HF model dir")
    p.add_argument("--prompt", type=str,
                   default="<|im_start|>user\nHello, how are you?<|im_end|>\n"
                           "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    p.add_argument("--prompt-file", type=str, default=None,
                   help="read prompt text from file (overrides --prompt)")
    p.add_argument("--reps", type=int, default=5, help="measured forward passes")
    p.add_argument("--warmup-reps", type=int, default=3, help="untimed warmup passes")
    p.add_argument("--dtype", choices=["half", "float16", "bfloat16"], default="half")
    p.add_argument("--attn", choices=["sdpa", "eager", "flash_attention_2"],
                   default="sdpa", help="attn_implementation (default sdpa, no compile)")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="cap context (defaults to model max_position_embeddings)")
    p.add_argument("--raw-prompt", action="store_true",
                   help="use prompt verbatim (matches grout --raw-prompt / the pp sweep)")
    p.add_argument("--json", type=str, default=None,
                   help="append one JSONL record per measured rep")
    p.add_argument("--pp-label", type=int, default=0, help="value for the `pp` JSON field")
    p.add_argument("--compile", action="store_true",
                   help="wrap the model in torch.compile (the standard PyTorch speedup). "
                        "Each prompt length triggers one compile, absorbed by warmup.")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune",
                            "max-autotune-no-cudagraphs"],
                   help="torch.compile mode (default is the safe/fast-to-compile option; "
                        "max-autotune chases peak kernels but compiles much slower)")
    p.add_argument("--variant", type=str, default=None,
                   help="JSON variant label (defaults to eager-sdpa / compiled-<mode>)")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"half": torch.float16, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    json_path = Path(args.json) if args.json else None

    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text()
    prompt_text = build_prompt_text(args.prompt, args.raw_prompt)

    print(f"Loading {args.model} (dtype={dtype}, attn={args.attn}) ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation=args.attn,
    ).to(device).eval()

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    variant = args.variant or (f"compiled-{args.compile_mode}" if args.compile else "eager-sdpa")
    print(f"variant={variant}")

    enc = tok(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    seq_len = int(input_ids.shape[1])
    print(f"prompt_tokens={seq_len}  reps={args.reps} warmup={args.warmup_reps}")

    def one_forward() -> float:
        """Run one prefill forward; return GPU ms (CUDA events) or wall ms (CPU)."""
        if device == "cuda":
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            t1.record()
            torch.cuda.synchronize()
            return t0.elapsed_time(t1)
        from time import perf_counter
        a = perf_counter()
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        return (perf_counter() - a) * 1000.0

    for _ in range(max(0, args.warmup_reps)):
        one_forward()

    ms_list = []
    for i in range(args.reps):
        ms = one_forward()
        ms_list.append(ms)
        tps = seq_len / (ms / 1000.0) if ms > 0 else 0.0
        print(f"  run {i+1}/{args.reps}: prefill {ms:.3f} ms  ({tps:.1f} prompt tok/s)")
        emit_jsonl(json_path, {
            "engine": "transformers",
            "variant": variant,
            "pp": args.pp_label,
            "tg": 0,
            "rep": i,
            "prompt_tokens": seq_len,
            "gen_tokens": 0,
            "e2e_ms": ms,
            "prefill_ms": ms,
        })

    ms_sorted = sorted(ms_list)
    median = ms_sorted[len(ms_sorted) // 2]
    mean = sum(ms_list) / len(ms_list)
    print(f"  median {median:.3f} ms ({seq_len / (median/1000.0):.1f} tok/s), "
          f"mean {mean:.3f} ms over {args.reps} reps")


if __name__ == "__main__":
    main()
