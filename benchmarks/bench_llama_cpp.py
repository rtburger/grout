#!/usr/bin/env python3
"""
Benchmark Qwen3-4B FP16 single-request decode performance with llama.cpp.

Drives a local llama-server via its HTTP /completion endpoint so the
methodology matches bench_vllm.py and bench_sglang.py: same pre-templated
Qwen3 chat prompt (18 tokens), same generation length (36 tokens, forced
via ignore_eos), 5 reps + 1 warmup, wall-clock elapsed per request.

Usage:
    python bench_llama_cpp.py \\
        --llama-server ../llama.cpp/build/bin/llama-server \\
        --gguf ../hf_models/qwen3_4b_f16.gguf
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    prefill_ms: Optional[float]  # from llama-server's timings.prompt_ms
    decode_ms: Optional[float]   # from llama-server's timings.predicted_ms

    @property
    def prompt_tps(self) -> float:
        return self.prompt_tokens / self.elapsed

    @property
    def request_gen_tps(self) -> float:
        return self.gen_tokens / self.elapsed

    @property
    def decode_phase_tps(self) -> Optional[float]:
        if self.decode_ms is None or self.decode_ms <= 0:
            return None
        return self.gen_tokens / (self.decode_ms / 1000.0)

    @property
    def total_tps(self) -> float:
        return (self.prompt_tokens + self.gen_tokens) / self.elapsed


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(proc: subprocess.Popen, base_url: str, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    health = f"{base_url}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early with code {proc.returncode}"
            )
        try:
            with urllib.request.urlopen(health, timeout=2.0) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read() or b"{}")
                    if body.get("status") == "ok":
                        return
        except (urllib.error.URLError, urllib.error.HTTPError,
                ConnectionError, socket.timeout):
            pass
        time.sleep(0.5)
    raise RuntimeError(
        f"llama-server did not become ready at {health} within {timeout_s}s"
    )


def bench_one(completion_url: str, prompt: str, max_new_tokens: int) -> RunResult:
    # ignore_eos=True (both per-request and via --ignore-eos at server
    # startup) was insufficient for Qwen3 on llama.cpp build e6ec21e:
    # the server still stopped at STOP_TYPE_EOS despite the logit_bias_eog
    # path supposedly pushing those tokens to -INFINITY. Workaround is an
    # explicit per-request `logit_bias` that forbids the Qwen3 end-of-turn
    # tokens (151643=<|endoftext|>, 151645=<|im_end|>) — llama-server's
    # logit_bias JSON format is [token_id, allow: bool|number], and
    # `allow: false` maps to -INFINITY cleanly.
    qwen3_eog_bias = [[151643, False], [151645, False]]
    body = json.dumps({
        "prompt": prompt,
        "n_predict": max_new_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "ignore_eos": True,
        "stop": [],
        "logit_bias": qwen3_eog_bias,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        completion_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - start

    # llama-server's JSON response includes a `timings` dict with
    # `prompt_ms` (prefill) and `predicted_ms` (decode). Keep them separate
    # from request-level wall-clock elapsed so aggregation does not mix
    # measurement boundaries.
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    timings = data.get("timings")
    if isinstance(timings, dict):
        val = timings.get("prompt_ms")
        if isinstance(val, (int, float)):
            prefill_ms = float(val)
        val = timings.get("predicted_ms")
        if isinstance(val, (int, float)):
            decode_ms = float(val)

    return RunResult(
        prompt_tokens=int(data.get("tokens_evaluated", 0)),
        gen_tokens=int(data.get("tokens_predicted", 0)),
        elapsed=elapsed,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
    )


def run_suite(
    completion_url: str,
    prompt: str,
    max_new_tokens: int,
    reps: int,
    warmup_reps: int,
    label: str,
    json_path: Optional[Path] = None,
    variant_name: Optional[str] = None,
    pp_label: int = 0,
) -> list[RunResult]:
    # Two warmups: first primes the slot / KV cache, second lets llama-server
    # finish its lazy CUDA-graph capture. One warmup leaves the first measured
    # run cold (~50% slower).
    for _ in range(warmup_reps):
        bench_one(completion_url, prompt, max_new_tokens)

    results: list[RunResult] = []
    for i in range(reps):
        r = bench_one(completion_url, prompt, max_new_tokens)
        results.append(r)
        prefill_str = (
            f", prefill_ms={r.prefill_ms:.2f}" if r.prefill_ms is not None else ""
        )
        decode_phase_str = (
            f", decode_ms={r.decode_ms:.2f}, decode_phase_tps={r.decode_phase_tps:.1f}"
            if r.decode_ms is not None and r.decode_phase_tps is not None else ""
        )
        print(
            f"  [{label}] run {i+1}/{reps}: "
            f"prompt_tokens={r.prompt_tokens}, gen_tokens={r.gen_tokens}, "
            f"elapsed={r.elapsed:.3f}s{prefill_str}{decode_phase_str}, "
            f"request_gen_tps={r.request_gen_tps:.1f}, "
            f"total_tps={r.total_tps:.1f}"
        )
        record = {
            "engine": "llama.cpp",
            "variant": variant_name or label,
            "pp": pp_label,
            "tg": max_new_tokens,
            "rep": i,
            "prompt_tokens": r.prompt_tokens,
            "gen_tokens": r.gen_tokens,
            "e2e_ms": r.elapsed * 1000.0,
        }
        if r.prefill_ms is not None:
            record["prefill_ms"] = r.prefill_ms
        if r.decode_ms is not None:
            record["decode_ms"] = r.decode_ms
        emit_jsonl(json_path, record)

    mean_request_gen = sum(r.request_gen_tps for r in results) / len(results)
    mean_total = sum(r.total_tps for r in results) / len(results)
    mean_elapsed = sum(r.elapsed for r in results) / len(results)
    decode_phase_vals = [r.decode_phase_tps for r in results if r.decode_phase_tps is not None]
    decode_phase_str = (
        f", decode_phase_tps={sum(decode_phase_vals) / len(decode_phase_vals):.1f}"
        if decode_phase_vals else ""
    )
    print(
        f"  [{label}] mean over {reps} runs: "
        f"request_gen_tps={mean_request_gen:.1f}{decode_phase_str}, "
        f"total_tps={mean_total:.1f}, elapsed={mean_elapsed:.3f}s"
    )
    return results


def _start_server(
    llama_server: str,
    gguf: str,
    port: int,
    flash_attn: bool,
    ngl: int,
    ctx_size: int,
) -> subprocess.Popen:
    cmd = [
        llama_server,
        "--model", gguf,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", str(ngl),
        "-c", str(ctx_size),
        "-fa", "1" if flash_attn else "0",
        "--parallel", "1",
        "--no-warmup",
        # Server-level ignore-eos: per-request `ignore_eos: true` in the
        # POST body is insufficient for Qwen3 under /completion; the server
        # still stops on chat-template markers. This CLI flag forces the
        # sampler to keep going to n_predict regardless.
        "--ignore-eos",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_server(proc: subprocess.Popen, timeout: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


# -- main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="llama.cpp single-request FP16 benchmark"
    )
    parser.add_argument("--llama-server", required=True,
                        help="Path to llama-server binary")
    parser.add_argument("--gguf", required=True,
                        help="Path to .gguf model file")
    # Pre-templated Qwen3 chat prompt (18 tokens) for fair cross-engine comparison.
    default_prompt = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    parser.add_argument("--prompt", type=str, default=default_prompt)
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read prompt text from file. Overrides --prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=36)
    parser.add_argument("--reps", type=int, default=5,
                        help="Measured reps (averaged); matches bench_vllm.py / bench_sglang.py")
    parser.add_argument("--warmup-reps", type=int, default=2,
                        help="Warmup reps run before measurement (discarded). "
                             "llama-server captures CUDA graphs lazily, so 2 is safer than 1.")
    parser.add_argument("--ngl", type=int, default=99,
                        help="Number of layers to offload to GPU")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--json", type=str, default=None,
                        help="Append JSONL per measured run to this file.")
    parser.add_argument("--pp-label", type=int, default=0,
                        help="Label for `pp` field in JSON (informational).")
    parser.add_argument("--mode", choices=["all", "no-fa", "fa"], default="all",
                        help="Which variant(s) to run.")
    args = parser.parse_args()
    json_path = Path(args.json) if args.json else None
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text()

    if not Path(args.llama_server).is_file():
        print(f"ERROR: llama-server not found at {args.llama_server}",
              file=sys.stderr)
        sys.exit(1)
    if not Path(args.gguf).is_file():
        print(f"ERROR: gguf not found at {args.gguf}", file=sys.stderr)
        sys.exit(1)

    variants = []
    if args.mode in ("all", "no-fa"):
        variants.append(("no-flash-attn", False))
    if args.mode in ("all", "fa"):
        variants.append(("flash-attn", True))
    for label, fa in variants:
        print(f"\n{'='*65}")
        print(f" llama.cpp benchmark: {label}")
        print(f"{'='*65}")
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        proc = _start_server(
            llama_server=args.llama_server,
            gguf=args.gguf,
            port=port,
            flash_attn=fa,
            ngl=args.ngl,
            ctx_size=args.ctx_size,
        )
        try:
            _wait_ready(proc, base_url)
            run_suite(
                f"{base_url}/completion",
                args.prompt,
                args.max_new_tokens,
                args.reps,
                args.warmup_reps,
                f"llama.cpp-{label}",
                json_path=json_path,
                variant_name=label,
                pp_label=args.pp_label,
            )
        finally:
            _stop_server(proc)

    print(f"\n{'='*65}")
    print(" Done.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
