#!/usr/bin/env python3
"""
Generate prompt text files at exact Qwen3 token counts for the pp sweep.

Why prompt *files* instead of inline strings: shell escaping for long prompts
is brittle; each bench script reads the file verbatim.

Usage:
    python make_prompts.py --model ../hf_models/qwen3_4b --out-dir prompts
        [--pp 18 128 512 2048]
        [--chat-template-pp 18]

For `--chat-template-pp` the prompt is the canonical Qwen3 "Hello, how are
you?" chat-templated string (18 tokens on Qwen3). All other pp values use
a raw ("the quick brown fox ..." style) prompt truncated to exactly pp
tokens after tokenization. The benchmark README documents that policy.

Output: one file per pp value, <out-dir>/pp_<N>.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path


# Canonical templated prompt used at pp=18 across all bench scripts.
CHAT_TEMPLATED_18 = (
    "<|im_start|>user\nHello, how are you?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# Seed corpus for synthetic raw prompts. Repeat once to span the largest pp.
# Lorem-ipsum-adjacent English so the tokenizer produces stable counts; we
# slice-and-redecode below to hit exact token targets.
SEED = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "liquor jugs. How vexingly quick daft zebras jump! Sphinx of black quartz, "
    "judge my vow. Amazingly few discotheques provide jukeboxes. The five boxing "
    "wizards jump quickly. Jinxed wizards pluck ivy from the big quilt. "
    "We promptly judged antique ivory buckles for the next prize. A large fawn "
    "jumped quickly over white zinc boxes. Crazy Fredrick bought many very "
    "exquisite opal jewels. Heavy boxes perform quick waltzes and jigs. Watch "
    "Jeopardy! Alex Trebek's fun TV quiz game. Two driven jocks help fax my "
    "big quiz. Jackdaws love my big sphinx of quartz. "
)


def make_prompt_of_length(tokenizer, target_tokens: int) -> tuple[str, int]:
    """Return (text, actual_tokens) where text tokenizes to exactly
    `target_tokens` tokens under the given tokenizer."""
    # Repeat the seed enough times to cover the target.
    body = SEED * max(1, (target_tokens // 40) + 2)
    ids = tokenizer(body, add_special_tokens=False)["input_ids"]
    if len(ids) < target_tokens:
        raise RuntimeError(
            f"seed corpus too short: got {len(ids)} tokens, need {target_tokens}"
        )
    truncated = ids[:target_tokens]
    text = tokenizer.decode(truncated, skip_special_tokens=False)
    # Re-encode to verify count (some tokenizers drift ±1 from BOS/EOS handling).
    reencoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text, len(reencoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt file generator for pp sweep")
    parser.add_argument("--model", required=True, help="Path to HF model (for tokenizer)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--pp", type=int, nargs="+", default=[18, 128, 512, 2048],
        help="Prompt lengths in tokens (pp=18 is always the chat-templated prompt).",
    )
    parser.add_argument(
        "--chat-template-pp", type=int, default=18,
        help="pp value that should use the chat-templated canonical prompt rather "
             "than the synthetic raw prompt. Set to 0 to disable.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(
            "transformers not available; run under a bench env, e.g. "
            "../bench_envs/vllm_env/bin/python3"
        )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pp in sorted(set(args.pp)):
        if pp == args.chat_template_pp:
            text = CHAT_TEMPLATED_18
            actual = len(tok(text, add_special_tokens=False)["input_ids"])
            kind = "chat-templated"
        else:
            text, actual = make_prompt_of_length(tok, pp)
            kind = "synthetic raw"
        path = args.out_dir / f"pp_{pp}.txt"
        path.write_text(text)
        mismatch = " (MISMATCH)" if actual != pp else ""
        print(f"  pp={pp:<4}  actual_tokens={actual:<4}  kind={kind:<14}  path={path}{mismatch}")

    print(f"\nWrote {len(args.pp)} prompt files to {args.out_dir}/")


if __name__ == "__main__":
    main()
