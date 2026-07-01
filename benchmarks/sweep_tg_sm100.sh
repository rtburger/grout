#!/usr/bin/env bash
# B200 / sm_100 profile for the canonical tg sweep.
#
# This wrapper intentionally keeps benchmark flow, Python baselines, and
# aggregation in sweep_tg.sh. Only the sm_100-specific roofline and Grout
# tuning knobs live here so they are easy to inspect and revise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${GROUT_TUNING_PROFILE:=sm100_b200}"

# B200 SXM nominal HBM3e bandwidth is 8 TB/s. The aggregate report also
# includes the 85% effective-bandwidth line unless explicitly overridden.
: "${ROOFLINE_GPU_BANDWIDTH_GBPS:=8000}"
: "${ROOFLINE_BANDWIDTH_FRACTION:=0.85}"

# Keep TRT-LLM out of these local paper sweeps unless explicitly requested.
: "${SWEEP_ENABLE_TRTLLM:=0}"

# Long-context tg cells dominate runtime; keep short cells at BENCH_REPS and
# run tg > 512 with fewer measured reps unless explicitly overridden.
: "${BENCH_REPS_LONG:=3}"

# B200 cuBLAS retune lives in Rust as a compute-capability default so 4B and
# 32B can pick shape-specific modes. Set GROUT_CUBLAS_COMPUTE16 externally to
# force a diagnostic override.

# Current sm_100 Grout tile profile. Attention tile values match the
# 5090-tuned defaults until the B200 TileGym retune produces better cells.
: "${GROUT_ATTN_BM_PREFILL:=16}"
: "${GROUT_ATTN_BN_PREFILL:=32}"
: "${GROUT_ATTN_BN_DECODE_TG_36:=16}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_36:=4}"
: "${GROUT_ATTN_BN_DECODE_TG_128:=32}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_128:=4}"
: "${GROUT_ATTN_BN_DECODE_TG_512:=64}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_512:=8}"
: "${GROUT_ATTN_BN_DECODE_TG_2048:=64}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_2048:=16}"
: "${GROUT_ATTN_BN_DECODE_TG_8192:=64}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_8192:=32}"

: "${GROUT_CUDA_GRAPH_DECODE:=1}"
: "${GROUT_FLASH_DECODE:=0}"
: "${GROUT_FMHA_SPLIT_KV:=1}"
: "${GROUT_FMHA_DECODE_LATENCY:=4}"
: "${GROUT_FMHA_DECODE_OCCUPANCY:=2}"
: "${GROUT_FMHA_MERGE_CHUNK_D:=16}"
: "${GROUT_FMHA_MERGE_LATENCY:=2}"
: "${GROUT_FUSED_QK_ROPE_KV_DECODE:=1}"
: "${GROUT_QK_ROPE_LATENCY:=2}"
: "${GROUT_QK_ROPE_OCCUPANCY:=1}"
: "${GROUT_QK_ROPE_CGA:=0}"
: "${GROUT_KV_CACHE_DYN_CHUNK_D:=32}"
: "${GROUT_KV_CACHE_BM_S:=16}"
: "${GROUT_EMBED_BLOCK:=1024}"
: "${GROUT_RMS_BLOCK:=8192}"
: "${GROUT_ARGMAX_BLOCK:=128}"
: "${GROUT_FMHA_PREFILL:=1}"
: "${GROUT_FMHA_PREFILL_GQA:=0}"
: "${GROUT_FMHA_PREFILL_LATENCY:=2}"
: "${GROUT_FMHA_PREFILL_OCCUPANCY:=2}"
: "${GROUT_FUSED_QK_ROPE_KV_PREFILL:=1}"

export \
    GROUT_TUNING_PROFILE \
    ROOFLINE_GPU_BANDWIDTH_GBPS \
    ROOFLINE_BANDWIDTH_FRACTION \
    SWEEP_ENABLE_TRTLLM \
    BENCH_REPS_LONG \
    GROUT_ATTN_BM_PREFILL \
    GROUT_ATTN_BN_PREFILL \
    GROUT_ATTN_BN_DECODE_TG_36 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_36 \
    GROUT_ATTN_BN_DECODE_TG_128 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_128 \
    GROUT_ATTN_BN_DECODE_TG_512 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_512 \
    GROUT_ATTN_BN_DECODE_TG_2048 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_2048 \
    GROUT_ATTN_BN_DECODE_TG_8192 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_8192 \
    GROUT_CUDA_GRAPH_DECODE \
    GROUT_FLASH_DECODE \
    GROUT_FMHA_SPLIT_KV \
    GROUT_FMHA_DECODE_LATENCY \
    GROUT_FMHA_DECODE_OCCUPANCY \
    GROUT_FMHA_MERGE_CHUNK_D \
    GROUT_FMHA_MERGE_LATENCY \
    GROUT_FUSED_QK_ROPE_KV_DECODE \
    GROUT_QK_ROPE_LATENCY \
    GROUT_QK_ROPE_OCCUPANCY \
    GROUT_QK_ROPE_CGA \
    GROUT_KV_CACHE_DYN_CHUNK_D \
    GROUT_KV_CACHE_BM_S \
    GROUT_EMBED_BLOCK \
    GROUT_RMS_BLOCK \
    GROUT_ARGMAX_BLOCK \
    GROUT_FMHA_PREFILL \
    GROUT_FMHA_PREFILL_GQA \
    GROUT_FMHA_PREFILL_LATENCY \
    GROUT_FMHA_PREFILL_OCCUPANCY \
    GROUT_FUSED_QK_ROPE_KV_PREFILL

exec bash "$SCRIPT_DIR/sweep_tg.sh" "$@"
