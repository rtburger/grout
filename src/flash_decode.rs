/*
 * Flash Decode (Split-K Grouped Query Attention) kernel.
 *
 * One of a two-kernel op:
 *   1. attention_decode_kernel_grouped (this file) — computes partial attention
 *      for one split tile, writing Att_Out and LSE_Out.
 *   2. splitk_reduce (separate kernel, already exists) — reduces partial results
 *      into final attention output.
 *
 * Inputs:
 *   Q:       [B, H_kv, NUM_Q_HEAD_PER_KV, HEAD_DIM]  E  (grouped view)
 *   K:       [B, H_kv, S_kv, HEAD_DIM]                E
 *   V:       [B, H_kv, S_kv, HEAD_DIM]                E
 *   Att_Out: [B, H_kv, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS, HEAD_DIM]  E  (5D)
 *   LSE_Out: [B, H_kv, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS]            f32 (4D)
 *
 * Grid: (batch_size, num_kv_heads, NUM_KV_SPLITS)
 *   batch_id = ct.bid(0), head_id = ct.bid(1), tile_id = ct.bid(2)
 *
 * Algorithm: Online softmax in log2 space with split-K parallelization.
 *   Each CTA handles one (batch, kv_head, split) tile.
 *   Loop over KV tiles in the assigned split:
 *     1. Load K tile, QK = mma(K, Q_T) -> [TILE_N, QUERY_GROUP_TILE_SIZE]
 *     2. Apply boundary mask if needed
 *     3. Online softmax: m_ij = max(m_i, max(qk_scaled)), p = exp2(qk - m_ij)
 *     4. Load V tile transposed, acc += mma(V_T, p_f16)
 *   Post-loop: normalize acc by l_i, compute LSE, scatter-store both outputs.
 *
 * Const generics (from ct.Constant params — Rule 23):
 *   HEAD_DIM, TILE_N, KV_LEN_PER_SPLIT, NUM_Q_HEAD_PER_KV,
 *   QUERY_GROUP_TILE_SIZE, NUM_KV_SPLITS
 *
 * Reference IR: cutile-kernels/flash_decode_kernel/reference/reference.mlir
 *
 * Single source of truth -- included via include!() in:
 *   - cutile/tests/flash_decode_pipeline.rs (compile test)
 *   - cutile-ffi/src/lib.rs (FFI entry)
 */

#[cutile::module]
mod flash_decode_module {
    use cutile::core::*;

    #[cutile::entry(unchecked_accesses = true,
        optimization_hints = (
            sm_120 = (max_divisibility = 16,),
        )
    )]
    unsafe fn attention_decode_kernel_grouped<
        E: ElementType,
        const HEAD_DIM: i32,
        const TILE_N: i32,
        const KV_LEN_PER_SPLIT: i32,
        const NUM_Q_HEAD_PER_KV: i32,
        const QUERY_GROUP_TILE_SIZE: i32,
        const NUM_KV_SPLITS: i32,
    >(
        // Q: 4D [B, H_kv, NUM_Q_HEAD_PER_KV, HEAD_DIM] E (grouped view)
        q_ptr: *mut E,
        q_s0: i32,
        q_s1: i32,
        q_s2: i32,
        q_s3: i32,
        q_str0: i32,
        q_str1: i32,
        q_str2: i32,
        // K: 4D [B, H_kv, S_kv, HEAD_DIM] E
        k_ptr: *mut E,
        k_s0: i32,
        k_s1: i32,
        k_s2: i32,
        k_s3: i32,
        k_str0: i32,
        k_str1: i32,
        k_str2: i32,
        // V: 4D [B, H_kv, S_kv, HEAD_DIM] E
        v_ptr: *mut E,
        v_s0: i32,
        v_s1: i32,
        v_s2: i32,
        v_s3: i32,
        v_str0: i32,
        v_str1: i32,
        v_str2: i32,
        // Att_Out: 5D [B, H_kv, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS, HEAD_DIM] E
        att_ptr: *mut E,
        att_s0: i32,
        att_s1: i32,
        att_s2: i32,
        att_s3: i32,
        att_s4: i32,
        att_str0: i32,
        att_str1: i32,
        att_str2: i32,
        att_str3: i32,
        // LSE_Out: 4D [B, H_kv, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS] f32
        lse_ptr: *mut f32,
        lse_s0: i32,
        lse_s1: i32,
        lse_s2: i32,
        lse_s3: i32,
        lse_str0: i32,
        lse_str1: i32,
        lse_str2: i32,
        // Runtime scalars
        softmax_scale: f32,
        // s_kv read from device tensor for CUDA graph compatibility
        s_kv_ptr: *mut i32,
    ) {
        // Read s_kv from device memory (updated via H2D copy before graph replay)
        let s_kv_ptr_a: *mut i32 = unsafe { assume_div_by::<_, 4>(s_kv_ptr) };
        let s_kv_ptile: PointerTile<*mut i32, { [] }> = pointer_to_tile(s_kv_ptr_a);
        let s_kv_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[1i32] };
        let s_kv_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let s_kv_tok: Token = new_token_unordered();
        let s_kv_tv: Tensor<i32, { [-1] }> =
            unsafe { make_tensor_view(s_kv_ptile, s_kv_shape, s_kv_strides, s_kv_tok) };
        let s_kv_part: Partition<i32, { [1] }> =
            s_kv_tv.partition_permuted(const_shape![1], const_array![0]);
        let s_kv_tile: Tile<i32, { [1] }> = load_view_tko(
            &s_kv_part,
            [0i32],
            ordering::Weak,
            scope::TileBlock,
            None,
            tma::Enabled,
        );
        let s_kv: i32 = tile_to_scalar(s_kv_tile.reshape(const_shape![]));
        // ---- Assumes for Q (4D) ----
        let q_ptr_a: *mut E = unsafe { assume_div_by::<_, 16>(q_ptr) };
        let q_s0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(q_s0) };
        let q_s1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(q_s1) };
        let q_s2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(q_s2) };
        let q_s3_b: i32 = unsafe { assume_bounds_lower::<_, 0>(q_s3) };
        let q_s3_a: i32 = unsafe { assume_div_by::<_, 16>(q_s3_b) };
        let q_str0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(q_str0) };
        let q_str0_a: i32 = unsafe { assume_div_by::<_, 8>(q_str0_b) };
        let q_str1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(q_str1) };
        let q_str1_a: i32 = unsafe { assume_div_by::<_, 8>(q_str1_b) };
        let q_str2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(q_str2) };
        let q_str2_a: i32 = unsafe { assume_div_by::<_, 8>(q_str2_b) };

        // ---- Assumes for K (4D) ----
        let k_ptr_a: *mut E = unsafe { assume_div_by::<_, 16>(k_ptr) };
        let k_s0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(k_s0) };
        let k_s1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(k_s1) };
        let k_s2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(k_s2) };
        let k_s2_a: i32 = unsafe { assume_div_by::<_, 16>(k_s2_b) };
        let k_s3_b: i32 = unsafe { assume_bounds_lower::<_, 0>(k_s3) };
        let k_s3_a: i32 = unsafe { assume_div_by::<_, 16>(k_s3_b) };
        let k_str0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(k_str0) };
        let k_str0_a: i32 = unsafe { assume_div_by::<_, 8>(k_str0_b) };
        let k_str1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(k_str1) };
        let k_str1_a: i32 = unsafe { assume_div_by::<_, 8>(k_str1_b) };
        let k_str2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(k_str2) };
        let k_str2_a: i32 = unsafe { assume_div_by::<_, 8>(k_str2_b) };

        // ---- Assumes for V (4D) ----
        let v_ptr_a: *mut E = unsafe { assume_div_by::<_, 16>(v_ptr) };
        let v_s0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(v_s0) };
        let v_s1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(v_s1) };
        let v_s2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(v_s2) };
        let v_s2_a: i32 = unsafe { assume_div_by::<_, 16>(v_s2_b) };
        let v_s3_b: i32 = unsafe { assume_bounds_lower::<_, 0>(v_s3) };
        let v_s3_a: i32 = unsafe { assume_div_by::<_, 16>(v_s3_b) };
        let v_str0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(v_str0) };
        let v_str0_a: i32 = unsafe { assume_div_by::<_, 8>(v_str0_b) };
        let v_str1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(v_str1) };
        let v_str1_a: i32 = unsafe { assume_div_by::<_, 8>(v_str1_b) };
        let v_str2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(v_str2) };
        let v_str2_a: i32 = unsafe { assume_div_by::<_, 8>(v_str2_b) };

        // ---- Assumes for Att_Out (5D) ----
        let att_ptr_a: *mut E = unsafe { assume_div_by::<_, 16>(att_ptr) };
        let att_s0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(att_s0) };
        let att_s1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(att_s1) };
        let att_s2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(att_s2) };
        let att_s3_b: i32 = unsafe { assume_bounds_lower::<_, 0>(att_s3) };
        let att_s4_b: i32 = unsafe { assume_bounds_lower::<_, 0>(att_s4) };
        let att_s4_a: i32 = unsafe { assume_div_by::<_, 16>(att_s4_b) };
        let att_str0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(att_str0) };
        let att_str0_a: i32 = unsafe { assume_div_by::<_, 8>(att_str0_b) };
        let att_str1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(att_str1) };
        let att_str1_a: i32 = unsafe { assume_div_by::<_, 8>(att_str1_b) };
        let att_str2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(att_str2) };
        let att_str2_a: i32 = unsafe { assume_div_by::<_, 8>(att_str2_b) };
        let att_str3_b: i32 = unsafe { assume_bounds_lower::<_, 0>(att_str3) };
        let att_str3_a: i32 = unsafe { assume_div_by::<_, 8>(att_str3_b) };

        // ---- Assumes for LSE_Out (4D f32) ----
        let lse_ptr_a: *mut f32 = unsafe { assume_div_by::<_, 16>(lse_ptr) };
        let lse_s0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(lse_s0) };
        let lse_s1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(lse_s1) };
        let lse_s2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(lse_s2) };
        let lse_s3_b: i32 = unsafe { assume_bounds_lower::<_, 0>(lse_s3) };
        let lse_str0_b: i32 = unsafe { assume_bounds_lower::<_, 0>(lse_str0) };
        let lse_str0_a: i32 = unsafe { assume_div_by::<_, 4>(lse_str0_b) };
        let lse_str1_b: i32 = unsafe { assume_bounds_lower::<_, 0>(lse_str1) };
        let lse_str1_a: i32 = unsafe { assume_div_by::<_, 4>(lse_str1_b) };
        let lse_str2_b: i32 = unsafe { assume_bounds_lower::<_, 0>(lse_str2) };

        // ---- Token ----
        let shared_token: Token = new_token_unordered();

        // ---- Q tensor view: 4D [B, H_kv, NUM_Q_HEAD_PER_KV, HEAD_DIM] ----
        let q_ptr_tile: PointerTile<*mut E, { [] }> = pointer_to_tile(q_ptr_a);
        let q_shape: Shape<{ [-1, -1, -1, -1] }> = Shape::<{ [-1, -1, -1, -1] }> {
            dims: &[q_s0_b, q_s1_b, q_s2_b, q_s3_a],
        };
        let q_strides: Array<{ [-1, -1, -1, 1] }> = Array::<{ [-1, -1, -1, 1] }> {
            dims: &[q_str0_a, q_str1_a, q_str2_a],
        };
        let q_tv: Tensor<E, { [-1, -1, -1, -1] }> =
            unsafe { make_tensor_view(q_ptr_tile, q_shape, q_strides, shared_token) };

        // ---- K tensor view: 4D [B, H_kv, S_kv, HEAD_DIM] ----
        let k_ptr_tile: PointerTile<*mut E, { [] }> = pointer_to_tile(k_ptr_a);
        let k_shape: Shape<{ [-1, -1, -1, -1] }> = Shape::<{ [-1, -1, -1, -1] }> {
            dims: &[k_s0_b, k_s1_b, k_s2_a, k_s3_a],
        };
        let k_strides: Array<{ [-1, -1, -1, 1] }> = Array::<{ [-1, -1, -1, 1] }> {
            dims: &[k_str0_a, k_str1_a, k_str2_a],
        };
        let k_tv: Tensor<E, { [-1, -1, -1, -1] }> =
            unsafe { make_tensor_view(k_ptr_tile, k_shape, k_strides, shared_token) };

        // ---- V tensor view: 4D [B, H_kv, S_kv, HEAD_DIM] ----
        let v_ptr_tile: PointerTile<*mut E, { [] }> = pointer_to_tile(v_ptr_a);
        let v_shape: Shape<{ [-1, -1, -1, -1] }> = Shape::<{ [-1, -1, -1, -1] }> {
            dims: &[v_s0_b, v_s1_b, v_s2_a, v_s3_a],
        };
        let v_strides: Array<{ [-1, -1, -1, 1] }> = Array::<{ [-1, -1, -1, 1] }> {
            dims: &[v_str0_a, v_str1_a, v_str2_a],
        };
        let v_tv: Tensor<E, { [-1, -1, -1, -1] }> =
            unsafe { make_tensor_view(v_ptr_tile, v_shape, v_strides, shared_token) };

        // ---- Att_Out tensor view: 5D [B, H_kv, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS, HEAD_DIM] ----
        let att_ptr_tile: PointerTile<*mut E, { [] }> = pointer_to_tile(att_ptr_a);
        let att_shape: Shape<{ [-1, -1, -1, -1, -1] }> = Shape::<{ [-1, -1, -1, -1, -1] }> {
            dims: &[att_s0_b, att_s1_b, att_s2_b, att_s3_b, att_s4_a],
        };
        let att_strides: Array<{ [-1, -1, -1, -1, 1] }> = Array::<{ [-1, -1, -1, -1, 1] }> {
            dims: &[att_str0_a, att_str1_a, att_str2_a, att_str3_a],
        };
        let att_tv: Tensor<E, { [-1, -1, -1, -1, -1] }> =
            unsafe { make_tensor_view(att_ptr_tile, att_shape, att_strides, shared_token) };

        // ---- LSE_Out tensor view: 4D [B, H_kv, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS] f32 ----
        let lse_ptr_tile: PointerTile<*mut f32, { [] }> = pointer_to_tile(lse_ptr_a);
        let lse_shape: Shape<{ [-1, -1, -1, -1] }> = Shape::<{ [-1, -1, -1, -1] }> {
            dims: &[lse_s0_b, lse_s1_b, lse_s2_b, lse_s3_b],
        };
        let lse_strides: Array<{ [-1, -1, -1, 1] }> = Array::<{ [-1, -1, -1, 1] }> {
            dims: &[lse_str0_a, lse_str1_a, lse_str2_b],
        };
        let lse_tv: Tensor<f32, { [-1, -1, -1, -1] }> =
            unsafe { make_tensor_view(lse_ptr_tile, lse_shape, lse_strides, shared_token) };

        // ---- Block IDs ----
        // Grid: (batch_size, num_kv_heads, NUM_KV_SPLITS)
        let pid: (i32, i32, i32) = get_tile_block_id();
        let batch_id: i32 = pid.0;
        let head_id: i32 = pid.1;
        let tile_id: i32 = pid.2;

        // ---- Pre-scale: qk_scale = softmax_scale * INV_LOG_2 ----
        let inv_log2_s: Tile<f32, { [] }> = scalar_to_tile(1.44269502f32);
        let sm_scale_s: Tile<f32, { [] }> = scalar_to_tile(softmax_scale);
        let qk_scale: Tile<f32, { [] }> = sm_scale_s * inv_log2_s;

        // ---- Load Q tile [1,1,QUERY_GROUP_TILE_SIZE,HEAD_DIM] ----
        let q_part: Partition<E, { [1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM] }> = q_tv
            .partition_permuted(
                const_shape![1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM],
                const_array![0, 1, 2, 3],
            );
        let q_tile_4d: Tile<E, { [1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM] }> = load_view_tko(
            &q_part,
            [batch_id, head_id, 0i32, 0i32],
            ordering::Weak,
            scope::TileBlock,
            None,
            tma::Enabled,
        );
        let q_tile_2d: Tile<E, { [QUERY_GROUP_TILE_SIZE, HEAD_DIM] }> =
            q_tile_4d.reshape(const_shape![QUERY_GROUP_TILE_SIZE, HEAD_DIM]);
        // Transpose Q: [QUERY_GROUP_TILE_SIZE, HEAD_DIM] -> [HEAD_DIM, QUERY_GROUP_TILE_SIZE]
        let q_t: Tile<E, { [HEAD_DIM, QUERY_GROUP_TILE_SIZE] }> =
            permute(q_tile_2d, const_array![1, 0]);

        // ---- Compute start/end KV indices for this split ----
        let start_idx: i32 = tile_id * KV_LEN_PER_SPLIT;
        let end_raw: i32 = start_idx + KV_LEN_PER_SPLIT;
        // min(end_raw, s_kv)
        let end_idx: i32 = if end_raw < s_kv { end_raw } else { s_kv };

        let has_work: bool = end_idx > start_idx;
        let tile_n_val: i32 = TILE_N;

        // num_tiles = cdiv(KV_LEN_PER_SPLIT, TILE_N) -- const generic division
        let num_tiles: i32 = (KV_LEN_PER_SPLIT + TILE_N - 1) / TILE_N;
        let start_tile: i32 = start_idx / TILE_N;

        // ---- Initialize accumulators ----
        let mut m_i: Tile<f32, { [QUERY_GROUP_TILE_SIZE] }> =
            constant(f32::NEG_INFINITY, const_shape![QUERY_GROUP_TILE_SIZE]);
        let mut l_i: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
            constant(1.0f32, const_shape![TILE_N, QUERY_GROUP_TILE_SIZE]);
        let mut acc: Tile<f32, { [HEAD_DIM, QUERY_GROUP_TILE_SIZE] }> =
            constant(0.0f32, const_shape![HEAD_DIM, QUERY_GROUP_TILE_SIZE]);

        // ---- QK zero accumulator ----
        let qk_zero: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
            constant(0.0f32, const_shape![TILE_N, QUERY_GROUP_TILE_SIZE]);

        // ---- Broadcast qk_scale for 2D ops ----
        let qk_scale_1x1: Tile<f32, { [1, 1] }> = qk_scale.reshape(const_shape![1, 1]);
        let qk_scale_bc: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
            qk_scale_1x1.broadcast(const_shape![TILE_N, QUERY_GROUP_TILE_SIZE]);

        // ---- Masking constant ----
        let neg_large: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
            constant(-1000000.0f32, const_shape![TILE_N, QUERY_GROUP_TILE_SIZE]);

        // ---- K partition view ----
        let k_part: Partition<E, { [1, 1, TILE_N, HEAD_DIM] }> = k_tv.partition_permuted(
            const_shape![1, 1, TILE_N, HEAD_DIM],
            const_array![0, 1, 2, 3],
        );

        // ---- V partition view ----
        let v_part: Partition<E, { [1, 1, TILE_N, HEAD_DIM] }> = v_tv.partition_permuted(
            const_shape![1, 1, TILE_N, HEAD_DIM],
            const_array![0, 1, 2, 3],
        );

        // ---- Outer if: only process if this split has work (Rule 22) ----
        if has_work {
            for idx in 0i32..num_tiles {
                let cnt: i32 = start_tile + idx;
                let curr_n: i32 = cnt * TILE_N;

                // Load K tile [1,1,TILE_N,HEAD_DIM]
                let k_tile_4d: Tile<E, { [1, 1, TILE_N, HEAD_DIM] }> = load_view_tko(
                    &k_part,
                    [batch_id, head_id, cnt, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                let k_tile: Tile<E, { [TILE_N, HEAD_DIM] }> =
                    k_tile_4d.reshape(const_shape![TILE_N, HEAD_DIM]);

                // QK = K[TILE_N, HEAD_DIM] @ Q_T[HEAD_DIM, QUERY_GROUP_TILE_SIZE]
                //    -> [TILE_N, QUERY_GROUP_TILE_SIZE]
                let qk: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> = mma(k_tile, q_t, qk_zero);

                // ---- Boundary mask: if curr_n + TILE_N > S_kv ----
                let curr_n_end: i32 = curr_n + tile_n_val;
                let boundary: bool = curr_n_end > s_kv;
                let mut qk_masked: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> = qk;
                if boundary {
                    // Create mask: arange(TILE_N) + curr_n < s_kv
                    let offs_n: Tile<i32, { [TILE_N] }> = iota(const_shape![TILE_N]);
                    let curr_n_bc: Tile<i32, { [TILE_N] }> =
                        broadcast_scalar(curr_n, const_shape![TILE_N]);
                    let abs_n: Tile<i32, { [TILE_N] }> = offs_n + curr_n_bc;
                    let skv_bc: Tile<i32, { [TILE_N] }> =
                        broadcast_scalar(s_kv, const_shape![TILE_N]);
                    let mask_1d: Tile<bool, { [TILE_N] }> = lt_tile(abs_n, skv_bc);
                    // Broadcast mask to 2D [TILE_N, QUERY_GROUP_TILE_SIZE]
                    let mask_col: Tile<bool, { [TILE_N, 1] }> =
                        mask_1d.reshape(const_shape![TILE_N, 1]);
                    let mask_2d: Tile<bool, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
                        mask_col.broadcast(const_shape![TILE_N, QUERY_GROUP_TILE_SIZE]);
                    qk_masked = select(mask_2d, qk, neg_large);
                } else {
                    // Rule 22: explicit else for mutable variable yield
                    qk_masked = qk_masked;
                }

                // ---- Online softmax ----
                // qk_scaled = qk * qk_scale
                let qk_scaled: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
                    qk_masked * qk_scale_bc;

                // row_max = reduce_max(qk_scaled, dim=0) -> [QUERY_GROUP_TILE_SIZE]
                let row_max: Tile<f32, { [QUERY_GROUP_TILE_SIZE] }> = {
                    let r: Tile<f32, { [QUERY_GROUP_TILE_SIZE, 1] }> = reduce_max(qk_scaled, 0);
                    r.reshape(const_shape![QUERY_GROUP_TILE_SIZE])
                };

                // m_ij = max(m_i, row_max)
                let m_ij: Tile<f32, { [QUERY_GROUP_TILE_SIZE] }> =
                    maxf(m_i, row_max, nan::Disabled, ftz::Disabled);

                // p = exp2(qk_scaled - m_ij[None,:])
                let m_ij_1xq: Tile<f32, { [1, QUERY_GROUP_TILE_SIZE] }> =
                    m_ij.reshape(const_shape![1, QUERY_GROUP_TILE_SIZE]);
                let m_ij_bc: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
                    m_ij_1xq.broadcast(const_shape![TILE_N, QUERY_GROUP_TILE_SIZE]);
                let qk_shifted: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
                    qk_scaled - m_ij_bc;
                let p: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
                    exp2(qk_shifted, ftz::Disabled);

                // alpha = exp2(m_i - m_ij, ftz::Disabled)
                let alpha: Tile<f32, { [QUERY_GROUP_TILE_SIZE] }> = exp2(m_i - m_ij, ftz::Disabled);

                // Update l_i: l_i = l_i * alpha[None,:] + p
                let alpha_1xq: Tile<f32, { [1, QUERY_GROUP_TILE_SIZE] }> =
                    alpha.reshape(const_shape![1, QUERY_GROUP_TILE_SIZE]);
                let alpha_bc_n: Tile<f32, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> =
                    alpha_1xq.broadcast(const_shape![TILE_N, QUERY_GROUP_TILE_SIZE]);
                l_i = fma(l_i, alpha_bc_n, p, rounding::NearestEven, ftz::Disabled);

                // Rescale acc: acc = acc * alpha[None,:]
                let alpha_bc_d: Tile<f32, { [HEAD_DIM, QUERY_GROUP_TILE_SIZE] }> =
                    alpha_1xq.broadcast(const_shape![HEAD_DIM, QUERY_GROUP_TILE_SIZE]);
                acc = acc * alpha_bc_d;

                // ---- Load V tile, transpose ----
                let v_tile_4d: Tile<E, { [1, 1, TILE_N, HEAD_DIM] }> = load_view_tko(
                    &v_part,
                    [batch_id, head_id, cnt, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                let v_tile: Tile<E, { [TILE_N, HEAD_DIM] }> =
                    v_tile_4d.reshape(const_shape![TILE_N, HEAD_DIM]);
                let v_t: Tile<E, { [HEAD_DIM, TILE_N] }> = permute(v_tile, const_array![1, 0]);

                // Convert p to E for MMA
                let p_e: Tile<E, { [TILE_N, QUERY_GROUP_TILE_SIZE] }> = convert_tile(p);

                // acc += V_T[HEAD_DIM, TILE_N] @ p_e[TILE_N, QUERY_GROUP_TILE_SIZE]
                acc = mma(v_t, p_e, acc);

                // Update m_i
                m_i = m_ij;
            }
        } else {
            // Rule 22: explicit else for mutable variable yield
            acc = acc;
            l_i = l_i;
            m_i = m_i;
        }

        // ---- Post-loop: normalize ----
        // l = reduce_sum(l_i, dim=0) -> [QUERY_GROUP_TILE_SIZE]
        let l_sum: Tile<f32, { [QUERY_GROUP_TILE_SIZE] }> = {
            let r: Tile<f32, { [QUERY_GROUP_TILE_SIZE, 1] }> = reduce_sum(l_i, 0);
            r.reshape(const_shape![QUERY_GROUP_TILE_SIZE])
        };

        // acc = acc / l_sum[None,:]  (approx, ftz)
        let l_1xq: Tile<f32, { [1, QUERY_GROUP_TILE_SIZE] }> =
            l_sum.reshape(const_shape![1, QUERY_GROUP_TILE_SIZE]);
        let l_bc: Tile<f32, { [HEAD_DIM, QUERY_GROUP_TILE_SIZE] }> =
            l_1xq.broadcast(const_shape![HEAD_DIM, QUERY_GROUP_TILE_SIZE]);
        let acc_norm: Tile<f32, { [HEAD_DIM, QUERY_GROUP_TILE_SIZE] }> = true_div(acc, l_bc);

        // Transpose: [HEAD_DIM, QUERY_GROUP_TILE_SIZE] -> [QUERY_GROUP_TILE_SIZE, HEAD_DIM]
        let acc_perm: Tile<f32, { [QUERY_GROUP_TILE_SIZE, HEAD_DIM] }> =
            permute(acc_norm, const_array![1, 0]);
        // Convert to E
        let acc_e: Tile<E, { [QUERY_GROUP_TILE_SIZE, HEAD_DIM] }> = convert_tile(acc_perm);

        // Compute LSE: l = m_i + log2(l_sum)
        let log2_l: Tile<f32, { [QUERY_GROUP_TILE_SIZE] }> = log2(l_sum);
        let lse_val: Tile<f32, { [QUERY_GROUP_TILE_SIZE] }> = m_i + log2_l;

        // ---- Store Att_Out via scatter (works for both variants) ----
        // Reshape acc to 5D [1,1,QUERY_GROUP_TILE_SIZE,1,HEAD_DIM] and store
        let acc_5d: Tile<E, { [1, 1, QUERY_GROUP_TILE_SIZE, 1, HEAD_DIM] }> =
            acc_e.reshape(const_shape![1, 1, QUERY_GROUP_TILE_SIZE, 1, HEAD_DIM]);
        let mut att_part_mut: PartitionMut<E, { [1, 1, QUERY_GROUP_TILE_SIZE, 1, HEAD_DIM] }> = unsafe {
            att_tv.partition_full_mut(const_shape![1, 1, QUERY_GROUP_TILE_SIZE, 1, HEAD_DIM])
        };
        let _att_tok: Token = unsafe {
            store_view_tko_mut(
                &mut att_part_mut,
                acc_5d,
                [batch_id, head_id, 0i32, tile_id, 0i32],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };

        // ---- Store LSE_Out via scatter ----
        let lse_2d: Tile<f32, { [1, 1, QUERY_GROUP_TILE_SIZE, 1] }> =
            lse_val.reshape(const_shape![1, 1, QUERY_GROUP_TILE_SIZE, 1]);
        let mut lse_part_mut: PartitionMut<f32, { [1, 1, QUERY_GROUP_TILE_SIZE, 1] }> =
            unsafe { lse_tv.partition_full_mut(const_shape![1, 1, QUERY_GROUP_TILE_SIZE, 1]) };
        let _lse_tok: Token = unsafe {
            store_view_tko_mut(
                &mut lse_part_mut,
                lse_2d,
                [batch_id, head_id, 0i32, tile_id],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
    }
}

pub use flash_decode_module::attention_decode_kernel_grouped;
