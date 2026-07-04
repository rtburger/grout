use crate::dequant::GgmlType;
use crate::gguf::{Content, TensorInfo};
use anyhow::{Context, Result, ensure};
use cutile::core::f16;

const TRANSFORMER_MATRIX_SUFFIXES: [&str; 7] = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillDequantScratchPlan {
    /// Number of f16 elements in the reusable prefill dequant scratch buffer.
    pub elems: usize,
    /// Buffer size in bytes.
    pub bytes: usize,
    /// Largest transformer matrix that determined the scratch size.
    pub tensor_name: String,
    pub shape: Vec<usize>,
}

/// Plan the single reusable prefill dequant scratch buffer for a GGUF model.
///
/// Only transformer block projection matrices are considered. Token embedding
/// and `output.weight`/LM-head tensors are deliberately excluded: prefill logits
/// are last-token-only through the existing gather-row path, so the LM head stays
/// quantized permanently instead of being expanded into this scratch buffer.
pub fn prefill_dequant_scratch_plan(content: &Content) -> Result<PrefillDequantScratchPlan> {
    let mut best: Option<&TensorInfo> = None;
    let mut best_elems = 0usize;

    for info in content.tensor_infos.values() {
        if !is_transformer_projection(info) {
            continue;
        }
        let elems = info.elem_count()?;
        if elems > best_elems {
            best = Some(info);
            best_elems = elems;
        }
    }

    let info = best.context("GGUF has no supported quantized transformer projection matrices")?;
    let bytes = best_elems
        .checked_mul(std::mem::size_of::<f16>())
        .context("prefill dequant scratch size overflows usize")?;
    Ok(PrefillDequantScratchPlan {
        elems: best_elems,
        bytes,
        tensor_name: info.name.clone(),
        shape: info.shape.clone(),
    })
}

fn is_transformer_projection(info: &TensorInfo) -> bool {
    if !matches!(
        info.dtype,
        GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q5K
    ) {
        return false;
    }
    if info.shape.len() != 2 {
        return false;
    }
    let Some((prefix, suffix)) = info.name.rsplit_once('.') else {
        return false;
    };
    let Some((_blk, layer_and_kind)) = prefix.split_once('.') else {
        return false;
    };
    let Some((_layer, kind_prefix)) = layer_and_kind.split_once('.') else {
        return false;
    };
    let full_suffix = format!("{kind_prefix}.{suffix}");
    info.name.starts_with("blk.") && TRANSFORMER_MATRIX_SUFFIXES.contains(&full_suffix.as_str())
}

/// Round a matrix element count up to a dequant kernel tile count, while
/// ensuring it still fits in the planned pooled scratch buffer.
pub fn dequant_tiles_for_scratch(
    matrix_elems: usize,
    tile_elems: usize,
    scratch_elems: usize,
) -> Result<usize> {
    ensure!(tile_elems > 0, "tile_elems must be non-zero");
    ensure!(
        matrix_elems <= scratch_elems,
        "matrix has {matrix_elems} f16 elements but prefill dequant scratch holds only {scratch_elems}"
    );
    ensure!(
        matrix_elems.is_multiple_of(tile_elems),
        "matrix element count {matrix_elems} is not divisible by dequant tile size {tile_elems}"
    );
    Ok(matrix_elems / tile_elems)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{Content, Version};
    use std::collections::HashMap;

    fn tensor(name: &str, dtype: GgmlType, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            offset: 0,
        }
    }

    #[test]
    fn scratch_plan_excludes_lm_head_and_embeddings() -> Result<()> {
        let tensors = [
            tensor("token_embd.weight", GgmlType::Q6K, &[151_936, 4096]),
            tensor("output.weight", GgmlType::Q6K, &[151_936, 4096]),
            tensor("blk.0.attn_q.weight", GgmlType::Q4K, &[4096, 4096]),
            tensor("blk.0.ffn_gate.weight", GgmlType::Q4K, &[12_288, 4096]),
            tensor("blk.0.ffn_norm.weight", GgmlType::F32, &[4096]),
        ];
        let content = Content {
            version: Version::V3,
            metadata: HashMap::new(),
            tensor_infos: tensors.into_iter().map(|t| (t.name.clone(), t)).collect(),
            tensor_data_offset: 0,
        };

        let plan = prefill_dequant_scratch_plan(&content)?;
        assert_eq!(plan.tensor_name, "blk.0.ffn_gate.weight");
        assert_eq!(plan.elems, 12_288 * 4096);
        assert_eq!(plan.bytes, 12_288 * 4096 * 2);
        Ok(())
    }

    #[test]
    fn dequant_tile_count_checks_scratch_capacity() -> Result<()> {
        assert_eq!(dequant_tiles_for_scratch(256, 32, 512)?, 8);
        assert!(dequant_tiles_for_scratch(544, 32, 512).is_err());
        assert!(dequant_tiles_for_scratch(33, 32, 512).is_err());
        Ok(())
    }
}
