use crate::config::{Qwen3Config, SafetensorsIndex};
use crate::dequant::{GgmlType, dequantize_to_f16};
use crate::gguf::GgufFile;
use crate::weights::{MatrixWeight, Weight};
use anyhow::{Context, Result, bail, ensure};
use memmap2::MmapOptions;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::core::f16;
use cutile::tensor::{Reshape, Tensor};

#[derive(Debug)]
pub struct HostTensor {
    pub data: Vec<f16>,
    pub shape: Vec<usize>,
}

pub struct WeightLoader {
    model_path: PathBuf,
    weight_map: HashMap<String, String>,
    shards: HashMap<String, memmap2::Mmap>,
    gguf: Option<GgufFile>,
    gguf_config: Option<Qwen3Config>,
}

impl WeightLoader {
    pub fn new(model_path: &Path) -> Result<Self> {
        if is_gguf_path(model_path) {
            let gguf = GgufFile::open(model_path)?;
            let gguf_config = config_from_gguf(&gguf)?;
            return Ok(Self {
                model_path: model_path.to_path_buf(),
                weight_map: HashMap::new(),
                shards: HashMap::new(),
                gguf: Some(gguf),
                gguf_config: Some(gguf_config),
            });
        }

        let index_path = model_path.join("model.safetensors.index.json");
        let index_text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_text)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;

        let mut shard_files = HashSet::new();
        for shard in index.weight_map.values() {
            shard_files.insert(shard.clone());
        }

        let shard_files: Vec<String> = shard_files.into_iter().collect();
        let shard_entries: Vec<(String, memmap2::Mmap)> = shard_files
            .par_iter()
            .map(|shard| -> Result<(String, memmap2::Mmap)> {
                let shard_path = model_path.join(shard);
                let file = std::fs::File::open(&shard_path)
                    .with_context(|| format!("failed to open {}", shard_path.display()))?;
                let mmap = unsafe { MmapOptions::new().map(&file) }
                    .with_context(|| format!("failed to mmap {}", shard_path.display()))?;
                Ok((shard.clone(), mmap))
            })
            .collect::<Result<Vec<_>>>()?;
        let shards = shard_entries.into_iter().collect();

        Ok(Self {
            model_path: model_path.to_path_buf(),
            weight_map: index.weight_map,
            shards,
            gguf: None,
            gguf_config: None,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_path
    }

    pub fn is_gguf(&self) -> bool {
        self.gguf.is_some()
    }

    pub fn gguf_config(&self) -> Option<&Qwen3Config> {
        self.gguf_config.as_ref()
    }

    pub fn load_host_f16(&self, name: &str) -> Result<HostTensor> {
        if let Some(gguf) = &self.gguf {
            return load_host_f16_gguf(gguf, name);
        }

        let shard_name = self
            .weight_map
            .get(name)
            .with_context(|| format!("tensor `{name}` not found in index"))?;
        let mmap = self
            .shards
            .get(shard_name)
            .with_context(|| format!("missing mmap for shard `{shard_name}`"))?;

        let st = SafeTensors::deserialize(&mmap[..])
            .with_context(|| format!("failed to deserialize `{shard_name}`"))?;
        let view = st
            .tensor(name)
            .with_context(|| format!("tensor `{name}` not found in `{shard_name}`"))?;

        let shape = view.shape().to_vec();
        let data = cast_to_f16(view.dtype(), view.data())
            .with_context(|| format!("failed to cast `{name}` from {:?}", view.dtype()))?;
        Ok(HostTensor { data, shape })
    }

    pub fn load_device_f16(
        &self,
        name: &str,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Arc<Tensor<f16>>> {
        let host = self.load_host_f16(name)?;
        copy_f16_to_device(host, stream)
    }

    pub fn load_device_weight(
        &self,
        name: &str,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<MatrixWeight> {
        if let Some(gguf) = &self.gguf {
            return load_device_weight_gguf(gguf, name, stream);
        }
        let tensor = self.load_device_f16(name, stream)?;
        Ok(MatrixWeight::single(Weight::f16(tensor)?))
    }

    /// Estimated resident device bytes for model weights after Grout's loader
    /// policy is applied. Safetensors weights are resident as fp16; GGUF
    /// quantized matrix tensors stay in their native block format while scalar
    /// and norm tensors are resident as fp16.
    pub fn resident_weight_bytes(&self) -> Result<usize> {
        if let Some(gguf) = &self.gguf {
            return resident_weight_bytes_gguf(gguf);
        }
        resident_weight_bytes_safetensors(&self.weight_map, &self.shards)
    }

    /// Pooled quantized-prefill dequant scratch bytes. This intentionally
    /// excludes token embeddings and LM head/output tensors.
    pub fn prefill_dequant_scratch_bytes(&self) -> Result<usize> {
        if let Some(gguf) = &self.gguf {
            return match crate::quant_scratch::prefill_dequant_scratch_plan(&gguf.content) {
                Ok(plan) => Ok(plan.bytes),
                Err(_) => Ok(0),
            };
        }
        Ok(0)
    }
}

fn resident_weight_bytes_gguf(gguf: &GgufFile) -> Result<usize> {
    let tied = !gguf.content.has_tensor("output.weight");
    let mut total = 0usize;
    for info in gguf.content.tensor_infos.values() {
        ensure!(
            info.dtype.is_supported_for_phase1(),
            "unsupported ggml type {} for tensor `{}`",
            info.dtype,
            info.name
        );
        let is_embed = info.name == "token_embd.weight";
        let elems = info.elem_count()?;
        // Mirrors the loader policy: Q8_0 keeps native + SoA; Q4K/Q6K keep
        // SoA only (plus native for the embedding table, which is gathered
        // in native layout and, when tied, also SoA as the LM head).
        let bytes = match info.dtype {
            GgmlType::F16 | GgmlType::F32 => elems
                .checked_mul(std::mem::size_of::<f16>())
                .with_context(|| format!("resident byte size overflows for `{}`", info.name))?,
            GgmlType::Q8_0 => info.size_in_bytes()?.checked_mul(2).with_context(|| {
                format!("Q8_0 resident byte size overflows for `{}`", info.name)
            })?,
            GgmlType::Q6K => {
                let soa = elems + elems / 16 + elems / 256 * 2;
                match (is_embed, tied) {
                    (true, false) => info.size_in_bytes()?,
                    (true, true) => info.size_in_bytes()? + soa,
                    (false, _) => soa,
                }
            }
            GgmlType::Q4K => {
                let soa = elems / 2 + elems / 32 * 4;
                match (is_embed, tied) {
                    (true, false) => info.size_in_bytes()?,
                    (true, true) => info.size_in_bytes()? + soa,
                    (false, _) => soa,
                }
            }
            GgmlType::Q5K => info.size_in_bytes()?,
            other => bail!("unsupported ggml type {other} for tensor `{}`", info.name),
        };
        total = total
            .checked_add(bytes)
            .context("GGUF resident weight byte total overflows usize")?;
    }
    Ok(total)
}

fn resident_weight_bytes_safetensors(
    weight_map: &HashMap<String, String>,
    shards: &HashMap<String, memmap2::Mmap>,
) -> Result<usize> {
    let mut total = 0usize;
    for (name, shard_name) in weight_map {
        let mmap = shards
            .get(shard_name)
            .with_context(|| format!("missing mmap for shard `{shard_name}`"))?;
        let st = SafeTensors::deserialize(&mmap[..])
            .with_context(|| format!("failed to deserialize `{shard_name}`"))?;
        let view = st
            .tensor(name)
            .with_context(|| format!("tensor `{name}` not found in `{shard_name}`"))?;
        ensure!(
            safetensor_dtype_supported_for_f16(view.dtype()),
            "unsupported dtype for fp16 resident weight estimate: {:?} in `{name}`",
            view.dtype()
        );
        let elems = view.shape().iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .with_context(|| format!("tensor `{name}` element count overflows usize"))
        })?;
        let bytes = elems
            .checked_mul(std::mem::size_of::<f16>())
            .with_context(|| format!("tensor `{name}` resident byte size overflows usize"))?;
        total = total
            .checked_add(bytes)
            .context("safetensors resident weight byte total overflows usize")?;
    }
    Ok(total)
}

fn safetensor_dtype_supported_for_f16(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64)
}

fn is_gguf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

fn config_from_gguf(gguf: &GgufFile) -> Result<Qwen3Config> {
    let metadata = &gguf.content;
    let arch = metadata
        .metadata_required("general.architecture")?
        .to_string()?;
    ensure!(
        arch == "qwen3",
        "unsupported GGUF architecture `{arch}`; this engine only supports qwen3"
    );
    // Exact Qwen3 GGUF metadata keys used by candle-transformers' quantized_qwen3 loader.
    let block_count_key = "qwen3.block_count";
    let embedding_length_key = "qwen3.embedding_length";
    let feed_forward_length_key = "qwen3.feed_forward_length";
    let head_count_key = "qwen3.attention.head_count";
    let head_count_kv_key = "qwen3.attention.head_count_kv";
    let key_length_key = "qwen3.attention.key_length";
    let rope_freq_base_key = "qwen3.rope.freq_base";
    let rms_epsilon_key = "qwen3.attention.layer_norm_rms_epsilon";
    let context_length_key = "qwen3.context_length";

    let token_embd = metadata.tensor_info("token_embd.weight")?;
    ensure!(
        token_embd.shape.len() == 2,
        "token_embd.weight must be rank-2, got {:?}",
        token_embd.shape
    );
    let vocab_size = token_embd.shape[0];

    let cfg = Qwen3Config {
        vocab_size,
        hidden_size: metadata
            .metadata_required(&embedding_length_key)?
            .to_u32()? as usize,
        intermediate_size: metadata
            .metadata_required(&feed_forward_length_key)?
            .to_u32()? as usize,
        num_hidden_layers: metadata.metadata_required(&block_count_key)?.to_u32()? as usize,
        num_attention_heads: metadata.metadata_required(&head_count_key)?.to_u32()? as usize,
        num_key_value_heads: metadata.metadata_required(&head_count_kv_key)?.to_u32()? as usize,
        head_dim: metadata.metadata_required(&key_length_key)?.to_u32()? as usize,
        rms_norm_eps: metadata.metadata_required(&rms_epsilon_key)?.to_f32()?,
        rope_theta: metadata.metadata_required(&rope_freq_base_key)?.to_f32()?,
        max_position_embeddings: metadata.metadata_required(&context_length_key)?.to_u32()?
            as usize,
        tie_word_embeddings: !metadata.has_tensor("output.weight"),
        use_sliding_window: false,
        eos_token_id: metadata
            .metadata_required("tokenizer.ggml.eos_token_id")?
            .to_u32()?,
    };
    Ok(cfg)
}

fn copy_f16_to_device(
    host: HostTensor,
    stream: &Arc<cuda_core::Stream>,
) -> Result<Arc<Tensor<f16>>> {
    let shape = host.shape.clone();
    let host_data = Arc::new(host.data);
    let device_tensor = api::copy_host_vec_to_device(&host_data)
        .sync_on(stream)
        .map_err(|e| anyhow::anyhow!("copy to device failed: {e:?}"))?;
    let device_tensor = device_tensor
        .reshape(&shape)
        .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?;
    Ok(Arc::new(device_tensor))
}

fn load_device_weight_gguf(
    gguf: &GgufFile,
    engine_name: &str,
    stream: &Arc<cuda_core::Stream>,
) -> Result<MatrixWeight> {
    let gguf_name = map_engine_tensor_name(engine_name, &gguf.content)
        .with_context(|| format!("failed to map engine tensor `{engine_name}` to GGUF name"))?;
    let (info, data) = gguf.tensor_data(&gguf_name)?;
    ensure!(
        info.dtype.is_supported_for_phase1(),
        "unsupported ggml type {} for tensor `{gguf_name}`",
        info.dtype
    );
    // The token embedding table is read by the native-layout gather kernels.
    // When embeddings are tied it doubles as the LM head and also needs the
    // SoA decode layout; untied it never sees a GEMV and stays native-only.
    let is_embed = gguf_name == "token_embd.weight";
    let tied_embed = is_embed && !gguf.content.has_tensor("output.weight");
    match info.dtype {
        GgmlType::Q8_0 => load_device_q8_0_soa(&gguf_name, info.shape.clone(), data, stream)
            .map(MatrixWeight::single),
        GgmlType::Q6K if !is_embed || tied_embed => {
            load_device_q6k_soa(&gguf_name, info.shape.clone(), data, is_embed, stream)
                .map(MatrixWeight::single)
        }
        GgmlType::Q4K if !is_embed || tied_embed => {
            load_device_q4k_soa(&gguf_name, info.shape.clone(), data, is_embed, stream)
                .map(MatrixWeight::single)
        }
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => {
            let device_tensor = load_native_device_bytes(&gguf_name, data, stream)?;
            let weight = Weight::quantized(info.dtype, device_tensor, info.shape.clone())?;
            Ok(MatrixWeight::single(weight))
        }
        GgmlType::F16 | GgmlType::F32 => {
            let host = load_host_f16_gguf(gguf, engine_name)?;
            let tensor = copy_f16_to_device(host, stream)?;
            Ok(MatrixWeight::single(Weight::f16(tensor)?))
        }
        other => bail!("unsupported ggml type {other} for tensor `{gguf_name}`"),
    }
}

/// Host-side Q6K SoA repack for the tile-parallel decode GEMV.
///
/// Element decode order matches Candle's `dequantize_q6k` exactly; `qs`
/// stores the 6-bit value minus 32 (fits i8), `sc` the native per-16-element
/// i8 sub-scales (already element-group ordered on disk), `d` the per-block
/// f16 super-scale.
pub fn repack_q6k_soa_host(
    data: &[u8],
    rows: usize,
    k: usize,
) -> Result<(Vec<i8>, Vec<i8>, Vec<f16>)> {
    ensure!(k.is_multiple_of(256), "Q6K K must be divisible by 256");
    let blocks_per_row = k / 256;
    let type_size = GgmlType::Q6K.type_size();
    ensure!(
        data.len() == rows * blocks_per_row * type_size,
        "Q6K byte length mismatch"
    );
    let mut qs = vec![0i8; rows * k];
    let mut sc = vec![0i8; rows * k / 16];
    let mut d = vec![f16::from_f32(0.0); rows * blocks_per_row];
    for (block_idx, block) in data.chunks_exact(type_size).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d_bits = u16::from_le_bytes([block[208], block[209]]);
        d[block_idx] = f16::from_bits(d_bits);
        for (j, &s) in scales.iter().enumerate() {
            sc[block_idx * 16 + j] = s as i8;
        }
        let out = &mut qs[block_idx * 256..(block_idx + 1) * 256];
        for half in 0..2 {
            let ql = &ql[64 * half..];
            let qh = &qh[32 * half..];
            let base = 128 * half;
            for l in 0..32 {
                out[base + l] = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                out[base + l + 32] = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                out[base + l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                out[base + l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
            }
        }
    }
    Ok((qs, sc, d))
}

/// Host-side Q4K SoA repack for the tile-parallel decode GEMV.
///
/// `qs` is plane-packed: byte `j` of a row holds element `j` in the low
/// nibble and element `j + k/2` in the high nibble, so the kernel unpacks
/// with one mask and one shift and both element planes stay contiguous.
/// `sc`/`mins` are the 6-bit sub-scales unpacked to one byte per 32-element
/// group via the same `get_scale_min_k4` used by the CPU reference.
pub fn repack_q4k_soa_host(
    data: &[u8],
    rows: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<f16>, Vec<f16>)> {
    ensure!(k.is_multiple_of(512), "Q4K K must be divisible by 512");
    let blocks_per_row = k / 256;
    let type_size = GgmlType::Q4K.type_size();
    ensure!(
        data.len() == rows * blocks_per_row * type_size,
        "Q4K byte length mismatch"
    );
    let mut sc = vec![f16::from_f32(0.0); rows * k / 32];
    let mut mins = vec![f16::from_f32(0.0); rows * k / 32];
    // Element-order 4-bit values for the whole tensor, then plane-pack per row.
    let mut vals = vec![0u8; rows * k];
    for (block_idx, block) in data.chunks_exact(type_size).enumerate() {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let scales = &block[4..16];
        for j in 0..8 {
            let (s, m) = crate::dequant::get_scale_min_k4(j, scales);
            sc[block_idx * 8 + j] = f16::from_f32(d * s as f32);
            mins[block_idx * 8 + j] = f16::from_f32(dmin * m as f32);
        }
        let q = &block[16..144];
        let out = &mut vals[block_idx * 256..(block_idx + 1) * 256];
        for group in 0..4 {
            let q = &q[group * 32..group * 32 + 32];
            let base = group * 64;
            for (l, &byte) in q.iter().enumerate() {
                out[base + l] = byte & 0xF;
                out[base + 32 + l] = byte >> 4;
            }
        }
    }
    let half_k = k / 2;
    let mut qs = vec![0u8; rows * half_k];
    for row in 0..rows {
        let row_vals = &vals[row * k..(row + 1) * k];
        let row_qs = &mut qs[row * half_k..(row + 1) * half_k];
        for (j, out) in row_qs.iter_mut().enumerate() {
            *out = row_vals[j] | (row_vals[j + half_k] << 4);
        }
    }
    Ok((qs, sc, mins))
}

fn copy_vec_to_device_2d<T: cutile::DType>(
    host: Vec<T>,
    rows: usize,
    cols: usize,
    label: &str,
    gguf_name: &str,
    stream: &Arc<cuda_core::Stream>,
) -> Result<Arc<Tensor<T>>> {
    let host = Arc::new(host);
    let dev = api::copy_host_vec_to_device(&host)
        .sync_on(stream)
        .map_err(|e| anyhow::anyhow!("copy {label} for `{gguf_name}` to device failed: {e:?}"))?
        .reshape(&[rows, cols])
        .map_err(|e| anyhow::anyhow!("reshape {label} for `{gguf_name}` failed: {e:?}"))?;
    Ok(Arc::new(dev))
}

fn load_native_device_bytes(
    gguf_name: &str,
    data: &[u8],
    stream: &Arc<cuda_core::Stream>,
) -> Result<Arc<Tensor<u8>>> {
    let host = Arc::new(data.to_vec());
    let dev = api::copy_host_vec_to_device(&host)
        .sync_on(stream)
        .map_err(|e| anyhow::anyhow!("copy native `{gguf_name}` to device failed: {e:?}"))?
        .reshape(&[data.len()])
        .map_err(|e| anyhow::anyhow!("reshape native `{gguf_name}` failed: {e:?}"))?;
    Ok(Arc::new(dev))
}

fn load_device_q6k_soa(
    gguf_name: &str,
    shape: Vec<usize>,
    data: &[u8],
    keep_native: bool,
    stream: &Arc<cuda_core::Stream>,
) -> Result<Weight> {
    ensure!(
        shape.len() == 2,
        "Q6K tensor `{gguf_name}` must be rank-2, got {shape:?}"
    );
    let rows = shape[0];
    let k = shape[1];
    let (qs, sc, d) = repack_q6k_soa_host(data, rows, k)?;
    let native = if keep_native {
        Some(load_native_device_bytes(gguf_name, data, stream)?)
    } else {
        None
    };
    let qs = copy_vec_to_device_2d(qs, rows, k, "Q6K qs", gguf_name, stream)?;
    let sc = copy_vec_to_device_2d(sc, rows, k / 16, "Q6K sc", gguf_name, stream)?;
    let d = copy_vec_to_device_2d(d, rows, k / 256, "Q6K d", gguf_name, stream)?;
    Weight::q6k_soa(native, qs, sc, d, shape)
}

fn load_device_q4k_soa(
    gguf_name: &str,
    shape: Vec<usize>,
    data: &[u8],
    keep_native: bool,
    stream: &Arc<cuda_core::Stream>,
) -> Result<Weight> {
    ensure!(
        shape.len() == 2,
        "Q4K tensor `{gguf_name}` must be rank-2, got {shape:?}"
    );
    let rows = shape[0];
    let k = shape[1];
    let (qs, sc, mins) = repack_q4k_soa_host(data, rows, k)?;
    let native = if keep_native {
        Some(load_native_device_bytes(gguf_name, data, stream)?)
    } else {
        None
    };
    let qs = copy_vec_to_device_2d(qs, rows, k / 2, "Q4K qs", gguf_name, stream)?;
    let sc = copy_vec_to_device_2d(sc, rows, k / 32, "Q4K sc", gguf_name, stream)?;
    let mins = copy_vec_to_device_2d(mins, rows, k / 32, "Q4K mins", gguf_name, stream)?;
    Weight::q4k_soa(native, qs, sc, mins, shape)
}

fn load_device_q8_0_soa(
    gguf_name: &str,
    shape: Vec<usize>,
    data: &[u8],
    stream: &Arc<cuda_core::Stream>,
) -> Result<Weight> {
    ensure!(
        shape.len() == 2,
        "Q8_0 tensor `{gguf_name}` must be rank-2, got {shape:?}"
    );
    let rows = shape[0];
    let k = shape[1];
    ensure!(
        k.is_multiple_of(32),
        "Q8_0 tensor `{gguf_name}` K must be divisible by 32, got {k}"
    );
    let blocks_per_row = k / 32;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(GgmlType::Q8_0.type_size()))
        .context("Q8_0 byte size overflows usize")?;
    ensure!(
        data.len() == expected,
        "Q8_0 tensor `{gguf_name}` byte length mismatch: got {}, expected {expected}",
        data.len()
    );

    let mut qs = Vec::<i8>::with_capacity(rows * k);
    let mut scales = Vec::<f16>::with_capacity(rows * blocks_per_row);
    for row in 0..rows {
        let row_base = row * blocks_per_row * 34;
        for block in 0..blocks_per_row {
            let block_base = row_base + block * 34;
            let d_bits = u16::from_le_bytes([data[block_base], data[block_base + 1]]);
            scales.push(f16::from_bits(d_bits));
            for j in 0..32 {
                qs.push(data[block_base + 2 + j] as i8);
            }
        }
    }

    let native_host = Arc::new(data.to_vec());
    let qs_host = Arc::new(qs);
    let scales_host = Arc::new(scales);
    let native_dev = api::copy_host_vec_to_device(&native_host)
        .sync_on(stream)
        .map_err(|e| anyhow::anyhow!("copy Q8_0 native `{gguf_name}` to device failed: {e:?}"))?
        .reshape(&[data.len()])
        .map_err(|e| anyhow::anyhow!("reshape Q8_0 native `{gguf_name}` failed: {e:?}"))?;
    let qs_dev = api::copy_host_vec_to_device(&qs_host)
        .sync_on(stream)
        .map_err(|e| anyhow::anyhow!("copy Q8_0 qs `{gguf_name}` to device failed: {e:?}"))?
        .reshape(&[rows, k])
        .map_err(|e| anyhow::anyhow!("reshape Q8_0 qs `{gguf_name}` failed: {e:?}"))?;
    let scales_dev = api::copy_host_vec_to_device(&scales_host)
        .sync_on(stream)
        .map_err(|e| anyhow::anyhow!("copy Q8_0 scales `{gguf_name}` to device failed: {e:?}"))?
        .reshape(&[rows, blocks_per_row])
        .map_err(|e| anyhow::anyhow!("reshape Q8_0 scales `{gguf_name}` failed: {e:?}"))?;

    Weight::q8_0_soa(
        Arc::new(native_dev),
        Arc::new(qs_dev),
        Arc::new(scales_dev),
        shape,
    )
}

fn load_host_f16_gguf(gguf: &GgufFile, engine_name: &str) -> Result<HostTensor> {
    let gguf_name = map_engine_tensor_name(engine_name, &gguf.content)
        .with_context(|| format!("failed to map engine tensor `{engine_name}` to GGUF name"))?;
    let (info, data) = gguf.tensor_data(&gguf_name)?;
    ensure!(
        info.dtype.is_supported_for_phase1(),
        "unsupported ggml type {} for tensor `{gguf_name}`",
        info.dtype
    );
    let elem_count = info.elem_count()?;
    let data = dequantize_to_f16(info.dtype, data, elem_count, &gguf_name).with_context(|| {
        format!(
            "failed to dequantize GGUF tensor `{gguf_name}` ({})",
            info.dtype
        )
    })?;
    Ok(HostTensor {
        data,
        shape: info.shape.clone(),
    })
}

fn map_engine_tensor_name(engine_name: &str, content: &crate::gguf::Content) -> Result<String> {
    match engine_name {
        "model.embed_tokens.weight" => return Ok("token_embd.weight".to_string()),
        "model.norm.weight" => return Ok("output_norm.weight".to_string()),
        "lm_head.weight" => {
            return Ok(if content.has_tensor("output.weight") {
                "output.weight".to_string()
            } else {
                "token_embd.weight".to_string()
            });
        }
        _ => {}
    }

    let rest = engine_name
        .strip_prefix("model.layers.")
        .with_context(|| format!("unrecognized engine tensor name `{engine_name}`"))?;
    let (idx, suffix) = rest
        .split_once('.')
        .with_context(|| format!("unrecognized layer tensor name `{engine_name}`"))?;
    idx.parse::<usize>()
        .with_context(|| format!("invalid layer index `{idx}` in `{engine_name}`"))?;

    let gguf_suffix = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => bail!("unrecognized engine tensor suffix `{suffix}` in `{engine_name}`"),
    };
    Ok(format!("blk.{idx}.{gguf_suffix}"))
}

use cutile::core::bf16;

fn cast_to_f16(dtype: Dtype, data: &[u8]) -> Result<Vec<f16>> {
    match dtype {
        Dtype::F16 => {
            let mut out = Vec::with_capacity(data.len() / 2);
            for bytes in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                out.push(f16::from_bits(bits));
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(data.len() / 2);
            for bytes in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                out.push(f16::from_f32(bf16::from_bits(bits).to_f32()));
            }
            Ok(out)
        }
        Dtype::F32 => {
            let mut out = Vec::with_capacity(data.len() / 4);
            for bytes in data.chunks_exact(4) {
                let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                out.push(f16::from_f32(x));
            }
            Ok(out)
        }
        Dtype::F64 => {
            let mut out = Vec::with_capacity(data.len() / 8);
            for bytes in data.chunks_exact(8) {
                let x = f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]);
                out.push(f16::from_f32(x as f32));
            }
            Ok(out)
        }
        other => bail!("unsupported dtype for fp16 cast: {other:?}"),
    }
}
