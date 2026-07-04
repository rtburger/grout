use crate::config::{Qwen3Config, SafetensorsIndex};
use crate::dequant::dequantize_to_f16;
use crate::gguf::GgufFile;
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
    let key = |suffix: &str| format!("{arch}.{suffix}");

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
            .metadata_required(&key("embedding_length"))?
            .to_u32()? as usize,
        intermediate_size: metadata
            .metadata_required(&key("feed_forward_length"))?
            .to_u32()? as usize,
        num_hidden_layers: metadata.metadata_required(&key("block_count"))?.to_u32()? as usize,
        num_attention_heads: metadata
            .metadata_required(&key("attention.head_count"))?
            .to_u32()? as usize,
        num_key_value_heads: metadata
            .metadata_required(&key("attention.head_count_kv"))?
            .to_u32()? as usize,
        head_dim: metadata
            .metadata_required(&key("attention.key_length"))?
            .to_u32()? as usize,
        rms_norm_eps: metadata
            .metadata_required(&key("attention.layer_norm_rms_epsilon"))?
            .to_f32()?,
        rope_theta: metadata
            .metadata_required(&key("rope.freq_base"))?
            .to_f32()?,
        max_position_embeddings: metadata
            .metadata_required(&key("context_length"))?
            .to_u32()? as usize,
        tie_word_embeddings: !metadata.has_tensor("output.weight"),
        use_sliding_window: false,
        eos_token_id: metadata
            .metadata_required("tokenizer.ggml.eos_token_id")?
            .to_u32()?,
    };
    println!("cfg: {cfg:#?}");
    Ok(cfg)
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
