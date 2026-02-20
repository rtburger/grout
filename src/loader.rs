use crate::config::SafetensorsIndex;
use anyhow::{Context, Result, bail};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tile_rust::api;
use tile_rust::api::DeviceOperationDynamicReshape;
use tile_rust::half::{bf16, f16};
use tile_rust::tensor::Tensor;

#[derive(Debug)]
pub struct HostTensor {
    pub data: Vec<f16>,
    pub shape: Vec<usize>,
}

pub struct WeightLoader {
    model_dir: PathBuf,
    weight_map: HashMap<String, String>,
    shards: HashMap<String, memmap2::Mmap>,
}

impl WeightLoader {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_text)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;

        let mut shard_files = HashSet::new();
        for shard in index.weight_map.values() {
            shard_files.insert(shard.clone());
        }

        let mut shards = HashMap::new();
        for shard in shard_files {
            let shard_path = model_dir.join(&shard);
            let file = File::open(&shard_path)
                .with_context(|| format!("failed to open {}", shard_path.display()))?;
            let mmap = unsafe { MmapOptions::new().map(&file) }
                .with_context(|| format!("failed to mmap {}", shard_path.display()))?;
            shards.insert(shard, mmap);
        }

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            weight_map: index.weight_map,
            shards,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn load_host_f16(&self, name: &str) -> Result<HostTensor> {
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

    pub async fn load_device_f16(&self, name: &str) -> Result<Arc<Tensor<f16>>> {
        let host = self.load_host_f16(name)?;
        let shape = host.shape.clone();
        let host_data = Arc::new(host.data);
        let device_tensor = api::copy_host_vec_to_device(&host_data)
            .reshape_dyn(shape)
            .await;
        Ok(Arc::new(device_tensor))
    }
}

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
