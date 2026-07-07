// Ported from Candle's GGUF reader structure:
// /home/rtb/code/agent/candle/candle-core/src/quantized/gguf_file.rs
// Stripped to plain structs over memmap2 for grout's loader.

use crate::dequant::GgmlType;
use anyhow::{Context, Result, bail, ensure};
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

pub const DEFAULT_ALIGNMENT: u64 = 32;
const GGUF_MAX_STRING_LENGTH: u64 = 1 << 30;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1 << 30;
const GGUF_MAX_TENSOR_DIMS: u32 = 4;
const GGUF_MAX_VALUE_DEPTH: usize = 64;
// Cumulative cap on metadata array elements per file. Each element
// materializes as a ~32-byte Value in RAM while costing as little as one
// disk byte (U8), so claimed lengths must be bounded by in-memory cost,
// not just remaining file bytes — and cumulatively, or sibling arrays
// re-create the same amplification. 16M elements ≈ 512 MiB worst case,
// ~30x headroom over a 152k-token tokenizer's metadata arrays.
const GGUF_MAX_METADATA_ELEMENTS: u64 = 16 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    V2,
    V3,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: GgmlType,
    pub shape: Vec<usize>,
    pub offset: u64,
}

impl TensorInfo {
    pub fn elem_count(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .with_context(|| format!("tensor `{}` element count overflows usize", self.name))
        })
    }

    pub fn size_in_bytes(&self) -> Result<usize> {
        let elem_count = self.elem_count()?;
        ensure!(
            elem_count.is_multiple_of(self.dtype.block_size()),
            "tensor `{}` has {elem_count} elements, not divisible by {} block size {}",
            self.name,
            self.dtype,
            self.dtype.block_size()
        );
        elem_count
            .checked_div(self.dtype.block_size())
            .and_then(|blocks| blocks.checked_mul(self.dtype.type_size()))
            .with_context(|| format!("tensor `{}` byte size overflows usize", self.name))
    }
}

#[derive(Debug)]
pub struct Content {
    pub version: Version,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

impl Content {
    pub fn metadata_required(&self, key: &str) -> Result<&Value> {
        self.metadata
            .get(key)
            .with_context(|| format!("GGUF metadata key `{key}` not found"))
    }

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor_infos
            .get(name)
            .with_context(|| format!("GGUF tensor `{name}` not found"))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_infos.contains_key(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    String,
    Array,
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        let ty = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => bail!("GGUF unrecognized value type {other:#08x}"),
        };
        Ok(ty)
    }

    fn min_disk_size(self) -> u64 {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String => 8,
            Self::Array => 12,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn to_u32(&self) -> Result<u32> {
        let value = self.to_u64()?;
        u32::try_from(value)
            .with_context(|| format!("GGUF metadata value {self:?} does not fit u32"))
    }

    pub fn to_u64(&self) -> Result<u64> {
        match self {
            Self::U8(v) => Ok(*v as u64),
            Self::U16(v) => Ok(*v as u64),
            Self::U32(v) => Ok(*v as u64),
            Self::U64(v) => Ok(*v),
            Self::Bool(v) => Ok(*v as u64),
            other => bail!("GGUF metadata value is not an unsigned integer: {other:?}"),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            Self::F64(v) => Ok(*v as f32),
            other => bail!("GGUF metadata value is not f32/f64: {other:?}"),
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            other => bail!("GGUF metadata value is not bool: {other:?}"),
        }
    }

    pub fn to_string(&self) -> Result<&str> {
        match self {
            Self::String(v) => Ok(v.as_str()),
            other => bail!("GGUF metadata value is not string: {other:?}"),
        }
    }

    pub fn as_array(&self) -> Result<&[Value]> {
        match self {
            Self::Array(v) => Ok(v),
            other => bail!("GGUF metadata value is not array: {other:?}"),
        }
    }
}

pub struct GgufFile {
    path: PathBuf,
    mmap: Mmap,
    pub content: Content,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;
        let content = parse_content(&mmap[..])
            .with_context(|| format!("failed to parse GGUF {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            content,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensor_data(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let info = self.content.tensor_info(name)?;
        let size = info.size_in_bytes()?;
        let start = self
            .content
            .tensor_data_offset
            .checked_add(info.offset)
            .with_context(|| format!("tensor `{name}` file offset overflows u64"))?
            as usize;
        let end = start
            .checked_add(size)
            .with_context(|| format!("tensor `{name}` end offset overflows usize"))?;
        ensure!(
            end <= self.mmap.len(),
            "tensor `{name}` needs bytes [{start}..{end}), file has {} bytes",
            self.mmap.len()
        );
        Ok((info, &self.mmap[start..end]))
    }
}

fn parse_content(bytes: &[u8]) -> Result<Content> {
    let file_size = bytes.len() as u64;
    let mut reader = Cursor::new(bytes);

    let magic = read_u32(&mut reader)?;
    ensure!(
        magic == 0x4655_4747,
        "GGUF magic must be little-endian 0x46554747, got {magic:#010x}"
    );
    let version_raw = read_u32(&mut reader)?;
    let version = match version_raw {
        2 => Version::V2,
        3 => Version::V3,
        other => bail!("GGUF version {other} is unsupported; expected v2 or v3"),
    };

    let tensor_count = read_u64(&mut reader)?;
    let metadata_kv_count = read_u64(&mut reader)?;
    ensure!(
        tensor_count <= GGUF_MAX_ARRAY_ELEMENTS,
        "GGUF tensor_count {tensor_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
    );
    ensure!(
        metadata_kv_count <= GGUF_MAX_ARRAY_ELEMENTS,
        "GGUF metadata_kv_count {metadata_kv_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
    );

    let mut metadata = HashMap::new();
    let mut element_budget = GGUF_MAX_METADATA_ELEMENTS;
    for _ in 0..metadata_kv_count {
        let key = read_string(&mut reader, file_size)?;
        let value_type = ValueType::from_u32(read_u32(&mut reader)?)?;
        let value = read_value(&mut reader, value_type, 0, file_size, &mut element_budget)?;
        metadata.insert(key, value);
    }

    let mut tensor_infos = HashMap::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut reader, file_size)?;
        let n_dimensions = read_u32(&mut reader)?;
        ensure!(
            n_dimensions <= GGUF_MAX_TENSOR_DIMS,
            "GGUF tensor `{name}` has {n_dimensions} dimensions, max is {GGUF_MAX_TENSOR_DIMS}"
        );
        let mut shape = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            let dim = read_u64(&mut reader)?;
            shape.push(usize::try_from(dim).with_context(|| {
                format!("GGUF tensor `{name}` dimension {dim} does not fit usize")
            })?);
        }
        // GGUF stores ggml-order dims, innermost first. Candle reverses them
        // before exposing tensor shapes; use the same convention here.
        shape.reverse();
        let dtype = GgmlType::from_u32(read_u32(&mut reader)?)?;
        let offset = read_u64(&mut reader)?;
        let info = TensorInfo {
            name: name.clone(),
            dtype,
            shape,
            offset,
        };
        tensor_infos.insert(name, info);
    }

    let position = reader.position();
    let alignment = match metadata.get("general.alignment") {
        Some(Value::U8(v)) => *v as u64,
        Some(Value::U16(v)) => *v as u64,
        Some(Value::U32(v)) => *v as u64,
        Some(Value::U64(v)) => *v,
        Some(Value::I8(v)) if *v >= 0 => *v as u64,
        Some(Value::I16(v)) if *v >= 0 => *v as u64,
        Some(Value::I32(v)) if *v >= 0 => *v as u64,
        Some(Value::I64(v)) if *v >= 0 => *v as u64,
        _ => DEFAULT_ALIGNMENT,
    };
    ensure!(alignment > 0, "GGUF general.alignment must be positive");
    let tensor_data_offset = align_to(position, alignment)?;
    ensure!(
        tensor_data_offset <= file_size,
        "GGUF tensor data offset {tensor_data_offset} exceeds file size {file_size}"
    );

    Ok(Content {
        version,
        metadata,
        tensor_infos,
        tensor_data_offset,
    })
}

fn read_value(
    reader: &mut Cursor<&[u8]>,
    value_type: ValueType,
    depth: usize,
    file_size: u64,
    element_budget: &mut u64,
) -> Result<Value> {
    ensure!(
        depth <= GGUF_MAX_VALUE_DEPTH,
        "GGUF value nesting depth exceeds max {GGUF_MAX_VALUE_DEPTH}"
    );
    let value = match value_type {
        ValueType::U8 => Value::U8(read_u8(reader)?),
        ValueType::I8 => Value::I8(read_u8(reader)? as i8),
        ValueType::U16 => Value::U16(read_u16(reader)?),
        ValueType::I16 => Value::I16(read_u16(reader)? as i16),
        ValueType::U32 => Value::U32(read_u32(reader)?),
        ValueType::I32 => Value::I32(read_u32(reader)? as i32),
        ValueType::U64 => Value::U64(read_u64(reader)?),
        ValueType::I64 => Value::I64(read_u64(reader)? as i64),
        ValueType::F32 => Value::F32(f32::from_bits(read_u32(reader)?)),
        ValueType::F64 => Value::F64(f64::from_bits(read_u64(reader)?)),
        ValueType::Bool => match read_u8(reader)? {
            0 => Value::Bool(false),
            1 => Value::Bool(true),
            other => bail!("GGUF invalid bool value {other}"),
        },
        ValueType::String => Value::String(read_string(reader, file_size)?),
        ValueType::Array => {
            let element_type = ValueType::from_u32(read_u32(reader)?)?;
            let len = read_u64(reader)?;
            ensure!(
                len <= GGUF_MAX_ARRAY_ELEMENTS,
                "GGUF array length {len} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
            );
            ensure!(
                len <= *element_budget,
                "GGUF metadata arrays exceed the total element budget \
                 {GGUF_MAX_METADATA_ELEMENTS} (array of {len} elements, {} budget remaining)",
                *element_budget
            );
            *element_budget -= len;
            let needed = len.saturating_mul(element_type.min_disk_size());
            ensure!(
                needed <= remaining(reader, file_size)?,
                "GGUF array of {len} elements needs at least {needed} bytes, only {} remaining",
                remaining(reader, file_size)?
            );
            // Grow as elements parse instead of eagerly reserving 32 bytes
            // per claimed element off an attacker-controlled length.
            let mut values = Vec::with_capacity(len.min(4096) as usize);
            for _ in 0..len {
                values.push(read_value(
                    reader,
                    element_type,
                    depth + 1,
                    file_size,
                    element_budget,
                )?);
            }
            Value::Array(values)
        }
    };
    Ok(value)
}

fn read_string(reader: &mut Cursor<&[u8]>, file_size: u64) -> Result<String> {
    let len = read_u64(reader)?;
    ensure!(
        len <= GGUF_MAX_STRING_LENGTH,
        "GGUF string length {len} exceeds max {GGUF_MAX_STRING_LENGTH}"
    );
    ensure!(
        len <= remaining(reader, file_size)?,
        "GGUF string length {len} exceeds remaining file bytes {}",
        remaining(reader, file_size)?
    );
    let mut bytes = vec![0u8; len as usize];
    reader.read_exact(&mut bytes)?;
    while let Some(0) = bytes.last() {
        bytes.pop();
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn remaining(reader: &Cursor<&[u8]>, file_size: u64) -> Result<u64> {
    Ok(file_size.saturating_sub(reader.position()))
}

fn align_to(position: u64, alignment: u64) -> Result<u64> {
    position
        .checked_add(alignment - 1)
        .map(|x| x / alignment * alignment)
        .context("GGUF alignment calculation overflowed")
}

fn read_exact<const N: usize>(reader: &mut Cursor<&[u8]>) -> Result<[u8; N]> {
    let mut bytes = [0u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_u8(reader: &mut Cursor<&[u8]>) -> Result<u8> {
    Ok(read_exact::<1>(reader)?[0])
}

fn read_u16(reader: &mut Cursor<&[u8]>) -> Result<u16> {
    Ok(u16::from_le_bytes(read_exact::<2>(reader)?))
}

fn read_u32(reader: &mut Cursor<&[u8]>) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact::<4>(reader)?))
}

fn read_u64(reader: &mut Cursor<&[u8]>) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact::<8>(reader)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(kv_count: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&kv_count.to_le_bytes());
        b
    }

    fn push_key(b: &mut Vec<u8>, key: &str) {
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
    }

    fn push_u8_array_header(b: &mut Vec<u8>, len: u64) {
        b.extend_from_slice(&9u32.to_le_bytes()); // ValueType::Array
        b.extend_from_slice(&0u32.to_le_bytes()); // element type U8
        b.extend_from_slice(&len.to_le_bytes());
    }

    /// H2 regression: a claimed U8 array length within the on-disk cap but
    /// far beyond in-memory cost must be rejected by the element budget
    /// before any allocation — pre-fix this line reserved 32 bytes per
    /// claimed element (32 GiB here) before reading a single element.
    #[test]
    fn array_len_beyond_element_budget_is_rejected_before_allocating() {
        let mut b = header(1);
        push_key(&mut b, "k");
        push_u8_array_header(&mut b, 1 << 30);
        let err = parse_content(&b).unwrap_err();
        assert!(
            err.to_string().contains("element budget"),
            "got: {err:#}"
        );
    }

    /// The budget is cumulative: sibling arrays individually under the cap
    /// must not amplify past it together.
    #[test]
    fn sibling_arrays_cannot_exceed_the_budget_cumulatively() {
        let half = GGUF_MAX_METADATA_ELEMENTS / 2 + 1;
        let mut b = header(2);
        push_key(&mut b, "a");
        push_u8_array_header(&mut b, half);
        b.resize(b.len() + half as usize, 0); // array `a` element bytes
        push_key(&mut b, "b");
        push_u8_array_header(&mut b, half);
        b.resize(b.len() + half as usize, 0); // array `b` element bytes
        let err = parse_content(&b).unwrap_err();
        assert!(
            err.to_string().contains("element budget"),
            "got: {err:#}"
        );
    }

    #[test]
    fn small_metadata_array_still_parses() {
        let mut b = header(1);
        push_key(&mut b, "k");
        push_u8_array_header(&mut b, 3);
        b.extend_from_slice(&[7u8, 8, 9]);
        // Pad to the default alignment so the (empty) tensor-data offset
        // check at the end of parsing passes.
        let aligned = b.len().div_ceil(DEFAULT_ALIGNMENT as usize) * DEFAULT_ALIGNMENT as usize;
        b.resize(aligned, 0);
        let content = parse_content(&b).expect("valid metadata-only file must parse");
        match content.metadata.get("k") {
            Some(Value::Array(v)) => {
                assert_eq!(v.len(), 3);
                assert!(matches!(v[0], Value::U8(7)));
                assert!(matches!(v[2], Value::U8(9)));
            }
            other => panic!("expected U8 array, got {other:?}"),
        }
    }
}
