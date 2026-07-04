use crate::dequant::GgmlType;
use anyhow::{Context, Result, bail, ensure};
use cutile::core::f16;
use cutile::tensor::Tensor;
use std::sync::Arc;

#[derive(Clone)]
pub enum Weight {
    F16 {
        data: Arc<Tensor<f16>>,
        shape: Vec<usize>,
    },
    Q8_0(QuantizedWeight),
    Q4K(QuantizedWeight),
    Q5K(QuantizedWeight),
    Q6K(QuantizedWeight),
}

#[derive(Clone)]
pub struct QuantizedWeight {
    pub storage: QuantizedStorage,
    pub shape: Vec<usize>,
}

#[derive(Clone)]
pub enum QuantizedStorage {
    Native {
        data: Arc<Tensor<u8>>,
    },
    Q8_0Soa {
        native: Arc<Tensor<u8>>,
        qs: Arc<Tensor<i8>>,
        scales: Arc<Tensor<f16>>,
    },
}

#[derive(Clone)]
pub struct MatrixWeight {
    parts: Vec<Weight>,
    shape: Vec<usize>,
}

impl Weight {
    pub fn f16(data: Arc<Tensor<f16>>) -> Result<Self> {
        let shape = tensor_shape_usize(&data)?;
        Ok(Self::F16 { data, shape })
    }

    pub fn quantized(dtype: GgmlType, data: Arc<Tensor<u8>>, shape: Vec<usize>) -> Result<Self> {
        ensure!(
            shape.len() == 2,
            "quantized weight must be rank-2, got {shape:?}"
        );
        let elems = shape[0]
            .checked_mul(shape[1])
            .context("quantized weight element count overflows usize")?;
        ensure!(
            elems.is_multiple_of(dtype.block_size()),
            "quantized weight shape {shape:?} has {elems} elements, not divisible by {dtype} block size {}",
            dtype.block_size()
        );
        let expected_bytes = elems / dtype.block_size() * dtype.type_size();
        ensure!(
            data.shape() == [expected_bytes as i32],
            "quantized weight raw buffer shape mismatch for {dtype}: got {:?}, expected [{expected_bytes}]",
            data.shape()
        );
        let q = QuantizedWeight {
            storage: QuantizedStorage::Native { data },
            shape,
        };
        match dtype {
            GgmlType::Q8_0 => Ok(Self::Q8_0(q)),
            GgmlType::Q4K => Ok(Self::Q4K(q)),
            GgmlType::Q5K => Ok(Self::Q5K(q)),
            GgmlType::Q6K => Ok(Self::Q6K(q)),
            other => bail!("unsupported quantized weight type {other}"),
        }
    }

    pub fn q8_0_soa(
        native: Arc<Tensor<u8>>,
        qs: Arc<Tensor<i8>>,
        scales: Arc<Tensor<f16>>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        ensure!(
            shape.len() == 2,
            "Q8_0 SoA weight must be rank-2, got {shape:?}"
        );
        let rows = shape[0];
        let k = shape[1];
        ensure!(
            k.is_multiple_of(32),
            "Q8_0 K must be divisible by 32, got {k}"
        );
        let expected_native_bytes = rows
            .checked_mul(k / 32)
            .and_then(|blocks| blocks.checked_mul(GgmlType::Q8_0.type_size()))
            .context("Q8_0 native byte count overflows usize")?;
        ensure!(
            native.shape() == [expected_native_bytes as i32],
            "Q8_0 native buffer shape mismatch: got {:?}, expected [{expected_native_bytes}]",
            native.shape()
        );
        ensure!(
            qs.shape() == [rows as i32, k as i32],
            "Q8_0 qs shape mismatch: got {:?}, expected [{rows}, {k}]",
            qs.shape()
        );
        ensure!(
            scales.shape() == [rows as i32, (k / 32) as i32],
            "Q8_0 scales shape mismatch: got {:?}, expected [{rows}, {}]",
            scales.shape(),
            k / 32
        );
        Ok(Self::Q8_0(QuantizedWeight {
            storage: QuantizedStorage::Q8_0Soa { native, qs, scales },
            shape,
        }))
    }

    pub fn dtype(&self) -> GgmlType {
        match self {
            Self::F16 { .. } => GgmlType::F16,
            Self::Q8_0(_) => GgmlType::Q8_0,
            Self::Q4K(_) => GgmlType::Q4K,
            Self::Q5K(_) => GgmlType::Q5K,
            Self::Q6K(_) => GgmlType::Q6K,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F16 { shape, .. } => shape,
            Self::Q8_0(q) | Self::Q4K(q) | Self::Q5K(q) | Self::Q6K(q) => &q.shape,
        }
    }

    pub fn rows(&self) -> usize {
        self.shape()[0]
    }

    pub fn cols(&self) -> usize {
        self.shape()[1]
    }

    pub fn elem_count(&self) -> usize {
        self.rows() * self.cols()
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F16 { .. })
    }

    pub fn as_f16(&self) -> Option<&Arc<Tensor<f16>>> {
        match self {
            Self::F16 { data, .. } => Some(data),
            _ => None,
        }
    }

    pub fn as_quantized(&self) -> Option<(GgmlType, &QuantizedWeight)> {
        match self {
            Self::Q8_0(q) => Some((GgmlType::Q8_0, q)),
            Self::Q4K(q) => Some((GgmlType::Q4K, q)),
            Self::Q5K(q) => Some((GgmlType::Q5K, q)),
            Self::Q6K(q) => Some((GgmlType::Q6K, q)),
            Self::F16 { .. } => None,
        }
    }
}

impl QuantizedWeight {
    pub fn native_data(&self) -> Option<&Arc<Tensor<u8>>> {
        match &self.storage {
            QuantizedStorage::Native { data } => Some(data),
            QuantizedStorage::Q8_0Soa { native, .. } => Some(native),
        }
    }

    pub fn q8_0_soa(&self) -> Option<(&Arc<Tensor<i8>>, &Arc<Tensor<f16>>)> {
        match &self.storage {
            QuantizedStorage::Q8_0Soa { qs, scales, .. } => Some((qs, scales)),
            QuantizedStorage::Native { .. } => None,
        }
    }

    pub fn resident_bytes(&self, dtype: GgmlType) -> Result<usize> {
        let elems = self.shape[0]
            .checked_mul(self.shape[1])
            .context("quantized weight element count overflows usize")?;
        match &self.storage {
            QuantizedStorage::Native { data } => Ok(data.size()),
            QuantizedStorage::Q8_0Soa { native, qs, scales } => {
                ensure!(
                    dtype == GgmlType::Q8_0,
                    "SoA storage is only valid for Q8_0"
                );
                let expected = elems / dtype.block_size() * dtype.type_size();
                let native_bytes = native.size();
                let soa_bytes = qs.size() * std::mem::size_of::<i8>()
                    + scales.size() * std::mem::size_of::<f16>();
                ensure!(
                    native_bytes == expected,
                    "Q8_0 native resident bytes mismatch: got {native_bytes}, expected {expected}"
                );
                ensure!(
                    soa_bytes == expected,
                    "Q8_0 SoA resident bytes mismatch: got {soa_bytes}, expected {expected}"
                );
                return Ok(native_bytes + soa_bytes);
            }
        }
        .and_then(|bytes| {
            let expected = elems / dtype.block_size() * dtype.type_size();
            ensure!(
                bytes == expected,
                "quantized resident bytes mismatch for {dtype}: got {bytes}, expected {expected}"
            );
            Ok(bytes)
        })
    }
}

impl MatrixWeight {
    pub fn single(weight: Weight) -> Self {
        let shape = weight.shape().to_vec();
        Self {
            parts: vec![weight],
            shape,
        }
    }

    pub fn row_concat(parts: Vec<Weight>) -> Result<Self> {
        ensure!(
            !parts.is_empty(),
            "row-concat weight requires at least one part"
        );
        let cols = parts[0].cols();
        let mut rows = 0usize;
        for part in &parts {
            ensure!(part.shape().len() == 2, "weight part must be rank-2");
            ensure!(
                part.cols() == cols,
                "row-concat weight column mismatch: got {}, expected {cols}",
                part.cols()
            );
            rows = rows
                .checked_add(part.rows())
                .context("row-concat weight row count overflows usize")?;
        }
        Ok(Self {
            parts,
            shape: vec![rows, cols],
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    pub fn parts(&self) -> &[Weight] {
        &self.parts
    }

    pub fn single_f16(&self) -> Option<&Arc<Tensor<f16>>> {
        if self.parts.len() == 1 {
            self.parts[0].as_f16()
        } else {
            None
        }
    }

    pub fn is_f16_single(&self) -> bool {
        self.single_f16().is_some()
    }

    pub fn has_quantized(&self) -> bool {
        self.parts.iter().any(Weight::is_quantized)
    }

    pub fn max_quantized_elems(&self) -> Option<usize> {
        self.parts
            .iter()
            .filter(|part| part.is_quantized())
            .map(Weight::elem_count)
            .max()
    }

    pub fn row_parts_for_slice(
        &self,
        row_offset: usize,
        out_rows: usize,
    ) -> Result<Vec<(usize, &Weight)>> {
        ensure!(
            row_offset + out_rows <= self.rows(),
            "row slice [{row_offset}..{}) exceeds weight rows {}",
            row_offset + out_rows,
            self.rows()
        );
        let mut result = Vec::new();
        let slice_start = row_offset;
        let slice_end = row_offset + out_rows;
        let mut cursor = 0usize;
        for part in &self.parts {
            let part_start = cursor;
            let part_end = cursor + part.rows();
            let overlap_start = slice_start.max(part_start);
            let overlap_end = slice_end.min(part_end);
            if overlap_start < overlap_end {
                ensure!(
                    overlap_start == part_start && overlap_end == part_end,
                    "quantized row slice [{slice_start}..{slice_end}) cuts through a weight part [{part_start}..{part_end}); load parts must align with projection slices"
                );
                result.push((overlap_start - slice_start, part));
            }
            cursor = part_end;
        }
        ensure!(!result.is_empty(), "row slice selected no weight parts");
        Ok(result)
    }
}

fn tensor_shape_usize<T: cutile::DType>(tensor: &Tensor<T>) -> Result<Vec<usize>> {
    tensor
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).context("tensor shape contains negative dimension"))
        .collect()
}
