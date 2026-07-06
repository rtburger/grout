// Ported from Candle's CPU quantization reference:
// /home/rtb/code/agent/candle/candle-core/src/quantized/k_quants.rs
// and dtype table in:
// /home/rtb/code/agent/candle/candle-core/src/quantized/mod.rs

use anyhow::{Result, bail, ensure};
use cutile::core::f16;
use std::fmt;

pub const QK_K: usize = 256;
pub const QK8_0: usize = 32;
pub const K_SCALE_SIZE: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlType {
    pub fn from_u32(value: u32) -> Result<Self> {
        let dtype = match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            other => bail!("unknown ggml dtype id {other}"),
        };
        Ok(dtype)
    }

    pub fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::BF16 => 30,
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => QK_K / 16 + QK_K / 4 + 2 * 2,
            Self::Q3K => QK_K / 8 + QK_K / 4 + 12 + 2,
            Self::Q4K => QK_K / 2 + K_SCALE_SIZE + 2 * 2,
            Self::Q5K => QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE,
            Self::Q6K => 3 * QK_K / 4 + QK_K / 16 + 2,
            Self::Q8K => 4 + QK_K + QK_K / 16 * 2,
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => QK8_0,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => QK_K,
        }
    }

    pub fn is_supported_for_phase1(self) -> bool {
        matches!(
            self,
            Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8_0 | Self::F16 | Self::F32
        )
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
        };
        f.write_str(s)
    }
}

pub fn dequantize_to_f32(
    dtype: GgmlType,
    data: &[u8],
    elem_count: usize,
    tensor_name: &str,
) -> Result<Vec<f32>> {
    ensure!(
        elem_count.is_multiple_of(dtype.block_size()),
        "tensor `{tensor_name}` has {elem_count} elements, not divisible by {dtype} block size {}",
        dtype.block_size()
    );
    let expected = elem_count / dtype.block_size() * dtype.type_size();
    ensure!(
        data.len() == expected,
        "tensor `{tensor_name}` of type {dtype} expected {expected} bytes for {elem_count} elements, got {}",
        data.len()
    );

    match dtype {
        GgmlType::F32 => dequantize_f32(data),
        GgmlType::F16 => dequantize_f16_to_f32(data),
        GgmlType::Q8_0 => dequantize_q8_0(data, elem_count),
        GgmlType::Q4K => dequantize_q4k(data, elem_count),
        GgmlType::Q5K => dequantize_q5k(data, elem_count),
        GgmlType::Q6K => dequantize_q6k(data, elem_count),
        other => bail!("unsupported ggml type {other} for tensor `{tensor_name}`"),
    }
}

pub fn dequantize_to_f16(
    dtype: GgmlType,
    data: &[u8],
    elem_count: usize,
    tensor_name: &str,
) -> Result<Vec<f16>> {
    ensure!(
        elem_count.is_multiple_of(dtype.block_size()),
        "tensor `{tensor_name}` has {elem_count} elements, not divisible by {dtype} block size {}",
        dtype.block_size()
    );
    let expected = elem_count / dtype.block_size() * dtype.type_size();
    ensure!(
        data.len() == expected,
        "tensor `{tensor_name}` of type {dtype} expected {expected} bytes for {elem_count} elements, got {}",
        data.len()
    );

    if dtype == GgmlType::F16 {
        let mut out = Vec::with_capacity(elem_count);
        for bytes in data.chunks_exact(2) {
            out.push(f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])));
        }
        return Ok(out);
    }

    let f32s = dequantize_to_f32(dtype, data, elem_count, tensor_name)?;
    Ok(f32s.into_iter().map(f16::from_f32).collect())
}

fn dequantize_f32(data: &[u8]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(data.len() / 4);
    for bytes in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
    }
    Ok(out)
}

fn dequantize_f16_to_f32(data: &[u8]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(data.len() / 2);
    for bytes in data.chunks_exact(2) {
        out.push(f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32());
    }
    Ok(out)
}

fn dequantize_q8_0(data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; elem_count];
    for (block_idx, block) in data.chunks_exact(GgmlType::Q8_0.type_size()).enumerate() {
        let d = f16::from_bits(read_u16(block, 0)).to_f32();
        let qs = &block[2..34];
        for j in 0..QK8_0 {
            out[block_idx * QK8_0 + j] = (qs[j] as i8) as f32 * d;
        }
    }
    Ok(out)
}

fn dequantize_q4k(data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; elem_count];
    for (block_idx, block) in data.chunks_exact(GgmlType::Q4K.type_size()).enumerate() {
        let d = f16::from_bits(read_u16(block, 0)).to_f32();
        let min = f16::from_bits(read_u16(block, 2)).to_f32();
        let scales = &block[4..16];
        let q = &block[16..144];
        let y = &mut out[block_idx * QK_K..(block_idx + 1) * QK_K];
        let mut is = 0;
        let mut ys_index = 0;

        for j in (0..QK_K).step_by(64) {
            let q = &q[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for q in q {
                y[ys_index] = d1 * (q & 0xF) as f32 - m1;
                ys_index += 1;
            }
            for q in q {
                y[ys_index] = d2 * (q >> 4) as f32 - m2;
                ys_index += 1;
            }
            is += 2;
        }
    }
    Ok(out)
}

fn dequantize_q5k(data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; elem_count];
    for (block_idx, block) in data.chunks_exact(GgmlType::Q5K.type_size()).enumerate() {
        let d = f16::from_bits(read_u16(block, 0)).to_f32();
        let min = f16::from_bits(read_u16(block, 2)).to_f32();
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];
        let y = &mut out[block_idx * QK_K..(block_idx + 1) * QK_K];
        let mut is = 0;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        let mut ys_index = 0;

        for j in (0..QK_K).step_by(64) {
            let ql = &ql[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for (ql, qh) in ql.iter().zip(qh) {
                let to_add = if qh & u1 != 0 { 16f32 } else { 0f32 };
                y[ys_index] = d1 * ((ql & 0xF) as f32 + to_add) - m1;
                ys_index += 1;
            }
            for (ql, qh) in ql.iter().zip(qh) {
                let to_add = if qh & u2 != 0 { 16f32 } else { 0f32 };
                y[ys_index] = d2 * ((ql >> 4) as f32 + to_add) - m2;
                ys_index += 1;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

fn dequantize_q6k(data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; elem_count];
    for (idx_x, x) in data.chunks_exact(GgmlType::Q6K.type_size()).enumerate() {
        let ql = &x[0..128];
        let qh = &x[128..192];
        let sc = &x[192..208];
        let d = f16::from_bits(read_u16(x, 208)).to_f32();
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let ys = &mut out[idx_x * QK_K + n..];
            let sc = &sc[8 * idx..];
            let ql = &ql[64 * idx..];
            let qh = &qh[32 * idx..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                ys[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                ys[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                ys[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                ys[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
    Ok(out)
}

pub(crate) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}
