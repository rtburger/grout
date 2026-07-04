use anyhow::{Context, Result, ensure};
use candle_core::quantized::{GgmlDType as CandleGgmlDType, QTensor};
use candle_core::{Device, Tensor};
use grout::dequant::{GgmlType, dequantize_to_f32};
use grout::gguf::GgufFile;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::io::Write;

#[test]
fn dequant_matches_candle_for_supported_types() -> Result<()> {
    let cases = [
        (GgmlType::F32, CandleGgmlDType::F32),
        (GgmlType::F16, CandleGgmlDType::F16),
        (GgmlType::Q8_0, CandleGgmlDType::Q8_0),
        (GgmlType::Q4K, CandleGgmlDType::Q4K),
        (GgmlType::Q5K, CandleGgmlDType::Q5K),
        (GgmlType::Q6K, CandleGgmlDType::Q6K),
    ];

    for (ours, candle) in cases {
        assert_dequant_case(ours, candle)?;
    }
    Ok(())
}

fn assert_dequant_case(ours: GgmlType, candle: CandleGgmlDType) -> Result<()> {
    let block = ours.block_size();
    let rows = 3usize;
    let cols = block * 2;
    let elem_count = rows * cols;
    let mut rng = StdRng::seed_from_u64(0x47525554 ^ ours.to_u32() as u64);
    let values: Vec<f32> = (0..elem_count)
        .map(|_| rng.gen_range(-3.0f32..3.0f32))
        .collect();

    let tensor = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?;
    let qtensor = QTensor::quantize(&tensor, candle)?;
    let raw = qtensor.data()?;

    let expected = qtensor
        .dequantize(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actual = dequantize_to_f32(ours, &raw, elem_count, "random")?;

    ensure!(
        actual.len() == expected.len(),
        "length mismatch for {ours}: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        ensure!(
            a.to_bits() == e.to_bits(),
            "{ours} mismatch at {idx}: actual {a:?} ({:#010x}) expected {e:?} ({:#010x})",
            a.to_bits(),
            e.to_bits()
        );
    }
    Ok(())
}

#[test]
fn unsupported_gguf_type_errors_loudly() -> Result<()> {
    let path = std::env::temp_dir().join(format!(
        "grout_synthetic_q2_{}_{}.gguf",
        std::process::id(),
        unique_nanos()
    ));
    write_synthetic_q2_gguf(&path)?;

    let gguf = GgufFile::open(&path)?;
    let (info, data) = gguf
        .content
        .tensor_info("bad.weight")
        .and_then(|_| gguf.tensor_data("bad.weight"))?;
    let err = dequantize_to_f32(info.dtype, data, info.elem_count()?, "bad.weight")
        .expect_err("Q2_K must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("unsupported ggml type Q2_K") && msg.contains("bad.weight"),
        "unexpected error: {msg}"
    );

    let _ = fs::remove_file(path);
    Ok(())
}

fn write_synthetic_q2_gguf(path: &std::path::Path) -> Result<()> {
    let mut bytes = Vec::new();
    write_u32(&mut bytes, 0x4655_4747)?; // GGUF little-endian magic.
    write_u32(&mut bytes, 3)?; // version.
    write_u64(&mut bytes, 1)?; // tensor count.
    write_u64(&mut bytes, 0)?; // metadata count.

    write_string(&mut bytes, "bad.weight")?;
    write_u32(&mut bytes, 1)?; // dims.
    write_u64(&mut bytes, 256)?; // one Q2_K block.
    write_u32(&mut bytes, GgmlType::Q2K.to_u32())?;
    write_u64(&mut bytes, 0)?;

    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    bytes.resize(bytes.len() + GgmlType::Q2K.type_size(), 0);
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn write_string(mut w: impl Write, s: &str) -> Result<()> {
    write_u64(&mut w, s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn write_u32(mut w: impl Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn write_u64(mut w: impl Write, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn unique_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}
