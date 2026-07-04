use anyhow::{Result, ensure};
use candle_core::quantized::k_quants::{
    BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType as CandleGgmlType,
};
use cuda_async::device_operation::{DeviceOp, value};
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::core::f16;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::TileKernel;
use grout::dequant::{GgmlType, dequantize_to_f16, dequantize_to_f32};
use grout::kernels::{
    add_2d_f16, dequant_q4k_to_f16, dequant_q5k_to_f16, dequant_q6k_to_f16, dequant_q8_0_to_f16,
    gemv_q4k_f16, gemv_q5k_f16, gemv_q6k_f16, gemv_q8_0_f16,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

#[test]
#[ignore = "GPU kernel integration: run with `cargo test -- --ignored` and a visible CUDA device"]
fn add_2d_kernel_compiles_and_executes() -> Result<()> {
    if !cuda_available()? {
        return Ok(());
    }

    const BLOCK: usize = 4;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let lhs_host = Arc::new(vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ]);
    let rhs_host = Arc::new(vec![
        f16::from_f32(10.0),
        f16::from_f32(20.0),
        f16::from_f32(30.0),
        f16::from_f32(40.0),
    ]);

    let lhs = Arc::new(
        api::copy_host_vec_to_device(&lhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let rhs = Arc::new(
        api::copy_host_vec_to_device(&rhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let out = api::zeros::<f16>(&[1, BLOCK]).sync_on(&stream)?;

    let result = add_2d_f16(value(out.partition([1, BLOCK])), value(lhs), value(rhs))
        .generics(vec![BLOCK.to_string()])
        .sync_on(&stream)?;
    let out = result.0.unpartition();
    let actual = out.to_host_vec().sync_on(&stream)?;

    let actual: Vec<f32> = actual.into_iter().map(|x| x.to_f32()).collect();
    assert_eq!(actual, vec![11.0, 22.0, 33.0, 44.0]);
    Ok(())
}

#[test]
#[ignore = "GPU quantized GEMV integration: run with `cargo test gemv_q8_0_f16_matches_cpu -- --ignored` and a visible CUDA device"]
fn gemv_q8_0_f16_matches_cpu() -> Result<()> {
    if !cuda_available()? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for (rows, k, checked_rows) in quant_gemv_shapes() {
        run_q8_0_case(&stream, rows, k, checked_rows)?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU quantized GEMV integration: run with `cargo test gemv_q4k_f16_matches_cpu -- --ignored` and a visible CUDA device"]
fn gemv_q4k_f16_matches_cpu() -> Result<()> {
    if !cuda_available()? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for (rows, k, checked_rows) in quant_gemv_shapes() {
        run_q4k_case(&stream, rows, k, checked_rows)?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU quantized GEMV integration: run with `cargo test gemv_q6k_f16_matches_cpu -- --ignored` and a visible CUDA device"]
fn gemv_q6k_f16_matches_cpu() -> Result<()> {
    if !cuda_available()? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for (rows, k, checked_rows) in quant_gemv_shapes() {
        run_q6k_case(&stream, rows, k, checked_rows)?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU quantized GEMV integration: run with `cargo test gemv_q5k_f16_matches_cpu -- --ignored` and a visible CUDA device"]
fn gemv_q5k_f16_matches_cpu() -> Result<()> {
    if !cuda_available()? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for (rows, k, checked_rows) in quant_gemv_shapes() {
        run_q5k_case(&stream, rows, k, checked_rows)?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU quantized dequant integration: run with `cargo test dequant_q8_0_to_f16_matches_cpu -- --ignored` and a visible CUDA device"]
fn dequant_q8_0_to_f16_matches_cpu() -> Result<()> {
    if !cuda_available()? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for (rows, k, checked_rows) in dequant_prefill_shapes() {
        run_dequant_q8_0_case(&stream, rows, k, checked_rows)?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU quantized dequant integration: run with `cargo test dequant_q4k_to_f16_matches_cpu -- --ignored` and a visible CUDA device"]
fn dequant_q4k_to_f16_matches_cpu() -> Result<()> {
    if !cuda_available()? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for (rows, k, checked_rows) in dequant_prefill_shapes() {
        run_dequant_q4k_case(&stream, rows, k, checked_rows)?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU quantized dequant integration: run with `cargo test dequant_q6k_to_f16_matches_cpu -- --ignored` and a visible CUDA device"]
fn dequant_q6k_to_f16_matches_cpu() -> Result<()> {
    if !cuda_available()? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for (rows, k, checked_rows) in dequant_prefill_shapes() {
        run_dequant_q6k_case(&stream, rows, k, checked_rows)?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU quantized dequant integration: run with `cargo test dequant_q5k_to_f16_matches_cpu -- --ignored` and a visible CUDA device"]
fn dequant_q5k_to_f16_matches_cpu() -> Result<()> {
    if !cuda_available()? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for (rows, k, checked_rows) in dequant_prefill_shapes() {
        run_dequant_q5k_case(&stream, rows, k, checked_rows)?;
    }
    Ok(())
}

fn quant_gemv_shapes() -> [(usize, usize, Option<Vec<usize>>); 5] {
    [
        (3usize, 2560usize, None),
        (4, 4096, None),
        (2, 9728, None),
        (2, 12288, None),
        (151_936, 2560, Some(vec![0usize, 1, 777, 75_968, 151_935])),
    ]
}

fn dequant_prefill_shapes() -> [(usize, usize, Option<Vec<usize>>); 5] {
    [
        (3usize, 2560usize, None),
        (4, 4096, None),
        (2, 9728, None),
        (2, 12288, None),
        // Largest 8B transformer matrix scratch (~100 MB f16). LM head is intentionally absent.
        (12_288, 4096, Some(vec![0usize, 1, 777, 12_287])),
    ]
}

fn cuda_available() -> Result<bool> {
    match Device::device_count() {
        Ok(count) if count > 0 => Ok(true),
        Ok(_) => {
            eprintln!("skipping CUDA kernel integration test: no CUDA devices found");
            Ok(false)
        }
        Err(err) => {
            eprintln!("skipping CUDA kernel integration test: CUDA unavailable: {err:?}");
            Ok(false)
        }
    }
}

fn run_q8_0_case(
    stream: &Arc<cuda_core::Stream>,
    rows: usize,
    k: usize,
    checked_rows: Option<Vec<usize>>,
) -> Result<()> {
    let dtype = GgmlType::Q8_0;
    let raw = make_quantized_matrix::<BlockQ8_0>(dtype, rows, k, checked_rows.as_deref())?;
    let x = make_activation(k);
    let expected = expected_rows(dtype, &raw, rows, k, &x, checked_rows.as_deref())?;

    let weights_host = Arc::new(raw);
    let x_host = Arc::new(x);
    let weights = Arc::new(
        api::copy_host_vec_to_device(&weights_host)
            .reshape(&[weights_host.len()])
            .sync_on(stream)?,
    );
    let x_dev = Arc::new(
        api::copy_host_vec_to_device(&x_host)
            .reshape(&[k])
            .sync_on(stream)?,
    );
    let out = api::zeros::<f16>(&[rows]).sync_on(stream)?;

    let result = unsafe { gemv_q8_0_f16(value(out.partition([1])), value(weights), value(x_dev)) }
        .generics(vec![k.to_string()])
        .sync_on(stream)?;
    let out = result.0.unpartition();
    let actual = out.to_host_vec().sync_on(stream)?;

    for (row, expected) in expected {
        let actual = actual[row].to_f32();
        assert_close(row, actual, expected)?;
    }
    Ok(())
}

fn run_q4k_case(
    stream: &Arc<cuda_core::Stream>,
    rows: usize,
    k: usize,
    checked_rows: Option<Vec<usize>>,
) -> Result<()> {
    let dtype = GgmlType::Q4K;
    let raw = make_quantized_matrix::<BlockQ4K>(dtype, rows, k, checked_rows.as_deref())?;
    let x = make_activation(k);
    let expected = expected_rows(dtype, &raw, rows, k, &x, checked_rows.as_deref())?;

    let weights_host = Arc::new(raw);
    let x_host = Arc::new(x);
    let weights = Arc::new(
        api::copy_host_vec_to_device(&weights_host)
            .reshape(&[weights_host.len()])
            .sync_on(stream)?,
    );
    let x_dev = Arc::new(
        api::copy_host_vec_to_device(&x_host)
            .reshape(&[k])
            .sync_on(stream)?,
    );
    let out = api::zeros::<f16>(&[rows]).sync_on(stream)?;

    let result = unsafe { gemv_q4k_f16(value(out.partition([1])), value(weights), value(x_dev)) }
        .generics(vec![k.to_string()])
        .sync_on(stream)?;
    let out = result.0.unpartition();
    let actual = out.to_host_vec().sync_on(stream)?;

    for (row, expected) in expected {
        let actual = actual[row].to_f32();
        assert_close(row, actual, expected)?;
    }
    Ok(())
}

fn run_q6k_case(
    stream: &Arc<cuda_core::Stream>,
    rows: usize,
    k: usize,
    checked_rows: Option<Vec<usize>>,
) -> Result<()> {
    let dtype = GgmlType::Q6K;
    let raw = make_quantized_matrix::<BlockQ6K>(dtype, rows, k, checked_rows.as_deref())?;
    let x = make_activation(k);
    let expected = expected_rows(dtype, &raw, rows, k, &x, checked_rows.as_deref())?;

    let weights_host = Arc::new(raw);
    let x_host = Arc::new(x);
    let weights = Arc::new(
        api::copy_host_vec_to_device(&weights_host)
            .reshape(&[weights_host.len()])
            .sync_on(stream)?,
    );
    let x_dev = Arc::new(
        api::copy_host_vec_to_device(&x_host)
            .reshape(&[k])
            .sync_on(stream)?,
    );
    let out = api::zeros::<f16>(&[rows]).sync_on(stream)?;

    let result = unsafe { gemv_q6k_f16(value(out.partition([1])), value(weights), value(x_dev)) }
        .generics(vec![k.to_string()])
        .sync_on(stream)?;
    let out = result.0.unpartition();
    let actual = out.to_host_vec().sync_on(stream)?;

    for (row, expected) in expected {
        let actual = actual[row].to_f32();
        assert_close(row, actual, expected)?;
    }
    Ok(())
}

fn run_q5k_case(
    stream: &Arc<cuda_core::Stream>,
    rows: usize,
    k: usize,
    checked_rows: Option<Vec<usize>>,
) -> Result<()> {
    let dtype = GgmlType::Q5K;
    let raw = make_quantized_matrix::<BlockQ5K>(dtype, rows, k, checked_rows.as_deref())?;
    let x = make_activation(k);
    let expected = expected_rows(dtype, &raw, rows, k, &x, checked_rows.as_deref())?;

    let weights_host = Arc::new(raw);
    let x_host = Arc::new(x);
    let weights = Arc::new(
        api::copy_host_vec_to_device(&weights_host)
            .reshape(&[weights_host.len()])
            .sync_on(stream)?,
    );
    let x_dev = Arc::new(
        api::copy_host_vec_to_device(&x_host)
            .reshape(&[k])
            .sync_on(stream)?,
    );
    let out = api::zeros::<f16>(&[rows]).sync_on(stream)?;

    let result = unsafe { gemv_q5k_f16(value(out.partition([1])), value(weights), value(x_dev)) }
        .generics(vec![k.to_string()])
        .sync_on(stream)?;
    let out = result.0.unpartition();
    let actual = out.to_host_vec().sync_on(stream)?;

    for (row, expected) in expected {
        let actual = actual[row].to_f32();
        assert_close(row, actual, expected)?;
    }
    Ok(())
}

const MAX_TRANSFORMER_MATRIX_ELEMS_8B: usize = 12_288 * 4_096;

fn run_dequant_q8_0_case(
    stream: &Arc<cuda_core::Stream>,
    rows: usize,
    k: usize,
    checked_rows: Option<Vec<usize>>,
) -> Result<()> {
    let dtype = GgmlType::Q8_0;
    let tile_elems = 32usize;
    let raw = make_quantized_matrix::<BlockQ8_0>(dtype, rows, k, checked_rows.as_deref())?;
    let scratch_elems = scratch_elems_for(rows * k, tile_elems);
    let weights_host = Arc::new(raw.clone());
    let weights = Arc::new(
        api::copy_host_vec_to_device(&weights_host)
            .reshape(&[weights_host.len()])
            .sync_on(stream)?,
    );
    let scratch = api::zeros::<f16>(&[scratch_elems]).sync_on(stream)?;
    let num_tiles = (rows * k / tile_elems) as i32;

    let result = unsafe {
        dequant_q8_0_to_f16(
            value(scratch.partition([tile_elems])),
            value(weights),
            value(num_tiles),
        )
    }
    .sync_on(stream)?;
    let scratch = result.0.unpartition();
    let actual = scratch.to_host_vec().sync_on(stream)?;
    compare_dequant_rows(dtype, &raw, rows, k, &actual, checked_rows.as_deref())?;
    assert_scratch_tail_zero(rows * k, &actual)?;
    Ok(())
}

fn run_dequant_q4k_case(
    stream: &Arc<cuda_core::Stream>,
    rows: usize,
    k: usize,
    checked_rows: Option<Vec<usize>>,
) -> Result<()> {
    let dtype = GgmlType::Q4K;
    let tile_elems = 32usize;
    let raw = make_quantized_matrix::<BlockQ4K>(dtype, rows, k, checked_rows.as_deref())?;
    let scratch_elems = scratch_elems_for(rows * k, tile_elems);
    let weights_host = Arc::new(raw.clone());
    let weights = Arc::new(
        api::copy_host_vec_to_device(&weights_host)
            .reshape(&[weights_host.len()])
            .sync_on(stream)?,
    );
    let scratch = api::zeros::<f16>(&[scratch_elems]).sync_on(stream)?;
    let num_tiles = (rows * k / tile_elems) as i32;

    let result = unsafe {
        dequant_q4k_to_f16(
            value(scratch.partition([tile_elems])),
            value(weights),
            value(num_tiles),
        )
    }
    .sync_on(stream)?;
    let scratch = result.0.unpartition();
    let actual = scratch.to_host_vec().sync_on(stream)?;
    compare_dequant_rows(dtype, &raw, rows, k, &actual, checked_rows.as_deref())?;
    assert_scratch_tail_zero(rows * k, &actual)?;
    Ok(())
}

fn run_dequant_q6k_case(
    stream: &Arc<cuda_core::Stream>,
    rows: usize,
    k: usize,
    checked_rows: Option<Vec<usize>>,
) -> Result<()> {
    let dtype = GgmlType::Q6K;
    let tile_elems = 16usize;
    let raw = make_quantized_matrix::<BlockQ6K>(dtype, rows, k, checked_rows.as_deref())?;
    let scratch_elems = scratch_elems_for(rows * k, tile_elems);
    let weights_host = Arc::new(raw.clone());
    let weights = Arc::new(
        api::copy_host_vec_to_device(&weights_host)
            .reshape(&[weights_host.len()])
            .sync_on(stream)?,
    );
    let scratch = api::zeros::<f16>(&[scratch_elems]).sync_on(stream)?;
    let num_tiles = (rows * k / tile_elems) as i32;

    let result = unsafe {
        dequant_q6k_to_f16(
            value(scratch.partition([tile_elems])),
            value(weights),
            value(num_tiles),
        )
    }
    .sync_on(stream)?;
    let scratch = result.0.unpartition();
    let actual = scratch.to_host_vec().sync_on(stream)?;
    compare_dequant_rows(dtype, &raw, rows, k, &actual, checked_rows.as_deref())?;
    assert_scratch_tail_zero(rows * k, &actual)?;
    Ok(())
}

fn run_dequant_q5k_case(
    stream: &Arc<cuda_core::Stream>,
    rows: usize,
    k: usize,
    checked_rows: Option<Vec<usize>>,
) -> Result<()> {
    let dtype = GgmlType::Q5K;
    let tile_elems = 32usize;
    let raw = make_quantized_matrix::<BlockQ5K>(dtype, rows, k, checked_rows.as_deref())?;
    let scratch_elems = scratch_elems_for(rows * k, tile_elems);
    let weights_host = Arc::new(raw.clone());
    let weights = Arc::new(
        api::copy_host_vec_to_device(&weights_host)
            .reshape(&[weights_host.len()])
            .sync_on(stream)?,
    );
    let scratch = api::zeros::<f16>(&[scratch_elems]).sync_on(stream)?;
    let num_tiles = (rows * k / tile_elems) as i32;

    let result = unsafe {
        dequant_q5k_to_f16(
            value(scratch.partition([tile_elems])),
            value(weights),
            value(num_tiles),
        )
    }
    .sync_on(stream)?;
    let scratch = result.0.unpartition();
    let actual = scratch.to_host_vec().sync_on(stream)?;
    compare_dequant_rows(dtype, &raw, rows, k, &actual, checked_rows.as_deref())?;
    assert_scratch_tail_zero(rows * k, &actual)?;
    Ok(())
}

fn scratch_elems_for(matrix_elems: usize, tile_elems: usize) -> usize {
    if matrix_elems == MAX_TRANSFORMER_MATRIX_ELEMS_8B {
        return MAX_TRANSFORMER_MATRIX_ELEMS_8B;
    }
    let with_tail = matrix_elems + tile_elems;
    with_tail.div_ceil(tile_elems) * tile_elems
}

fn compare_dequant_rows(
    dtype: GgmlType,
    raw: &[u8],
    rows: usize,
    k: usize,
    actual: &[f16],
    checked_rows: Option<&[usize]>,
) -> Result<()> {
    let row_bytes = k / dtype.block_size() * dtype.type_size();
    let rows_to_check: Vec<usize> = match checked_rows {
        Some(rows) => rows.to_vec(),
        None => (0..rows).collect(),
    };
    for row in rows_to_check {
        let raw_start = row * row_bytes;
        let expected =
            dequantize_to_f16(dtype, &raw[raw_start..raw_start + row_bytes], k, "test row")?;
        let out_start = row * k;
        for (col, (actual, expected)) in actual[out_start..out_start + k]
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            let actual = actual.to_f32();
            let expected = expected.to_f32();
            let tol = 1.0e-2f32 * expected.abs().max(1.0);
            ensure!(
                (actual - expected).abs() <= tol,
                "dequant row {row} col {col}: actual {actual} expected {expected} tolerance {tol}"
            );
        }
    }
    Ok(())
}

fn assert_scratch_tail_zero(matrix_elems: usize, actual: &[f16]) -> Result<()> {
    if matrix_elems < actual.len() {
        let value = actual[matrix_elems].to_f32();
        ensure!(
            value == 0.0,
            "dequant kernel wrote past matrix prefix: tail value {value} at {matrix_elems}"
        );
    }
    Ok(())
}

fn make_activation(k: usize) -> Vec<f16> {
    (0..k)
        .map(|i| f16::from_f32(((i % 251) as f32 - 125.0) / 251.0))
        .collect()
}

fn make_quantized_matrix<T: CandleGgmlType>(
    dtype: GgmlType,
    rows: usize,
    k: usize,
    checked_rows: Option<&[usize]>,
) -> Result<Vec<u8>> {
    ensure!(
        k.is_multiple_of(dtype.block_size()),
        "k must tile quant blocks"
    );
    let row_bytes = k / dtype.block_size() * dtype.type_size();
    let mut raw = vec![0u8; rows * row_bytes];
    let rows_to_randomize: Vec<usize> = match checked_rows {
        Some(rows) => rows.to_vec(),
        None => (0..rows).collect(),
    };
    for &row in &rows_to_randomize {
        ensure!(row < rows, "checked row {row} out of range for {rows} rows");
        let dense = make_dense_row(row, k);
        let row_raw = quantize_row::<T>(&dense);
        raw[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(&row_raw);
    }
    Ok(raw)
}

fn make_dense_row(row: usize, k: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(0x4752_4f55_5451_3800u64 ^ row as u64 ^ ((k as u64) << 32));
    (0..k).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect()
}

fn quantize_row<T: CandleGgmlType>(dense: &[f32]) -> Vec<u8> {
    let mut blocks = vec![T::zeros(); dense.len() / T::BLCK_SIZE];
    T::from_float(dense, &mut blocks);
    let byte_len = std::mem::size_of_val(blocks.as_slice());
    let bytes = unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
    bytes.to_vec()
}

fn expected_rows(
    dtype: GgmlType,
    raw: &[u8],
    rows: usize,
    k: usize,
    x: &[f16],
    checked_rows: Option<&[usize]>,
) -> Result<Vec<(usize, f32)>> {
    let row_bytes = k / dtype.block_size() * dtype.type_size();
    let rows_to_check: Vec<usize> = match checked_rows {
        Some(rows) => rows.to_vec(),
        None => (0..rows).collect(),
    };
    let x: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let mut out = Vec::with_capacity(rows_to_check.len());
    for row in rows_to_check {
        let start = row * row_bytes;
        let dense = dequantize_to_f32(dtype, &raw[start..start + row_bytes], k, "test row")?;
        let expected = dense.iter().zip(&x).map(|(w, x)| w * x).sum::<f32>();
        out.push((row, expected));
    }
    Ok(out)
}

fn assert_close(row: usize, actual: f32, expected: f32) -> Result<()> {
    let tol = 1.0e-2f32 * expected.abs().max(1.0);
    ensure!(
        (actual - expected).abs() <= tol,
        "row {row}: actual {actual} expected {expected} tolerance {tol}"
    );
    Ok(())
}
