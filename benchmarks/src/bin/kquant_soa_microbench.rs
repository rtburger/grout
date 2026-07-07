use anyhow::{Result, ensure};
use clap::Parser;
use cuda_async::device_operation::{DeviceOp, ExecutionContext, value};
use cuda_core::{Device, IntoResult, Stream, sys as cu_sys};
use cutile::api::{self, DeviceOpReshape};
use cutile::core::f16;
use cutile::tensor::{IntoPartition, Tensor};
use cutile::tile_kernel::{CompileOptions, TileKernel};
use grout::kernels::{gemv_q4k_soa_f16, gemv_q6k_soa_f16};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Synthetic Q4K/Q6K SoA tiled/TMA GEMV microbench"
)]
struct Args {
    /// Timed launches per shape/occupancy.
    #[arg(long, default_value_t = 20)]
    iters: usize,

    /// Untimed launches before timing.
    #[arg(long, default_value_t = 5)]
    warmup_iters: usize,

    /// Occupancy compile-options to sweep, comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4")]
    occupancies: Vec<i32>,

    /// Print only CSV rows, without header.
    #[arg(long)]
    no_header: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Dtype {
    Q4K,
    Q6K,
}

#[derive(Clone, Copy)]
struct Shape {
    dtype: Dtype,
    label: &'static str,
    rows: usize,
    k: usize,
}

// Exact Qwen3-4B and Qwen3-8B Q4_K_M decode GEMV shapes per dtype
// (per-tensor dtypes from the Phase 2 Task 1 GGUF tensor tables).
const SHAPES: [Shape; 14] = [
    // Qwen3-4B (hidden 2560, ffn 9728, vocab 151936)
    Shape { dtype: Dtype::Q4K, label: "4b_attn_q", rows: 4096, k: 2560 },
    Shape { dtype: Dtype::Q4K, label: "4b_attn_k_v", rows: 1024, k: 2560 },
    Shape { dtype: Dtype::Q4K, label: "4b_attn_output", rows: 2560, k: 4096 },
    Shape { dtype: Dtype::Q4K, label: "4b_ffn_gate_up", rows: 9728, k: 2560 },
    Shape { dtype: Dtype::Q6K, label: "4b_attn_v_q6k", rows: 1024, k: 2560 },
    Shape { dtype: Dtype::Q6K, label: "4b_ffn_down", rows: 2560, k: 9728 },
    Shape { dtype: Dtype::Q6K, label: "4b_lm_head", rows: 151_936, k: 2560 },
    // Qwen3-8B (hidden 4096, ffn 12288)
    Shape { dtype: Dtype::Q4K, label: "8b_attn_q_o", rows: 4096, k: 4096 },
    Shape { dtype: Dtype::Q4K, label: "8b_attn_k_v", rows: 1024, k: 4096 },
    Shape { dtype: Dtype::Q4K, label: "8b_ffn_gate_up", rows: 12_288, k: 4096 },
    Shape { dtype: Dtype::Q4K, label: "8b_ffn_down_q4k", rows: 4096, k: 12_288 },
    Shape { dtype: Dtype::Q6K, label: "8b_attn_v_q6k", rows: 1024, k: 4096 },
    Shape { dtype: Dtype::Q6K, label: "8b_ffn_down", rows: 4096, k: 12_288 },
    Shape { dtype: Dtype::Q6K, label: "8b_lm_head", rows: 151_936, k: 4096 },
];

// Rotate independent weight copies so repeated iterations stream from DRAM
// instead of re-hitting a 36 MB L2-resident working set (the artifact noted
// on the Q8_0 SoA checkpoint rows).
const L2_DEFEAT_BYTES: usize = 64 * 1024 * 1024;
const MAX_COPIES: usize = 8;

struct CudaEvent {
    event: cu_sys::CUevent,
}

impl CudaEvent {
    fn new() -> Result<Self> {
        let mut event = std::mem::MaybeUninit::<cu_sys::CUevent>::uninit();
        unsafe {
            cu_sys::cuEventCreate(
                event.as_mut_ptr(),
                cu_sys::CUevent_flags_enum_CU_EVENT_DEFAULT,
            )
            .result()
            .map_err(|e| anyhow::anyhow!("cuEventCreate failed: {e:?}"))?;
            Ok(Self {
                event: event.assume_init(),
            })
        }
    }

    fn record(&self, stream: &Arc<Stream>) -> Result<()> {
        unsafe {
            cu_sys::cuEventRecord(self.event, stream.cu_stream())
                .result()
                .map_err(|e| anyhow::anyhow!("cuEventRecord failed: {e:?}"))
        }
    }

    fn synchronize(&self) -> Result<()> {
        unsafe {
            cu_sys::cuEventSynchronize(self.event)
                .result()
                .map_err(|e| anyhow::anyhow!("cuEventSynchronize failed: {e:?}"))
        }
    }

    fn elapsed_ms_since(&self, start: &CudaEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        unsafe {
            cu_sys::cuEventElapsedTime_v2(&mut ms, start.event, self.event)
                .result()
                .map_err(|e| anyhow::anyhow!("cuEventElapsedTime_v2 failed: {e:?}"))?;
        }
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = cu_sys::cuEventDestroy_v2(self.event);
        }
    }
}

struct WeightCopy {
    qs_i8: Option<Arc<Tensor<i8>>>,
    qs_u8: Option<Arc<Tensor<u8>>>,
    sc_i8: Option<Arc<Tensor<i8>>>,
    sc_f16: Option<Arc<Tensor<f16>>>,
    mins_f16: Option<Arc<Tensor<f16>>>,
    d: Option<Arc<Tensor<f16>>>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.iters > 0, "--iters must be positive");
    ensure!(args.warmup_iters > 0, "--warmup-iters must be positive");
    ensure!(
        !args.occupancies.is_empty(),
        "--occupancies must not be empty"
    );

    let device =
        Device::new(0).map_err(|e| anyhow::anyhow!("failed to open CUDA device 0: {e:?}"))?;
    device
        .bind_to_thread()
        .map_err(|e| anyhow::anyhow!("failed to bind CUDA context: {e:?}"))?;
    let stream = device
        .new_stream()
        .map_err(|e| anyhow::anyhow!("failed to create CUDA stream: {e:?}"))?;
    let ctx = ExecutionContext::new(stream.clone());

    if !args.no_header {
        println!(
            "backend,label,rows,k,copies,occupancy,weight_bytes,total_bytes,iters,total_ms,avg_us,achieved_gbps"
        );
    }

    for occupancy in &args.occupancies {
        ensure!(*occupancy > 0, "occupancy values must be positive");
        for shape in SHAPES {
            run_shape(
                &ctx,
                &stream,
                shape,
                *occupancy,
                args.warmup_iters,
                args.iters,
            )?;
        }
    }
    Ok(())
}

fn weight_bytes(shape: Shape) -> usize {
    let elems = shape.rows * shape.k;
    match shape.dtype {
        Dtype::Q4K => elems / 2 + elems / 32 * 4,
        Dtype::Q6K => elems + elems / 16 + elems / 256 * 2,
    }
}

fn upload_copy(
    stream: &Arc<Stream>,
    shape: Shape,
    salt: usize,
) -> Result<WeightCopy> {
    let rows = shape.rows;
    let k = shape.k;
    match shape.dtype {
        Dtype::Q6K => {
            let mut qs = vec![0i8; rows * k];
            for (i, v) in qs.iter_mut().enumerate() {
                *v = (((i + salt) % 63) as i8) - 31;
            }
            let mut sc = vec![0i8; rows * k / 16];
            for (i, v) in sc.iter_mut().enumerate() {
                *v = (((i + salt) % 63) as i8) - 31;
            }
            let d = vec![f16::from_f32(0.01); rows * k / 256];
            Ok(WeightCopy {
                qs_i8: Some(to_dev(stream, qs, rows, k)?),
                qs_u8: None,
                sc_i8: Some(to_dev(stream, sc, rows, k / 16)?),
                sc_f16: None,
                mins_f16: None,
                d: Some(to_dev(stream, d, rows, k / 256)?),
            })
        }
        Dtype::Q4K => {
            let mut qs = vec![0u8; rows * k / 2];
            for (i, v) in qs.iter_mut().enumerate() {
                *v = ((i + salt) % 251) as u8;
            }
            let mut sc = vec![f16::from_f32(0.0); rows * k / 32];
            let mut mins = vec![f16::from_f32(0.0); rows * k / 32];
            for i in 0..sc.len() {
                sc[i] = f16::from_f32(0.01 * (((i + salt) % 64) as f32));
                mins[i] = f16::from_f32(0.001 * (((i + salt * 3) % 64) as f32));
            }
            Ok(WeightCopy {
                qs_i8: None,
                qs_u8: Some(to_dev(stream, qs, rows, k / 2)?),
                sc_i8: None,
                sc_f16: Some(to_dev(stream, sc, rows, k / 32)?),
                mins_f16: Some(to_dev(stream, mins, rows, k / 32)?),
                d: None,
            })
        }
    }
}

fn to_dev<T: cutile::DType>(
    stream: &Arc<Stream>,
    host: Vec<T>,
    rows: usize,
    cols: usize,
) -> Result<Arc<Tensor<T>>> {
    let host = Arc::new(host);
    Ok(Arc::new(
        api::copy_host_vec_to_device(&host)
            .reshape(&[rows, cols])
            .sync_on(stream)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn launch(
    ctx: &ExecutionContext,
    stream: &Arc<Stream>,
    shape: Shape,
    copy: &WeightCopy,
    activation: &Arc<Tensor<f16>>,
    out: Tensor<f16>,
    occupancy: i32,
    timed: bool,
) -> Result<Tensor<f16>> {
    let rows = shape.rows;
    let k = shape.k;
    let opts = CompileOptions::default().occupancy(occupancy);
    match shape.dtype {
        Dtype::Q6K => {
            let grid = ((rows / 8) as u32, 1u32, 1u32);
            let op = unsafe {
                gemv_q6k_soa_f16(
                    value(out.partition([8])),
                    value(copy.qs_i8.clone().unwrap()),
                    value(copy.sc_i8.clone().unwrap()),
                    value(copy.d.clone().unwrap()),
                    value(activation.clone()),
                    value(rows as i32),
                )
            }
            .generics(vec![
                k.to_string(),
                (k / 16).to_string(),
                (k / 256).to_string(),
                "1".to_string(),
            ])
            .grid(grid)
            .compile_options(opts);
            let result = if timed {
                unsafe { op.execute(ctx)? }
            } else {
                op.sync_on(stream)?
            };
            Ok(result.0.unpartition())
        }
        Dtype::Q4K => {
            let grid = ((rows / 16) as u32, 1u32, 1u32);
            let op = unsafe {
                gemv_q4k_soa_f16(
                    value(out.partition([16])),
                    value(copy.qs_u8.clone().unwrap()),
                    value(copy.sc_f16.clone().unwrap()),
                    value(copy.mins_f16.clone().unwrap()),
                    value(activation.clone()),
                    value(rows as i32),
                )
            }
            .generics(vec![
                (k / 2).to_string(),
                (k / 32).to_string(),
                "1".to_string(),
            ])
            .grid(grid)
            .compile_options(opts);
            let result = if timed {
                unsafe { op.execute(ctx)? }
            } else {
                op.sync_on(stream)?
            };
            Ok(result.0.unpartition())
        }
    }
}

fn run_shape(
    ctx: &ExecutionContext,
    stream: &Arc<Stream>,
    shape: Shape,
    occupancy: i32,
    warmup_iters: usize,
    iters: usize,
) -> Result<()> {
    ensure!(shape.rows.is_multiple_of(16), "rows must be divisible by 16");
    ensure!(shape.k.is_multiple_of(512), "K must be divisible by 512");

    let w_bytes = weight_bytes(shape);
    let copies = (L2_DEFEAT_BYTES / w_bytes).clamp(1, MAX_COPIES).max(1);
    let weights: Vec<WeightCopy> = (0..copies)
        .map(|i| upload_copy(stream, shape, i * 7919))
        .collect::<Result<Vec<_>>>()?;

    let activation: Arc<Vec<f16>> = Arc::new(
        (0..shape.k)
            .map(|i| f16::from_f32(((i % 251) as f32 - 125.0) / 125.0))
            .collect(),
    );
    let activation = Arc::new(
        api::copy_host_vec_to_device(&activation)
            .reshape(&[shape.k])
            .sync_on(stream)?,
    );
    let mut out = api::zeros::<f16>(&[shape.rows]).sync_on(stream)?;

    for i in 0..warmup_iters {
        out = launch(
            ctx,
            stream,
            shape,
            &weights[i % copies],
            &activation,
            out,
            occupancy,
            false,
        )?;
    }

    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    start.record(stream)?;
    for i in 0..iters {
        out = launch(
            ctx,
            stream,
            shape,
            &weights[i % copies],
            &activation,
            out,
            occupancy,
            true,
        )?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let total_ms = end.elapsed_ms_since(&start)?;

    let activation_bytes = shape.k * std::mem::size_of::<f16>();
    let output_bytes = shape.rows * std::mem::size_of::<f16>();
    let total_bytes = w_bytes + activation_bytes + output_bytes;
    let avg_us = total_ms as f64 * 1000.0 / iters as f64;
    let achieved_gbps = total_bytes as f64 * iters as f64 / (total_ms as f64 / 1000.0) / 1.0e9;
    let backend = match shape.dtype {
        Dtype::Q4K => "q4k_soa_tile",
        Dtype::Q6K => "q6k_soa_tile",
    };
    println!(
        "{backend},{},{},{},{},{},{},{},{},{:.6},{:.3},{:.3}",
        shape.label,
        shape.rows,
        shape.k,
        copies,
        occupancy,
        w_bytes,
        total_bytes,
        iters,
        total_ms,
        avg_us,
        achieved_gbps,
    );
    Ok(())
}
