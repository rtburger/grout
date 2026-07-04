use anyhow::{Result, ensure};
use clap::Parser;
use cuda_async::device_operation::{DeviceOp, ExecutionContext, value};
use cuda_core::{Device, IntoResult, Stream, sys as cu_sys};
use cutile::api::{self, DeviceOpReshape};
use cutile::core::f16;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::{CompileOptions, TileKernel};
use grout::kernels::gemv_q8_0_soa_f16;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Synthetic Q8_0 SoA tiled/TMA GEMV microbench"
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

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    rows: usize,
    k: usize,
}

const SHAPES: [Shape; 5] = [
    Shape {
        label: "attn_q_o",
        rows: 4096,
        k: 4096,
    },
    Shape {
        label: "attn_k_v",
        rows: 1024,
        k: 4096,
    },
    Shape {
        label: "ffn_gate_up",
        rows: 12288,
        k: 4096,
    },
    Shape {
        label: "ffn_down",
        rows: 4096,
        k: 12288,
    },
    Shape {
        label: "lm_head",
        rows: 151936,
        k: 4096,
    },
];

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
            "backend,label,rows,k,r,bk,block_scales,latency,occupancy,weight_bytes,activation_bytes,output_bytes,total_bytes,iters,total_ms,avg_us,achieved_gbps"
        );
    }

    for occupancy in args.occupancies {
        ensure!(occupancy > 0, "occupancy values must be positive");
        for shape in SHAPES {
            run_shape(
                &ctx,
                &stream,
                shape,
                occupancy,
                args.warmup_iters,
                args.iters,
            )?;
        }
    }
    Ok(())
}

fn run_shape(
    ctx: &ExecutionContext,
    stream: &Arc<Stream>,
    shape: Shape,
    occupancy: i32,
    warmup_iters: usize,
    iters: usize,
) -> Result<()> {
    ensure!(
        shape.rows.is_multiple_of(8),
        "rows must be divisible by R=8"
    );
    ensure!(shape.k.is_multiple_of(512), "K must be divisible by BK=512");
    ensure!(
        shape.k.is_multiple_of(32),
        "K must be divisible by Q8_0 block size"
    );

    let (qs, scales) = synthetic_q8_0_soa(shape.rows, shape.k);
    let qs = Arc::new(qs);
    let scales = Arc::new(scales);
    let activation: Arc<Vec<f16>> = Arc::new(
        (0..shape.k)
            .map(|i| f16::from_f32(((i % 251) as f32 - 125.0) / 125.0))
            .collect(),
    );

    let qs_dev = Arc::new(
        api::copy_host_vec_to_device(&qs)
            .reshape(&[shape.rows, shape.k])
            .sync_on(stream)?,
    );
    let scales_dev = Arc::new(
        api::copy_host_vec_to_device(&scales)
            .reshape(&[shape.rows, shape.k / 32])
            .sync_on(stream)?,
    );
    let activation_dev = Arc::new(
        api::copy_host_vec_to_device(&activation)
            .reshape(&[shape.k])
            .sync_on(stream)?,
    );
    let mut out = api::zeros::<f16>(&[shape.rows]).sync_on(stream)?;

    for _ in 0..warmup_iters {
        let result = unsafe {
            gemv_q8_0_soa_f16(
                value(out.partition([8])),
                value(qs_dev.clone()),
                value(scales_dev.clone()),
                value(activation_dev.clone()),
                value(shape.rows as i32),
            )
        }
        .generics(vec![
            shape.k.to_string(),
            (shape.k / 32).to_string(),
            "1".to_string(),
        ])
        .grid(((shape.rows / 8) as u32, 1, 1))
        .compile_options(CompileOptions::default().occupancy(occupancy))
        .sync_on(stream)?;
        out = result.0.unpartition();
    }

    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    start.record(stream)?;
    for _ in 0..iters {
        let result = unsafe {
            gemv_q8_0_soa_f16(
                value(out.partition([8])),
                value(qs_dev.clone()),
                value(scales_dev.clone()),
                value(activation_dev.clone()),
                value(shape.rows as i32),
            )
        }
        .generics(vec![
            shape.k.to_string(),
            (shape.k / 32).to_string(),
            "1".to_string(),
        ])
        .grid(((shape.rows / 8) as u32, 1, 1))
        .compile_options(CompileOptions::default().occupancy(occupancy));
        out = unsafe { result.execute(ctx)? }.0.unpartition();
    }
    end.record(stream)?;
    end.synchronize()?;
    let total_ms = end.elapsed_ms_since(&start)?;

    let weight_bytes =
        shape.rows * shape.k + shape.rows * (shape.k / 32) * std::mem::size_of::<f16>();
    let activation_bytes = shape.k * std::mem::size_of::<f16>();
    let output_bytes = shape.rows * std::mem::size_of::<f16>();
    let total_bytes = weight_bytes + activation_bytes + output_bytes;
    let avg_us = total_ms as f64 * 1000.0 / iters as f64;
    let achieved_gbps = total_bytes as f64 * iters as f64 / (total_ms as f64 / 1000.0) / 1.0e9;
    println!(
        "q8_0_soa_tile,{},{},{},8,512,16,1,{},{},{},{},{},{},{:.6},{:.3},{:.3}",
        shape.label,
        shape.rows,
        shape.k,
        occupancy,
        weight_bytes,
        activation_bytes,
        output_bytes,
        total_bytes,
        iters,
        total_ms,
        avg_us,
        achieved_gbps,
    );
    Ok(())
}

fn synthetic_q8_0_soa(rows: usize, k: usize) -> (Vec<i8>, Vec<f16>) {
    let blocks_per_row = k / 32;
    let mut qs = vec![0i8; rows * k];
    let scales = vec![f16::from_f32(1.0); rows * blocks_per_row];
    for row in 0..rows {
        for col in 0..k {
            qs[row * k + col] = (((row + (col / 32) + (col & 31)) % 127) as i8) - 63;
        }
    }
    (qs, scales)
}
