use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use cuda_core::{
    Device, Function, IntoResult, Stream, free_async, launch_kernel, malloc_async,
    memcpy_htod_async, sys as cu_sys,
};
use cutile::core::f16;
use grout::dequant::GgmlType;
use grout::gguf::{GgufFile, TensorInfo};
use std::collections::HashSet;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Standalone quantized GEMV microbench for GGUF tensors"
)]
struct Args {
    /// GGUF files to inspect and benchmark. Pass both Qwen3-4B and Qwen3-8B.
    #[arg(long = "gguf", required = true)]
    ggufs: Vec<PathBuf>,

    /// Timed GEMV launches per tensor.
    #[arg(long, default_value_t = 20)]
    iters: usize,

    /// Untimed launches per tensor before timing.
    #[arg(long, default_value_t = 5)]
    warmup_iters: usize,

    /// CUDA architecture for nvcc, e.g. sm_89.
    #[arg(long, default_value = "sm_89")]
    arch: String,

    /// Suppress CSV header.
    #[arg(long)]
    no_header: bool,
}

#[derive(Clone, Debug)]
struct BenchTensor {
    model: String,
    tensor_kind: String,
    tensor_name: String,
    dtype: GgmlType,
    rows: usize,
    k: usize,
    weight_bytes: usize,
}

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

    fn record(&self, stream: &Stream) -> Result<()> {
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

    let cubin = compile_kernels(&args.arch)?;
    let device =
        Device::new(0).map_err(|e| anyhow::anyhow!("failed to open CUDA device 0: {e:?}"))?;
    device
        .bind_to_thread()
        .map_err(|e| anyhow::anyhow!("failed to bind CUDA context: {e:?}"))?;
    let stream = device
        .new_stream()
        .map_err(|e| anyhow::anyhow!("failed to create CUDA stream: {e:?}"))?;
    let module = device
        .load_module_from_file(cubin.to_string_lossy().as_ref())
        .map_err(|e| anyhow::anyhow!("failed to load {}: {e:?}", cubin.display()))?;

    if !args.no_header {
        println!(
            "model,tensor_kind,tensor_name,dtype,rows,k,weight_bytes,activation_bytes,output_bytes,total_bytes,iters,total_ms,avg_us,achieved_gbps"
        );
    }

    for path in &args.ggufs {
        let gguf = GgufFile::open(path)?;
        let tensors = discover_bench_tensors(path, &gguf)?;
        for tensor in tensors {
            let (_, raw) = gguf.tensor_data(&tensor.tensor_name)?;
            let function = load_function_for_dtype(&module, tensor.dtype)?;
            let total_ms = run_one(
                &stream,
                &function,
                &tensor,
                raw,
                args.warmup_iters,
                args.iters,
            )?;
            let activation_bytes = tensor.k * std::mem::size_of::<f16>();
            let output_bytes = tensor.rows * std::mem::size_of::<f16>();
            let total_bytes = tensor.weight_bytes + activation_bytes + output_bytes;
            let avg_us = total_ms as f64 * 1000.0 / args.iters as f64;
            let achieved_gbps =
                total_bytes as f64 * args.iters as f64 / (total_ms as f64 / 1000.0) / 1.0e9;
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.3},{:.3}",
                tensor.model,
                tensor.tensor_kind,
                tensor.tensor_name,
                tensor.dtype,
                tensor.rows,
                tensor.k,
                tensor.weight_bytes,
                activation_bytes,
                output_bytes,
                total_bytes,
                args.iters,
                total_ms,
                avg_us,
                achieved_gbps,
            );
        }
    }

    Ok(())
}

fn load_function_for_dtype(module: &Arc<cuda_core::Module>, dtype: GgmlType) -> Result<Function> {
    let name = match dtype {
        GgmlType::Q4K => "q4k_gemv",
        GgmlType::Q5K => "q5k_gemv",
        GgmlType::Q6K => "q6k_gemv",
        GgmlType::Q8_0 => "q8_0_gemv",
        other => bail!("no quantized GEMV kernel for dtype {other}"),
    };
    module
        .load_function(name)
        .map_err(|e| anyhow::anyhow!("failed to load kernel {name}: {e:?}"))
}

fn run_one(
    stream: &Arc<Stream>,
    function: &Function,
    tensor: &BenchTensor,
    raw: &[u8],
    warmup_iters: usize,
    iters: usize,
) -> Result<f32> {
    ensure!(
        raw.len() == tensor.weight_bytes,
        "raw tensor byte length mismatch"
    );
    let activation: Vec<f16> = (0..tensor.k)
        .map(|i| f16::from_f32(((i % 251) as f32 - 125.0) / 125.0))
        .collect();

    let weight_ptr = unsafe { malloc_async(tensor.weight_bytes, stream) };
    let activation_ptr = unsafe { malloc_async(tensor.k * std::mem::size_of::<f16>(), stream) };
    let output_ptr = unsafe { malloc_async(tensor.rows * std::mem::size_of::<f16>(), stream) };

    unsafe {
        memcpy_htod_async::<u8>(weight_ptr, raw.as_ptr(), raw.len(), stream);
        memcpy_htod_async::<f16>(
            activation_ptr,
            activation.as_ptr(),
            activation.len(),
            stream,
        );
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("initial copy synchronize failed: {e:?}"))?;
    }

    let rows = i32::try_from(tensor.rows).context("rows does not fit i32")?;
    let k = i32::try_from(tensor.k).context("k does not fit i32")?;
    let row_stride_bytes =
        i32::try_from(tensor.weight_bytes / tensor.rows).context("row stride does not fit i32")?;

    for _ in 0..warmup_iters {
        launch_quant_gemv(
            stream,
            function,
            weight_ptr,
            activation_ptr,
            output_ptr,
            rows,
            k,
            row_stride_bytes,
        )?;
    }
    unsafe {
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("warmup synchronize failed: {e:?}"))?;
    }

    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    start.record(stream)?;
    for _ in 0..iters {
        launch_quant_gemv(
            stream,
            function,
            weight_ptr,
            activation_ptr,
            output_ptr,
            rows,
            k,
            row_stride_bytes,
        )?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let total_ms = end.elapsed_ms_since(&start)?;

    unsafe {
        free_async(weight_ptr, stream);
        free_async(activation_ptr, stream);
        free_async(output_ptr, stream);
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("free synchronize failed: {e:?}"))?;
    }

    Ok(total_ms)
}

#[allow(clippy::too_many_arguments)]
fn launch_quant_gemv(
    stream: &Arc<Stream>,
    function: &Function,
    weight_ptr: cu_sys::CUdeviceptr,
    activation_ptr: cu_sys::CUdeviceptr,
    output_ptr: cu_sys::CUdeviceptr,
    rows: i32,
    k: i32,
    row_stride_bytes: i32,
) -> Result<()> {
    let mut weight = weight_ptr;
    let mut activation = activation_ptr;
    let mut output = output_ptr;
    let mut rows_arg = rows;
    let mut k_arg = k;
    let mut stride_arg = row_stride_bytes;
    let mut params: [*mut c_void; 6] = [
        &mut weight as *mut _ as *mut c_void,
        &mut activation as *mut _ as *mut c_void,
        &mut output as *mut _ as *mut c_void,
        &mut rows_arg as *mut _ as *mut c_void,
        &mut k_arg as *mut _ as *mut c_void,
        &mut stride_arg as *mut _ as *mut c_void,
    ];
    unsafe {
        launch_kernel(
            function.cu_function(),
            (rows as u32, 1, 1),
            (256, 1, 1),
            256 * std::mem::size_of::<f32>() as u32,
            stream.cu_stream(),
            &mut params,
        )
        .map_err(|e| anyhow::anyhow!("quant GEMV launch failed: {e:?}"))
    }
}

fn discover_bench_tensors(path: &Path, gguf: &GgufFile) -> Result<Vec<BenchTensor>> {
    let model = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("gguf")
        .to_string();
    let block_count = gguf
        .content
        .metadata_required("qwen3.block_count")?
        .to_u32()? as usize;

    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let projections = [
        ("attn_q", "attn_q.weight"),
        ("attn_k", "attn_k.weight"),
        ("attn_v", "attn_v.weight"),
        ("attn_output", "attn_output.weight"),
        ("ffn_gate", "ffn_gate.weight"),
        ("ffn_up", "ffn_up.weight"),
        ("ffn_down", "ffn_down.weight"),
    ];

    for layer in 0..block_count {
        for (kind, suffix) in projections {
            let name = format!("blk.{layer}.{suffix}");
            if let Some(info) = gguf.content.tensor_infos.get(&name) {
                push_unique(&model, kind, info, &mut seen, &mut out)?;
            }
        }
    }

    let lm_name = if gguf.content.has_tensor("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let lm = gguf.content.tensor_info(lm_name)?;
    push_unique(&model, "lm_head", lm, &mut seen, &mut out)?;

    ensure!(
        !out.is_empty(),
        "no quantized projection tensors found in {}",
        path.display()
    );
    Ok(out)
}

fn push_unique(
    model: &str,
    kind: &str,
    info: &TensorInfo,
    seen: &mut HashSet<(String, GgmlType, Vec<usize>)>,
    out: &mut Vec<BenchTensor>,
) -> Result<()> {
    if !matches!(
        info.dtype,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
    ) {
        return Ok(());
    }
    ensure!(
        info.shape.len() == 2,
        "tensor `{}` must be rank-2, got {:?}",
        info.name,
        info.shape
    );
    let key = (kind.to_string(), info.dtype, info.shape.clone());
    if !seen.insert(key) {
        return Ok(());
    }
    out.push(BenchTensor {
        model: model.to_string(),
        tensor_kind: kind.to_string(),
        tensor_name: info.name.clone(),
        dtype: info.dtype,
        rows: info.shape[0],
        k: info.shape[1],
        weight_bytes: info.size_in_bytes()?,
    });
    Ok(())
}

fn compile_kernels(arch: &str) -> Result<PathBuf> {
    let toolkit = std::env::var("CUDA_TOOLKIT_PATH").unwrap_or_else(|_| "/opt/cuda".to_string());
    let nvcc = Path::new(&toolkit).join("bin/nvcc");
    ensure!(nvcc.exists(), "nvcc not found at {}", nvcc.display());

    let out_dir = std::env::temp_dir().join("grout_quant_gemv_microbench");
    fs::create_dir_all(&out_dir)?;
    let cu_path = out_dir.join("quant_gemv.cu");
    let cubin_path = out_dir.join(format!("quant_gemv_{arch}.cubin"));
    fs::write(&cu_path, CUDA_SRC)?;

    let output = Command::new(&nvcc)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--cubin")
        .arg(format!("-arch={arch}"))
        .arg("-o")
        .arg(&cubin_path)
        .arg(&cu_path)
        .output()
        .with_context(|| format!("failed to run {}", nvcc.display()))?;
    if !output.status.success() {
        bail!(
            "nvcc failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    Ok(cubin_path)
}

const CUDA_SRC: &str = r#"
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float load_half_le(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ void get_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float deq_q4k(const unsigned char* row, int col) {
    const int block = col >> 8;
    const int n = col & 255;
    const unsigned char* b = row + block * 144;
    const float d = load_half_le(b + 0);
    const float dmin = load_half_le(b + 2);
    const unsigned char* scales = b + 4;
    const unsigned char* qs = b + 16;
    const int group = n >> 6;
    const int in64 = n & 63;
    const int is = group * 2 + (in64 >= 32);
    int sc, m;
    get_scale_min_k4(is, scales, &sc, &m);
    const unsigned char qb = qs[group * 32 + (in64 & 31)];
    const int q = (in64 < 32) ? (qb & 0xF) : (qb >> 4);
    return d * (float)sc * (float)q - dmin * (float)m;
}

__device__ __forceinline__ float deq_q5k(const unsigned char* row, int col) {
    const int block = col >> 8;
    const int n = col & 255;
    const unsigned char* b = row + block * 176;
    const float d = load_half_le(b + 0);
    const float dmin = load_half_le(b + 2);
    const unsigned char* scales = b + 4;
    const unsigned char* qh = b + 16;
    const unsigned char* ql = b + 48;
    const int group = n >> 6;
    const int in64 = n & 63;
    const int is = group * 2 + (in64 >= 32);
    int sc, m;
    get_scale_min_k4(is, scales, &sc, &m);
    const int idx = in64 & 31;
    const unsigned char qb = ql[group * 32 + idx];
    const int low = in64 < 32;
    const int qbase = low ? (qb & 0xF) : (qb >> 4);
    const unsigned char mask = low ? (unsigned char)(1u << (2 * group)) : (unsigned char)(2u << (2 * group));
    const int q = qbase + ((qh[idx] & mask) ? 16 : 0);
    return d * (float)sc * (float)q - dmin * (float)m;
}

__device__ __forceinline__ float deq_q6k(const unsigned char* row, int col) {
    const int block = col >> 8;
    const int n = col & 255;
    const unsigned char* b = row + block * 210;
    const unsigned char* ql_all = b + 0;
    const unsigned char* qh_all = b + 128;
    const signed char* sc_all = (const signed char*)(b + 192);
    const float d = load_half_le(b + 208);
    const int idx = n >> 7;
    const int within = n & 127;
    const unsigned char* ql = ql_all + 64 * idx;
    const unsigned char* qh = qh_all + 32 * idx;
    const signed char* sc = sc_all + 8 * idx;
    int l, is, q;
    if (within < 32) {
        l = within;
        is = l / 16;
        q = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
    } else if (within < 64) {
        l = within - 32;
        is = l / 16 + 2;
        q = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    } else if (within < 96) {
        l = within - 64;
        is = l / 16 + 4;
        q = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    } else {
        l = within - 96;
        is = l / 16 + 6;
        q = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    }
    return d * (float)sc[is] * (float)q;
}

__device__ __forceinline__ float deq_q8_0(const unsigned char* row, int col) {
    const int block = col >> 5;
    const int n = col & 31;
    const unsigned char* b = row + block * 34;
    const float d = load_half_le(b + 0);
    const signed char* qs = (const signed char*)(b + 2);
    return d * (float)qs[n];
}

#define DEFINE_GEMV(NAME, DEQ) \
extern "C" __global__ void NAME(const unsigned char* __restrict__ w, const __half* __restrict__ x, __half* __restrict__ y, int rows, int k, int row_stride_bytes) { \
    const int row = blockIdx.x; \
    const int tid = threadIdx.x; \
    extern __shared__ float smem[]; \
    if (row >= rows) return; \
    const unsigned char* rowp = w + (size_t)row * (size_t)row_stride_bytes; \
    float acc = 0.0f; \
    for (int col = tid; col < k; col += blockDim.x) { \
        acc += DEQ(rowp, col) * __half2float(x[col]); \
    } \
    smem[tid] = acc; \
    __syncthreads(); \
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) { \
        if (tid < stride) smem[tid] += smem[tid + stride]; \
        __syncthreads(); \
    } \
    if (tid == 0) y[row] = __float2half_rn(smem[0]); \
}

DEFINE_GEMV(q4k_gemv, deq_q4k)
DEFINE_GEMV(q5k_gemv, deq_q5k)
DEFINE_GEMV(q6k_gemv, deq_q6k)
DEFINE_GEMV(q8_0_gemv, deq_q8_0)
"#;
