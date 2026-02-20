use anyhow::{Result, anyhow, bail};
use nv_cuda::{CudaStream, DriverError, IntoResult, stream, sys};
use std::mem::MaybeUninit;
use std::sync::Arc;

const CU_STREAM_CAPTURE_MODE_RELAXED: u32 = 2;

pub struct CudaGraphExec {
    stream: Arc<CudaStream>,
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
}

impl CudaGraphExec {
    pub fn capture<F>(stream: Arc<CudaStream>, f: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        let ctx = stream.context().clone();
        ctx.bind_to_thread()
            .map_err(|e| anyhow!("failed to bind CUDA context: {e:?}"))?;
        unsafe {
            stream::begin_capture(stream.cu_stream(), CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| anyhow!("cuStreamBeginCapture_v2 failed: {e:?}"))?;
        }

        let work_res = f();
        let end_capture = unsafe { stream::end_capture(stream.cu_stream()) };
        let cu_graph = match (work_res, end_capture) {
            (Err(err), Ok(cu_graph)) => {
                if !cu_graph.is_null() {
                    let _ = unsafe { sys::cuGraphDestroy(cu_graph).result() };
                }
                return Err(err);
            }
            (Err(err), Err(_capture_err)) => {
                return Err(err);
            }
            (Ok(()), Err(err)) => {
                return Err(anyhow!("cuStreamEndCapture failed: {err:?}"));
            }
            (Ok(()), Ok(cu_graph)) => cu_graph,
        };

        if cu_graph.is_null() {
            bail!("cuStreamEndCapture returned null graph");
        }

        let mut cu_graph_exec = MaybeUninit::<sys::CUgraphExec>::uninit();
        let cu_graph_exec = unsafe {
            // CU_GRAPH_INSTANTIATE_FLAG_NONE = 0
            match sys::cuGraphInstantiateWithFlags(cu_graph_exec.as_mut_ptr(), cu_graph, 0).result()
            {
                Ok(()) => cu_graph_exec.assume_init(),
                Err(e) => {
                    let _ = destroy_graph(cu_graph);
                    return Err(anyhow!("cuGraphInstantiateWithFlags failed: {e:?}"));
                }
            }
        };
        if let Err(e) = unsafe { sys::cuGraphUpload(cu_graph_exec, stream.cu_stream()).result() } {
            unsafe {
                let _ = destroy_graph_exec(cu_graph_exec);
                let _ = destroy_graph(cu_graph);
            }
            return Err(anyhow!("cuGraphUpload failed: {e:?}"));
        }
        Ok(Self {
            stream,
            cu_graph,
            cu_graph_exec,
        })
    }

    pub fn launch(&self) -> Result<()> {
        unsafe {
            sys::cuGraphLaunch(self.cu_graph_exec, self.stream.cu_stream())
                .result()
                .map_err(|e| anyhow!("cuGraphLaunch failed: {e:?}"))?;
        }
        Ok(())
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        let ctx = self.stream.context();
        ctx.record_err(ctx.bind_to_thread());

        let cu_graph_exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !cu_graph_exec.is_null() {
            ctx.record_err(unsafe { destroy_graph_exec(cu_graph_exec) });
        }

        let cu_graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !cu_graph.is_null() {
            ctx.record_err(unsafe { destroy_graph(cu_graph) });
        }
    }
}

unsafe fn destroy_graph_exec(graph_exec: sys::CUgraphExec) -> Result<(), DriverError> {
    unsafe { sys::cuGraphExecDestroy(graph_exec) }.result()
}

unsafe fn destroy_graph(graph: sys::CUgraph) -> Result<(), DriverError> {
    unsafe { sys::cuGraphDestroy(graph) }.result()
}
