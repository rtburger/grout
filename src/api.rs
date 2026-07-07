use crate::model::Qwen3Engine;
use anyhow::{Result, bail, ensure};
use cuda_async::device_context::set_default_device;
use std::path::Path;
use tokio::runtime::{Builder, Runtime};

/// Synchronous Grout inference engine.
///
/// The engine API is deliberately token-based and blocking. Tokenization,
/// cancellation, streaming, and async orchestration belong in callers/adapters.
pub struct Engine {
    inner: Qwen3Engine,
    rt: Runtime,
    meta: ModelMeta,
    device_ord: usize,
    next_pos: usize,
    session_active: bool,
    pending_reset_error: Option<anyhow::Error>,
}

/// Model load options.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadOpts {
    /// Maximum context length to allocate. Use `0` for the model-specific default.
    pub max_ctx: usize,
    /// CUDA device ordinal.
    pub device_ord: usize,
}

impl Default for LoadOpts {
    fn default() -> Self {
        Self {
            max_ctx: 0,
            device_ord: 0,
        }
    }
}

/// Last-token logits copied to host as `f32`.
#[derive(Clone, Debug)]
pub struct Logits {
    /// Host logits, length `meta().vocab_size`.
    pub values: Vec<f32>,
}

/// Static model metadata needed by token-driven callers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelMeta {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// End-of-sequence token ids used by the model/generation config.
    pub eos_token_ids: Vec<u32>,
    /// Architecture name.
    pub arch: String,
    /// Allocated maximum context length.
    pub max_ctx: usize,
}

impl Engine {
    /// Load a model from a safetensors directory or a GGUF file.
    pub fn load(path: impl AsRef<Path>, opts: LoadOpts) -> Result<Self> {
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| anyhow::anyhow!("failed to create engine runtime: {e}"))?;
        set_default_device(opts.device_ord);
        let max_ctx = (opts.max_ctx != 0).then_some(opts.max_ctx);
        let inner = rt.block_on(Qwen3Engine::load(path.as_ref(), max_ctx))?;
        let meta = ModelMeta {
            vocab_size: inner.api_vocab_size(),
            eos_token_ids: inner.api_eos_token_ids().to_vec(),
            arch: inner.api_arch().to_string(),
            max_ctx: inner.api_max_seq_len(),
        };
        Ok(Self {
            inner,
            rt,
            meta,
            device_ord: opts.device_ord,
            next_pos: 0,
            session_active: false,
            pending_reset_error: None,
        })
    }

    /// Run a new-session prefill over `tokens` and return last-token logits.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Logits> {
        self.take_pending_reset_error()?;
        ensure!(!tokens.is_empty(), "prefill requires at least one token");
        ensure!(
            tokens.len() <= self.meta.max_ctx,
            "prefill length {} exceeds max_ctx={}",
            tokens.len(),
            self.meta.max_ctx
        );
        set_default_device(self.device_ord);
        let values = self.rt.block_on(self.inner.api_prefill_logits(tokens))?;
        self.next_pos = tokens.len();
        self.session_active = true;
        Ok(Logits { values })
    }

    /// Decode one token into the current session and return next-token logits.
    pub fn decode(&mut self, token: u32) -> Result<Logits> {
        self.take_pending_reset_error()?;
        self.ensure_decode_ready()?;
        let position = self.next_pos;
        set_default_device(self.device_ord);
        let values = self
            .rt
            .block_on(self.inner.api_decode_logits(token, position))?;
        self.next_pos += 1;
        Ok(Logits { values })
    }

    /// Decode one token and return the greedy next token using device argmax.
    ///
    /// This avoids copying the full logits vector to the host.
    pub fn decode_greedy(&mut self, token: u32) -> Result<u32> {
        self.take_pending_reset_error()?;
        self.ensure_decode_ready()?;
        let position = self.next_pos;
        set_default_device(self.device_ord);
        let next = self
            .rt
            .block_on(self.inner.api_decode_greedy(token, position))?;
        self.next_pos += 1;
        Ok(next)
    }

    /// Clear KV state and start a new session.
    ///
    /// On failure the error is returned AND stashed: a caller that ignores
    /// the Result still fails loudly on its next engine call instead of
    /// running against a CUDA context in an undefined error state.
    pub fn reset(&mut self) -> Result<()> {
        self.next_pos = 0;
        self.session_active = false;
        set_default_device(self.device_ord);
        if let Err(err) = self.rt.block_on(self.inner.api_reset()) {
            let msg = format!("reset failed: {err:#}");
            self.pending_reset_error = Some(err);
            bail!("{msg}");
        }
        Ok(())
    }

    /// Return static model metadata.
    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    /// Run a dummy prefill+decode to force CUDA/cuTile JIT before user traffic.
    pub fn warmup(&mut self) -> Result<()> {
        self.take_pending_reset_error()?;
        let token = self.meta.eos_token_ids.first().copied().unwrap_or(0);
        set_default_device(self.device_ord);
        self.rt.block_on(self.inner.api_warmup(token))?;
        self.next_pos = 0;
        self.session_active = false;
        Ok(())
    }

    fn ensure_decode_ready(&self) -> Result<()> {
        if !self.session_active {
            bail!("decode requires a preceding prefill");
        }
        ensure!(
            self.next_pos < self.meta.max_ctx,
            "decode position {} exceeds max_ctx={}",
            self.next_pos,
            self.meta.max_ctx
        );
        Ok(())
    }

    fn take_pending_reset_error(&mut self) -> Result<()> {
        if let Some(err) = self.pending_reset_error.take() {
            Err(err).map_err(|e| anyhow::anyhow!("previous reset failed: {e:#}"))
        } else {
            Ok(())
        }
    }
}

impl Logits {
    /// Return logits as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Consume and return the backing vector.
    pub fn into_vec(self) -> Vec<f32> {
        self.values
    }
}
