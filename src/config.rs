use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
    pub eos_token_id: u32,
}

impl Qwen3Config {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let cfg_text = fs::read_to_string(&cfg_path)
            .with_context(|| format!("failed to read {}", cfg_path.display()))?;
        let cfg: Self = serde_json::from_str(&cfg_text)
            .with_context(|| format!("failed to parse {}", cfg_path.display()))?;
        cfg.validate()
            .with_context(|| format!("invalid config in {}", cfg_path.display()))?;
        Ok(cfg)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Range-check an untrusted config (config.json or GGUF metadata).
    /// The upper bounds are tripwires for corrupt or malicious files, not
    /// statements of supported model sizes — real models sit far inside
    /// them. Without this, num_key_value_heads=0 panics at inference setup
    /// (num_kv_groups division) and NaN eps/theta silently poisons math.
    pub fn validate(&self) -> Result<()> {
        let size_fields = [
            ("vocab_size", self.vocab_size, 1usize << 24),
            ("hidden_size", self.hidden_size, 1 << 20),
            ("intermediate_size", self.intermediate_size, 1 << 22),
            ("num_hidden_layers", self.num_hidden_layers, 1 << 12),
            ("num_attention_heads", self.num_attention_heads, 1 << 12),
            ("num_key_value_heads", self.num_key_value_heads, 1 << 12),
            ("head_dim", self.head_dim, 1 << 12),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                1 << 25,
            ),
        ];
        for (name, value, max) in size_fields {
            anyhow::ensure!(
                (1..=max).contains(&value),
                "config {name}={value} out of range [1, {max}]"
            );
        }
        anyhow::ensure!(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "num_attention_heads={} must be divisible by num_key_value_heads={}",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        anyhow::ensure!(
            self.head_dim % 2 == 0,
            "head_dim={} must be even (RoPE rotates half-dimensions)",
            self.head_dim
        );
        anyhow::ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0,
            "rms_norm_eps={} must be a positive finite float",
            self.rms_norm_eps
        );
        anyhow::ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "rope_theta={} must be a positive finite float",
            self.rope_theta
        );
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct SafetensorsIndex {
    pub weight_map: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenId {
    pub fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(x) => vec![x],
            Self::Many(xs) => xs,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    pub do_sample: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub eos_token_id: Option<EosTokenId>,
}

impl GenerationConfig {
    pub fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("generation_config.json");
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Some(cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_4b() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 151936,
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            max_position_embeddings: 40960,
            tie_word_embeddings: true,
            use_sliding_window: false,
            eos_token_id: 151645,
        }
    }

    #[test]
    fn valid_config_passes() {
        qwen3_4b().validate().unwrap();
    }

    #[test]
    fn zero_kv_heads_is_an_error_not_a_setup_panic() {
        let mut c = qwen3_4b();
        c.num_key_value_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn non_divisible_head_grouping_is_rejected() {
        let mut c = qwen3_4b();
        c.num_key_value_heads = 5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn nan_eps_and_zero_theta_are_rejected() {
        let mut c = qwen3_4b();
        c.rms_norm_eps = f32::NAN;
        assert!(c.validate().is_err());
        let mut c = qwen3_4b();
        c.rope_theta = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn zero_and_absurd_sizes_are_rejected() {
        let mut c = qwen3_4b();
        c.hidden_size = 0;
        assert!(c.validate().is_err());
        let mut c = qwen3_4b();
        c.vocab_size = 1 << 30;
        assert!(c.validate().is_err());
    }
}
