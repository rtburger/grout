use anyhow::{Result, bail, ensure};
use clap::Parser;
use grout::config::GenerationConfig;
use grout::{Engine, LoadOpts, Logits};
use rand::Rng;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3 inference engine on cutile-rs")]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long = "max-ctx", visible_alias = "max-seq-len")]
    max_ctx: Option<usize>,

    #[arg(long, default_value_t = 0)]
    device_ord: usize,

    #[arg(long, default_value_t = false)]
    sample: bool,

    #[arg(long, default_value_t = false)]
    raw_prompt: bool,

    #[arg(long, default_value_t = false)]
    device_argmax: bool,

    #[arg(long, default_value_t = false)]
    profile: bool,

    /// Discarded warmup generations before the measured run. The first
    /// generation pays CUDA/cuTile JIT, which otherwise lands in the reported
    /// prompt t/s. The default 1 warmup makes the reported t/s reflect steady
    /// state; set 0 to see cold-start numbers.
    #[arg(long, default_value_t = 1)]
    warmup_reps: usize,
}

struct SamplingParams {
    temperature: f32,
    top_k: usize,
    top_p: f32,
}

struct GenerationOutput {
    text: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    prompt_elapsed: Duration,
    decode_elapsed: Duration,
    total_elapsed: Duration,
}

impl GenerationOutput {
    fn prompt_tps(&self) -> f64 {
        let secs = self.prompt_elapsed.as_secs_f64().max(1.0e-9);
        self.prompt_tokens as f64 / secs
    }

    fn decode_phase_tps(&self) -> f64 {
        let secs = self.decode_elapsed.as_secs_f64().max(1.0e-9);
        self.generated_tokens as f64 / secs
    }

    fn total_tps(&self) -> f64 {
        let secs = self.total_elapsed.as_secs_f64().max(1.0e-9);
        (self.prompt_tokens + self.generated_tokens) as f64 / secs
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let opts = LoadOpts {
        max_ctx: args.max_ctx.unwrap_or(0),
        device_ord: args.device_ord,
    };
    let mut engine = Engine::load(&args.model, opts)?;
    let tokenizer_path = tokenizer_json_path(&args.model)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load {}: {e}", tokenizer_path.display()))?;
    let sampling = sampling_params(&args.model)?;

    println!("Loaded model from {}", args.model.display());
    println!("Prompt: {}", args.prompt);
    println!("Generating {} tokens...", args.max_new_tokens);

    // Preserve the compatibility flag but keep the frozen library API minimal.
    let _profile_requested = args.profile;

    let warmup_tokens = args.max_new_tokens.min(8);
    for _ in 0..args.warmup_reps {
        let _ = generate_with_api(
            &mut engine,
            &tokenizer,
            &args.prompt,
            warmup_tokens,
            args.raw_prompt,
            args.sample,
            args.device_argmax,
            &sampling,
        )?;
    }

    let output = generate_with_api(
        &mut engine,
        &tokenizer,
        &args.prompt,
        args.max_new_tokens,
        args.raw_prompt,
        args.sample,
        args.device_argmax,
        &sampling,
    )?;
    println!();
    println!("{}", output.text);
    println!(
        "t/s: {:.2} prompt, {:.2} decode phase, {:.2} end-to-end (prompt_tokens={}, generated_tokens={}, prompt_s={:.3}, decode_s={:.3}, total_s={:.3})",
        output.prompt_tps(),
        output.decode_phase_tps(),
        output.total_tps(),
        output.prompt_tokens,
        output.generated_tokens,
        output.prompt_elapsed.as_secs_f64(),
        output.decode_elapsed.as_secs_f64(),
        output.total_elapsed.as_secs_f64(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_with_api(
    engine: &mut Engine,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    raw_prompt: bool,
    sample: bool,
    device_argmax: bool,
    sampling: &SamplingParams,
) -> Result<GenerationOutput> {
    let prompt = maybe_apply_chat_template(tokenizer, raw_prompt, prompt);
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    ensure!(!prompt_ids.is_empty(), "prompt produced no tokens");
    ensure!(
        prompt_ids
            .len()
            .checked_add(max_new_tokens)
            .is_some_and(|total| total <= engine.meta().max_ctx),
        "requested total sequence length exceeds max_ctx={}",
        engine.meta().max_ctx
    );

    let total_start = Instant::now();
    let prompt_start = Instant::now();
    let mut logits = engine.prefill(&prompt_ids)?;
    let prompt_elapsed = prompt_start.elapsed();

    let decode_start = Instant::now();
    let mut generated_ids = Vec::with_capacity(max_new_tokens);
    let mut rng = rand::thread_rng();

    if max_new_tokens > 0 {
        let mut next = choose_next(&logits, sample, sampling, &mut rng)?;
        if !engine.meta().eos_token_ids.contains(&next) {
            generated_ids.push(next);
            for _ in 1..max_new_tokens {
                next = if device_argmax && !sample {
                    engine.decode_greedy(next)?
                } else {
                    logits = engine.decode(next)?;
                    choose_next(&logits, sample, sampling, &mut rng)?
                };
                if engine.meta().eos_token_ids.contains(&next) {
                    break;
                }
                generated_ids.push(next);
            }
        }
    }

    let decode_elapsed = decode_start.elapsed();
    let total_elapsed = total_start.elapsed();
    let text = if generated_ids.is_empty() {
        String::new()
    } else {
        tokenizer
            .decode(&generated_ids, false)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?
    };

    Ok(GenerationOutput {
        text,
        prompt_tokens: prompt_ids.len(),
        generated_tokens: generated_ids.len(),
        prompt_elapsed,
        decode_elapsed,
        total_elapsed,
    })
}

fn choose_next<R: Rng + ?Sized>(
    logits: &Logits,
    sample: bool,
    sampling: &SamplingParams,
    rng: &mut R,
) -> Result<u32> {
    if sample {
        sample_next(logits.as_slice(), sampling, rng).map(|x| x as u32)
    } else {
        Ok(argmax_f32(logits.as_slice()) as u32)
    }
}

fn argmax_f32(values: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn sample_next<R: Rng + ?Sized>(
    logits: &[f32],
    params: &SamplingParams,
    rng: &mut R,
) -> Result<usize> {
    if logits.is_empty() {
        bail!("cannot sample from empty logits");
    }
    if params.temperature <= 1.0e-5 {
        return Ok(argmax_f32(logits));
    }

    let inv_temp = 1.0f32 / params.temperature;
    let mut scores: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v * inv_temp))
        .collect();

    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    if params.top_k > 0 && params.top_k < scores.len() {
        scores.truncate(params.top_k);
    }

    let max_score = scores[0].1;
    let mut probs: Vec<(usize, f64)> = scores
        .into_iter()
        .map(|(i, s)| (i, ((s - max_score) as f64).exp()))
        .collect();

    if params.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total: f64 = probs.iter().map(|x| x.1).sum();
        if total.is_finite() && total > 0.0 {
            let mut kept = Vec::new();
            let mut cum = 0.0f64;
            for (idx, p) in probs {
                let pn = p / total;
                cum += pn;
                kept.push((idx, p));
                if cum >= params.top_p as f64 && !kept.is_empty() {
                    break;
                }
            }
            probs = kept;
        }
    }

    let sum_p: f64 = probs.iter().map(|x| x.1).sum();
    if !sum_p.is_finite() || sum_p <= 0.0 {
        return Ok(argmax_f32(logits));
    }
    let mut threshold = rng.r#gen::<f64>() * sum_p;
    for (idx, p) in probs {
        threshold -= p;
        if threshold <= 0.0 {
            return Ok(idx);
        }
    }
    Ok(argmax_f32(logits))
}

fn sampling_params(model_path: &Path) -> Result<SamplingParams> {
    let gen_cfg = if model_path.is_dir() {
        GenerationConfig::from_model_dir(model_path)?
    } else {
        None
    };
    Ok(SamplingParams {
        temperature: gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.temperature)
            .unwrap_or(1.0)
            .max(1.0e-5),
        top_k: gen_cfg.as_ref().and_then(|cfg| cfg.top_k).unwrap_or(0),
        top_p: gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.top_p)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0),
    })
}

fn tokenizer_json_path(model_path: &Path) -> Result<PathBuf> {
    if model_path.is_dir() {
        return Ok(model_path.join("tokenizer.json"));
    }
    let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
    let sibling = parent.join("tokenizer.json");
    if sibling.exists() {
        return Ok(sibling);
    }
    bail!(
        "GGUF model `{}` requires tokenizer.json next to the file",
        model_path.display()
    )
}

fn maybe_apply_chat_template(tokenizer: &Tokenizer, raw_prompt: bool, prompt: &str) -> String {
    if raw_prompt || prompt.contains("<|im_start|>") {
        return prompt.to_string();
    }
    let vocab = tokenizer.get_vocab(true);
    if !(vocab.contains_key("<|im_start|>") && vocab.contains_key("<|im_end|>")) {
        return prompt.to_string();
    }
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}
