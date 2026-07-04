use anyhow::{Context, Result};
use grout::Qwen3Engine;
use std::path::{Path, PathBuf};

#[tokio::test(flavor = "current_thread")]
#[ignore = "GGUF GPU e2e: set GROUT_QWEN3_06B_GGUF; tokenizer.json must be next to the file"]
async fn qwen3_06b_gguf_generates_100_greedy_tokens() -> Result<()> {
    let Some(path) = env_path("GROUT_QWEN3_06B_GGUF") else {
        eprintln!("skipping: GROUT_QWEN3_06B_GGUF is not set");
        return Ok(());
    };
    run_gguf_coherence(&path, "Explain compiler design in one short paragraph.").await
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "GGUF GPU e2e: set GROUT_QWEN3_4B_Q4_K_M_GGUF and run headless"]
async fn qwen3_4b_q4_k_m_gguf_generates_100_greedy_tokens() -> Result<()> {
    let Some(path) = env_path("GROUT_QWEN3_4B_Q4_K_M_GGUF") else {
        eprintln!("skipping: GROUT_QWEN3_4B_Q4_K_M_GGUF is not set");
        return Ok(());
    };
    run_gguf_coherence(&path, "Write one paragraph about compiler design.").await
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "optional parser proof: set GROUT_QWEN3_06B_F16_GGUF and GROUT_QWEN3_06B_SAFETENSORS"]
async fn qwen3_06b_f16_gguf_matches_safetensors_tokens_when_available() -> Result<()> {
    let Some(gguf_path) = env_path("GROUT_QWEN3_06B_F16_GGUF") else {
        eprintln!("skipping: GROUT_QWEN3_06B_F16_GGUF is not set");
        return Ok(());
    };
    let Some(safetensors_dir) = env_path("GROUT_QWEN3_06B_SAFETENSORS") else {
        eprintln!("skipping: GROUT_QWEN3_06B_SAFETENSORS is not set");
        return Ok(());
    };

    let prompt = "Hello, how are you?";
    let mut gguf = Qwen3Engine::load(&gguf_path, Some(512)).await?;
    gguf.set_sampling_enabled(false);
    gguf.set_ignore_eos(true);
    let gguf_out = gguf.generate(prompt, 64).await?;

    let mut safetensors = Qwen3Engine::load(&safetensors_dir, Some(512)).await?;
    safetensors.set_sampling_enabled(false);
    safetensors.set_ignore_eos(true);
    let safetensors_out = safetensors.generate(prompt, 64).await?;

    assert_eq!(
        gguf_out.token_ids, safetensors_out.token_ids,
        "f16 GGUF and safetensors loaders produced different greedy token ids\nGGUF text: {:?}\nsafetensors text: {:?}",
        gguf_out.text, safetensors_out.text
    );
    Ok(())
}

async fn run_gguf_coherence(path: &Path, prompt: &str) -> Result<()> {
    let mut engine = Qwen3Engine::load(path, Some(512)).await?;
    engine.set_sampling_enabled(false);
    engine.set_ignore_eos(true);
    let output = engine.generate(prompt, 100).await?;
    assert_eq!(
        output.generated_tokens, 100,
        "ignore_eos should force 100 generated tokens"
    );
    assert_coherent(&output.text)
        .with_context(|| format!("generated text was {:?}", output.text))?;
    Ok(())
}

fn assert_coherent(text: &str) -> Result<()> {
    let alphabetic = text.chars().filter(|c| c.is_alphabetic()).count();
    let spaces = text.chars().filter(|c| c.is_whitespace()).count();
    anyhow::ensure!(text.len() >= 80, "too short");
    anyhow::ensure!(alphabetic >= 40, "too few alphabetic characters");
    anyhow::ensure!(spaces >= 8, "too few spaces");
    anyhow::ensure!(
        !text.contains('\u{fffd}'),
        "contains replacement characters"
    );
    Ok(())
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}
