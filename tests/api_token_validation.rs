use anyhow::Result;
use grout::{Engine, LoadOpts};
use std::path::PathBuf;

/// H1 regression: out-of-range token ids must be rejected host-side before
/// they reach the unchecked embedding-gather kernels (where they become
/// out-of-bounds device reads, or negative offsets for ids >= 2^31), and
/// the rejection must fire before any device work so the CUDA context is
/// left usable.
#[test]
#[ignore = "GPU e2e: set GROUT_QWEN3_4B_Q4_K_M_GGUF (tokenizer.json next to the file)"]
fn out_of_range_token_ids_are_rejected_without_poisoning_the_context() -> Result<()> {
    let Some(path) = std::env::var_os("GROUT_QWEN3_4B_Q4_K_M_GGUF").map(PathBuf::from) else {
        eprintln!("skipping: GROUT_QWEN3_4B_Q4_K_M_GGUF is not set");
        return Ok(());
    };
    let mut engine = Engine::load(
        &path,
        LoadOpts {
            max_ctx: 512,
            device_ord: 0,
        },
    )?;
    let vocab = engine.meta().vocab_size;

    // Prefill with one out-of-range id mixed into valid ids.
    let bad = engine.prefill(&[1u32, vocab as u32, 2u32]);
    let msg = format!(
        "{:#}",
        bad.expect_err("prefill accepted an out-of-range token id")
    );
    assert!(msg.contains("out of range"), "unexpected error: {msg}");

    // An id >= 2^31 bitcasts to a negative kernel offset pre-fix.
    engine
        .prefill(&[u32::MAX])
        .expect_err("prefill accepted a token id >= 2^31");

    // The guard fires before any device work, so a valid session must
    // still run end to end afterwards.
    let logits = engine.prefill(&[1u32, 2, 3])?;
    assert_eq!(logits.as_slice().len(), vocab);
    engine
        .decode(vocab as u32)
        .expect_err("decode accepted an out-of-range token id");
    engine
        .decode_greedy(u32::MAX)
        .expect_err("decode_greedy accepted an out-of-range token id");
    let next = engine.decode_greedy(1u32)?;
    assert!((next as usize) < vocab, "greedy token {next} out of vocab");
    Ok(())
}
