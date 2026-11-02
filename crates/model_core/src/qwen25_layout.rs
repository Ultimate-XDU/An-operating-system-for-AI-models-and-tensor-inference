//! Layout constants for `Qwen/Qwen2.5-1.5B-Instruct` (Phase1 BF16 track).
//!
//! Source of the numbers: `docs/qwen25-1.5b-instruct-layout.md`.
//! [`crate::QwenStub`] does not read these. The smoke model still uses
//! `tok_embeddings` / `lm_head` from [`crate::cycle_fixture`].

/// `config.json` `num_hidden_layers`. Layer index `i` runs `0..NUM_HIDDEN_LAYERS`.
pub const NUM_HIDDEN_LAYERS: usize = 28;

/// `config.json` `hidden_size`.
pub const HIDDEN_SIZE: usize = 1536;

/// `config.json` `vocab_size`.
///
/// `tie_word_embeddings=true`: the checkpoint has no `lm_head` tensor.
/// Output projection is [`EMBED_TOKENS`].
pub const VOCAB_SIZE: usize = 151936;

/// Token embedding. Also the tied output projection.
pub const EMBED_TOKENS: &str = "model.embed_tokens.weight";

/// Final RMSNorm weight (`model.norm`, not a per-layer norm).
pub const FINAL_NORM: &str = "model.norm.weight";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_matches_layout_doc() {
        assert_eq!(NUM_HIDDEN_LAYERS, 28);
        assert_eq!(HIDDEN_SIZE, 1536);
        assert_eq!(VOCAB_SIZE, 151936);
        assert_eq!(EMBED_TOKENS, "model.embed_tokens.weight");
        assert_eq!(FINAL_NORM, "model.norm.weight");
        assert!(!EMBED_TOKENS.starts_with("lm_head"));
        assert!(!FINAL_NORM.starts_with("lm_head"));
    }
}
