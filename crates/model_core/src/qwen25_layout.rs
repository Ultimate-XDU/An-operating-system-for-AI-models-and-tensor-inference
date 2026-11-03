//! Layout constants and SafeTensors key names for `Qwen/Qwen2.5-1.5B-Instruct`.
//!
//! Numbers and spellings are pinned in `docs/qwen25-1.5b-instruct-layout.md`.
//! The `model.layers.{i}.*` suffixes match the key pattern in ember
//! `Model::load` (MIT, Copyright (c) 2026 Ali Zulfiqar,
//! <https://github.com/codewithfourtix/ember> `src/model.rs`). Only those
//! spellings are shared. This module does not decode BF16 into `Vec<f32>`.
//!
//! [`crate::QwenStub`] does not read these. The smoke model still uses
//! `tok_embeddings` / `lm_head` from [`crate::cycle_fixture`].
//! Byte views live in [`crate::qwen25_bf16`].

use alloc::string::String;

/// `config.json` `num_hidden_layers`. Layer index `i` runs `0..NUM_HIDDEN_LAYERS`.
pub const NUM_HIDDEN_LAYERS: usize = 28;

/// `config.json` `hidden_size`.
pub const HIDDEN_SIZE: usize = 1536;

/// `config.json` `intermediate_size` (SwiGLU hidden width).
pub const INTERMEDIATE_SIZE: usize = 8960;

/// `config.json` `num_attention_heads`.
pub const NUM_ATTENTION_HEADS: usize = 12;

/// `config.json` `num_key_value_heads` (GQA).
pub const NUM_KEY_VALUE_HEADS: usize = 2;

/// `hidden_size / num_attention_heads`. The config has no separate field.
pub const HEAD_DIM: usize = HIDDEN_SIZE / NUM_ATTENTION_HEADS;

/// K or V projection width: `num_key_value_heads * head_dim`.
pub const KV_DIM: usize = NUM_KEY_VALUE_HEADS * HEAD_DIM;

/// `config.json` `vocab_size`.
///
/// `tie_word_embeddings=true`: the checkpoint has no `lm_head` tensor.
/// Output projection is [`EMBED_TOKENS`].
pub const VOCAB_SIZE: usize = 151936;

/// `config.json` `tie_word_embeddings`. There is no independent `lm_head` key.
pub const TIE_WORD_EMBEDDINGS: bool = true;

/// Tensors inside one `model.layers.{i}` block. `o_proj` and the MLP have no bias.
pub const TENSORS_PER_LAYER: usize = 12;

/// `TENSORS_PER_LAYER * NUM_HIDDEN_LAYERS` plus embed and final norm.
pub const TENSOR_COUNT: usize = TENSORS_PER_LAYER * NUM_HIDDEN_LAYERS + 2;

/// Token embedding. Also the tied output projection.
pub const EMBED_TOKENS: &str = "model.embed_tokens.weight";

/// Final RMSNorm weight (`model.norm`, not a per-layer norm).
pub const FINAL_NORM: &str = "model.norm.weight";

/// `[vocab, hidden]`.
pub const EMBED_SHAPE: [usize; 2] = [VOCAB_SIZE, HIDDEN_SIZE];

/// `[hidden]`.
pub const FINAL_NORM_SHAPE: [usize; 1] = [HIDDEN_SIZE];

/// `[hidden]`.
pub const INPUT_LAYERNORM_SHAPE: [usize; 1] = [HIDDEN_SIZE];

/// `[hidden]`.
pub const POST_ATTENTION_LAYERNORM_SHAPE: [usize; 1] = [HIDDEN_SIZE];

/// `q_proj` weight `[num_attention_heads * head_dim, hidden]` = `[1536, 1536]`.
pub const Q_PROJ_WEIGHT_SHAPE: [usize; 2] = [HIDDEN_SIZE, HIDDEN_SIZE];

/// `q_proj` bias `[1536]`.
pub const Q_PROJ_BIAS_SHAPE: [usize; 1] = [HIDDEN_SIZE];

/// `k_proj` weight `[kv_dim, hidden]` = `[256, 1536]`.
pub const K_PROJ_WEIGHT_SHAPE: [usize; 2] = [KV_DIM, HIDDEN_SIZE];

/// `k_proj` bias `[256]`.
pub const K_PROJ_BIAS_SHAPE: [usize; 1] = [KV_DIM];

/// `v_proj` weight `[256, 1536]`.
pub const V_PROJ_WEIGHT_SHAPE: [usize; 2] = [KV_DIM, HIDDEN_SIZE];

/// `v_proj` bias `[256]`.
pub const V_PROJ_BIAS_SHAPE: [usize; 1] = [KV_DIM];

/// `o_proj` weight `[hidden, hidden]`. No `o_proj` bias in this checkpoint.
pub const O_PROJ_WEIGHT_SHAPE: [usize; 2] = [HIDDEN_SIZE, HIDDEN_SIZE];

/// `gate_proj` weight `[intermediate, hidden]`.
pub const GATE_PROJ_WEIGHT_SHAPE: [usize; 2] = [INTERMEDIATE_SIZE, HIDDEN_SIZE];

/// `up_proj` weight `[intermediate, hidden]`.
pub const UP_PROJ_WEIGHT_SHAPE: [usize; 2] = [INTERMEDIATE_SIZE, HIDDEN_SIZE];

/// `down_proj` weight `[hidden, intermediate]`.
pub const DOWN_PROJ_WEIGHT_SHAPE: [usize; 2] = [HIDDEN_SIZE, INTERMEDIATE_SIZE];

/// Suffixes of `model.layers.{i}.*`, in header order.
///
/// No `self_attn.o_proj.bias`, no `mlp.*.bias`, and no `lm_head`.
pub const LAYER_SUFFIXES: [&str; TENSORS_PER_LAYER] = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// `model.layers.{layer}.{suffix}`.
///
/// `layer` is not range-checked here. [`crate::qwen25_bf16::layer`] rejects
/// `layer >= NUM_HIDDEN_LAYERS` before it looks anything up.
pub fn layer_key(layer: usize, suffix: &str) -> String {
    alloc::format!("model.layers.{layer}.{suffix}")
}

pub fn input_layernorm_key(layer: usize) -> String {
    layer_key(layer, "input_layernorm.weight")
}

pub fn q_proj_weight_key(layer: usize) -> String {
    layer_key(layer, "self_attn.q_proj.weight")
}

pub fn q_proj_bias_key(layer: usize) -> String {
    layer_key(layer, "self_attn.q_proj.bias")
}

pub fn k_proj_weight_key(layer: usize) -> String {
    layer_key(layer, "self_attn.k_proj.weight")
}

pub fn k_proj_bias_key(layer: usize) -> String {
    layer_key(layer, "self_attn.k_proj.bias")
}

pub fn v_proj_weight_key(layer: usize) -> String {
    layer_key(layer, "self_attn.v_proj.weight")
}

pub fn v_proj_bias_key(layer: usize) -> String {
    layer_key(layer, "self_attn.v_proj.bias")
}

pub fn o_proj_weight_key(layer: usize) -> String {
    layer_key(layer, "self_attn.o_proj.weight")
}

pub fn post_attention_layernorm_key(layer: usize) -> String {
    layer_key(layer, "post_attention_layernorm.weight")
}

pub fn gate_proj_weight_key(layer: usize) -> String {
    layer_key(layer, "mlp.gate_proj.weight")
}

pub fn up_proj_weight_key(layer: usize) -> String {
    layer_key(layer, "mlp.up_proj.weight")
}

pub fn down_proj_weight_key(layer: usize) -> String {
    layer_key(layer, "mlp.down_proj.weight")
}

/// The twelve per-layer keys, in [`LAYER_SUFFIXES`] order.
pub fn layer_tensor_keys(layer: usize) -> [String; TENSORS_PER_LAYER] {
    [
        input_layernorm_key(layer),
        q_proj_weight_key(layer),
        q_proj_bias_key(layer),
        k_proj_weight_key(layer),
        k_proj_bias_key(layer),
        v_proj_weight_key(layer),
        v_proj_bias_key(layer),
        o_proj_weight_key(layer),
        post_attention_layernorm_key(layer),
        gate_proj_weight_key(layer),
        up_proj_weight_key(layer),
        down_proj_weight_key(layer),
    ]
}

/// Expected BF16 shapes for [`LAYER_SUFFIXES`], same order.
pub fn layer_tensor_shapes() -> [&'static [usize]; TENSORS_PER_LAYER] {
    [
        INPUT_LAYERNORM_SHAPE.as_slice(),
        Q_PROJ_WEIGHT_SHAPE.as_slice(),
        Q_PROJ_BIAS_SHAPE.as_slice(),
        K_PROJ_WEIGHT_SHAPE.as_slice(),
        K_PROJ_BIAS_SHAPE.as_slice(),
        V_PROJ_WEIGHT_SHAPE.as_slice(),
        V_PROJ_BIAS_SHAPE.as_slice(),
        O_PROJ_WEIGHT_SHAPE.as_slice(),
        POST_ATTENTION_LAYERNORM_SHAPE.as_slice(),
        GATE_PROJ_WEIGHT_SHAPE.as_slice(),
        UP_PROJ_WEIGHT_SHAPE.as_slice(),
        DOWN_PROJ_WEIGHT_SHAPE.as_slice(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_matches_layout_doc() {
        assert_eq!(NUM_HIDDEN_LAYERS, 28);
        assert_eq!(HIDDEN_SIZE, 1536);
        assert_eq!(INTERMEDIATE_SIZE, 8960);
        assert_eq!(NUM_ATTENTION_HEADS, 12);
        assert_eq!(NUM_KEY_VALUE_HEADS, 2);
        assert_eq!(HEAD_DIM, 128);
        assert_eq!(KV_DIM, 256);
        assert_eq!(VOCAB_SIZE, 151936);
        assert_eq!(TENSORS_PER_LAYER, 12);
        assert_eq!(TENSOR_COUNT, 338);
        assert_eq!(EMBED_TOKENS, "model.embed_tokens.weight");
        assert_eq!(FINAL_NORM, "model.norm.weight");
        assert!(!EMBED_TOKENS.starts_with("lm_head"));
        assert!(!FINAL_NORM.starts_with("lm_head"));
        assert_eq!(TIE_WORD_EMBEDDINGS, !EMBED_TOKENS.contains("lm_head"));
        assert_eq!(EMBED_SHAPE, [151936, 1536]);
        assert_eq!(FINAL_NORM_SHAPE, [1536]);
        assert_eq!(Q_PROJ_WEIGHT_SHAPE, [1536, 1536]);
        assert_eq!(Q_PROJ_BIAS_SHAPE, [1536]);
        assert_eq!(K_PROJ_WEIGHT_SHAPE, [256, 1536]);
        assert_eq!(K_PROJ_BIAS_SHAPE, [256]);
        assert_eq!(V_PROJ_WEIGHT_SHAPE, [256, 1536]);
        assert_eq!(V_PROJ_BIAS_SHAPE, [256]);
        assert_eq!(O_PROJ_WEIGHT_SHAPE, [1536, 1536]);
        assert_eq!(GATE_PROJ_WEIGHT_SHAPE, [8960, 1536]);
        assert_eq!(UP_PROJ_WEIGHT_SHAPE, [8960, 1536]);
        assert_eq!(DOWN_PROJ_WEIGHT_SHAPE, [1536, 8960]);
        assert_eq!(INPUT_LAYERNORM_SHAPE, [1536]);
        assert_eq!(POST_ATTENTION_LAYERNORM_SHAPE, [1536]);
        assert_eq!(HEAD_DIM, tensor_core::HEAD_DIM_QWEN25_1_5B);
        assert_eq!(
            NUM_ATTENTION_HEADS,
            tensor_core::NUM_ATTENTION_HEADS_QWEN25_1_5B
        );
        assert_eq!(NUM_KEY_VALUE_HEADS, tensor_core::NUM_KV_HEADS_QWEN25_1_5B);
    }

    #[test]
    fn layer_keys_at_zero_and_last() {
        let layer0 = [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ];
        let layer27 = [
            "model.layers.27.input_layernorm.weight",
            "model.layers.27.self_attn.q_proj.weight",
            "model.layers.27.self_attn.q_proj.bias",
            "model.layers.27.self_attn.k_proj.weight",
            "model.layers.27.self_attn.k_proj.bias",
            "model.layers.27.self_attn.v_proj.weight",
            "model.layers.27.self_attn.v_proj.bias",
            "model.layers.27.self_attn.o_proj.weight",
            "model.layers.27.post_attention_layernorm.weight",
            "model.layers.27.mlp.gate_proj.weight",
            "model.layers.27.mlp.up_proj.weight",
            "model.layers.27.mlp.down_proj.weight",
        ];
        let keys0 = layer_tensor_keys(0);
        let keys27 = layer_tensor_keys(27);
        for (got, expect) in keys0.iter().zip(layer0) {
            assert_eq!(got, expect);
            assert!(!got.contains("lm_head"));
            assert!(!got.contains("o_proj.bias"));
            assert!(!got.contains("gate_proj.bias"));
            assert!(!got.contains("up_proj.bias"));
            assert!(!got.contains("down_proj.bias"));
        }
        for (got, expect) in keys27.iter().zip(layer27) {
            assert_eq!(got, expect);
            assert!(!got.contains("lm_head"));
        }
        assert_eq!(keys0.len(), TENSORS_PER_LAYER);
        assert_eq!(keys27.len(), TENSORS_PER_LAYER);
        assert_ne!(keys0[0], keys27[0]);
        let shapes = layer_tensor_shapes();
        assert_eq!(shapes.len(), LAYER_SUFFIXES.len());
        for (suffix, key) in LAYER_SUFFIXES.iter().zip(keys0.iter()) {
            assert_eq!(layer_key(0, suffix), *key);
        }
    }
}
