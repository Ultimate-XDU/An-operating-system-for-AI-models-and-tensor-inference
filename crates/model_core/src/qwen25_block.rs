//! One token through every Qwen2.5 block, then the tied head.
//!
//! Control flow follows ember `Model::forward` (MIT, Copyright (c) 2026 Ali
//! Zulfiqar, <https://github.com/codewithfourtix/ember> `src/model.rs`): for
//! each layer, RMSNorm, attention, residual, RMSNorm, SwiGLU, residual; then
//! the final norm and a matvec with the token embedding. The greedy loop
//! follows ember `src/main.rs` only as far as token ids: this module does
//! not tokenize. Prefill runs the prompt as one chunk (projections and a
//! causal KV write), not `n_prompt` calls of [`forward_token`]. Decode stays
//! on [`forward_token`].
//!
//! Weights stay little-endian BF16 views from [`crate::qwen25_bf16`]. Promotion
//! stays inside `tensor_core` (`quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`).
//! [`crate::attention_layer`] still owns QKV bias, NeoX RoPE, `gqa_attn_f32_strided`
//! (`e / sum`), and the `o_proj` residual. This module does not load a table
//! with `load_f32_tensor`, does not use rayon, and does not look up `lm_head`.
//!
//! Shapes come from [`BlockShape`], not from the 1.5B constants, so a fixture
//! can be small. [`qwen25_bf16::embed_tokens`] and [`qwen25_bf16::layer`] still
//! require the checkpoint shapes. The forward below borrows the same key names
//! with the shape this call asked for.
//!
//! [`qwen25_bf16::embed_tokens`]: crate::qwen25_bf16::embed_tokens
//! [`qwen25_bf16::layer`]: crate::qwen25_bf16::layer

use alloc::vec::Vec;
use core::fmt;

use tensor_core::{
    add_bf16_bias, embed_gather_bf16, gemm_bf16, gemm_bf16_scratch_elems, gemv_bf16,
    gemv_bf16_threads, gqa_attn_f32_strided, rmsnorm_bf16_weight, rope_f32_inplace, swiglu_bf16,
    swiglu_mul_f32, Bf16OpError, GqaMask, GqaShape, SwigluShape, RMS_NORM_EPS_QWEN25,
};
use weight_loader::{MmapWeights, WeightMap};

use crate::attn_wire::{attention_layer, AttnError, AttnShape, AttnWorkspace};
use crate::greedy_token;
use crate::kv_cache::{KvCache, KvError};
use crate::qwen25_bf16::{bf16_tensor, QwenBf16Error};
use crate::qwen25_layout::{
    down_proj_weight_key, gate_proj_weight_key, input_layernorm_key, k_proj_bias_key,
    k_proj_weight_key, o_proj_weight_key, post_attention_layernorm_key, q_proj_bias_key,
    q_proj_weight_key, up_proj_weight_key, v_proj_bias_key, v_proj_weight_key, EMBED_TOKENS,
    FINAL_NORM, INTERMEDIATE_SIZE, NUM_HIDDEN_LAYERS, VOCAB_SIZE,
};

/// Widths for [`forward_token`].
///
/// Qwen2.5-1.5B passes [`BlockShape::qwen25_1_5b`]. A test shrinks the fields.
/// `rms_eps` is the RMSNorm epsilon. The 1.5B track passes
/// [`RMS_NORM_EPS_QWEN25`] (`1e-6`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockShape {
    pub attn: AttnShape,
    pub intermediate: usize,
    pub vocab: usize,
    pub num_layers: usize,
    pub rms_eps: f32,
}

impl BlockShape {
    /// 28 layers, hidden 1536, intermediate 8960, vocab 151936, eps `1e-6`.
    ///
    /// Attention widths are [`AttnShape::qwen25_1_5b`] (12 query heads, 2 KV
    /// heads, head dim 128, `rope_theta` 1e6).
    pub const fn qwen25_1_5b() -> Self {
        Self {
            attn: AttnShape::qwen25_1_5b(),
            intermediate: INTERMEDIATE_SIZE,
            vocab: VOCAB_SIZE,
            num_layers: NUM_HIDDEN_LAYERS,
            rms_eps: RMS_NORM_EPS_QWEN25,
        }
    }
}

/// Scratch for [`forward_token`]. One allocation, reused across tokens.
///
/// `max_seq` matches the [`KvCache`] this workspace is paired with.
#[derive(Debug, Clone)]
pub struct BlockWorkspace {
    shape: BlockShape,
    attn: AttnWorkspace,
    /// Residual stream. Embed writes it; each block adds onto it.
    x: Vec<f32>,
    /// RMSNorm output (input norm, post-attention norm, final norm).
    normed: Vec<f32>,
    /// SwiGLU down-projection, added onto `x`.
    mlp: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    logits: Vec<f32>,
}

impl BlockWorkspace {
    pub fn new(shape: BlockShape, max_seq: usize) -> Result<Self, BlockError> {
        if shape.intermediate == 0 || shape.vocab == 0 || shape.num_layers == 0 {
            return Err(BlockError::Shape);
        }
        let attn = AttnWorkspace::new(shape.attn, max_seq).map_err(|_| BlockError::Shape)?;
        let hidden = shape.attn.hidden;
        Ok(Self {
            shape,
            attn,
            x: zeros(hidden),
            normed: zeros(hidden),
            mlp: zeros(hidden),
            gate: zeros(shape.intermediate),
            up: zeros(shape.intermediate),
            act: zeros(shape.intermediate),
            logits: zeros(shape.vocab),
        })
    }

    pub fn shape(&self) -> BlockShape {
        self.shape
    }

    /// Logits from the last successful [`forward_token`], or zeros before that.
    ///
    /// A failed forward leaves this slice as it was. The tied head writes it
    /// only after every layer has stored, and only then does the cache advance.
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

/// Failure from [`forward_token`], [`prefill`], or [`greedy_decode`].
///
/// On `Err`, the cache cursor is not advanced. Layers that already stored this
/// token keep that slot; a later `forward_token` at the same cursor overwrites
/// them. The workspace may hold a partial residual. Logits stay as they were
/// unless the tied head ran, and the tied head runs only after every layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    /// A count is zero, or a buffer / cache / workspace does not match [`BlockShape`].
    Shape,
    /// `pos` is not the cache cursor.
    Position,
    /// `token` is outside `0..vocab`.
    Token,
    /// A tensor is missing, not BF16, or not the shape this call asked for.
    Weight(QwenBf16Error),
    /// [`attention_layer`] rejected the token. The cursor was not moved there.
    Attn(AttnError),
    /// The cache rejected the cursor.
    Cache(KvError),
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockError::Shape => write!(f, "block shape does not match the buffers"),
            BlockError::Position => write!(f, "block position does not match the kv cursor"),
            BlockError::Token => write!(f, "token id is out of range"),
            BlockError::Weight(err) => write!(f, "block weight: {err}"),
            BlockError::Attn(err) => write!(f, "attention: {err}"),
            BlockError::Cache(err) => write!(f, "kv cache: {err}"),
        }
    }
}

fn zeros(n: usize) -> Vec<f32> {
    let mut out = Vec::new();
    out.resize(n, 0.0);
    out
}

/// `y[i] += x[i]`, left to right. Lengths match: both are `hidden`.
fn add_assign(y: &mut [f32], x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    for i in 0..y.len() {
        y[i] += x[i];
    }
}

fn map_op(err: Bf16OpError) -> BlockError {
    match err {
        Bf16OpError::TokenOutOfBounds => BlockError::Token,
        Bf16OpError::Shape | Bf16OpError::RowOutOfBounds => BlockError::Shape,
    }
}

fn ready(
    shape: BlockShape,
    cache: &KvCache,
    work: &BlockWorkspace,
    pos: usize,
) -> Result<(), BlockError> {
    if work.shape != shape || work.attn.max_seq() != cache.max_seq() {
        return Err(BlockError::Shape);
    }
    let kv = cache.shape();
    if kv.num_layers != shape.num_layers
        || kv.num_kv_heads != shape.attn.num_kv_heads
        || kv.head_dim != shape.attn.head_dim
    {
        return Err(BlockError::Shape);
    }
    if pos != cache.len() {
        return Err(BlockError::Position);
    }
    if pos >= cache.max_seq() {
        return Err(BlockError::Cache(KvError::Full));
    }
    Ok(())
}

/// Embed `token`, run every block, write logits, and advance the cache once.
///
/// `pos` must equal `cache.len()`. Each layer stores K/V at that cursor inside
/// [`attention_layer`]. This function does not advance between layers. After
/// the last layer it applies [`FINAL_NORM`], then
/// [`gemv_bf16`](tensor_core::gemv_bf16) of [`EMBED_TOKENS`] (the tied head;
/// there is no `lm_head` lookup), then [`KvCache::advance`] once.
///
/// The returned slice is `work.logits()` and is valid until the next call that
/// borrows `work` mutably.
pub fn forward_token<'a>(
    token: u32,
    pos: usize,
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &mut KvCache,
    work: &'a mut BlockWorkspace,
) -> Result<&'a [f32], BlockError> {
    ready(shape, cache, work, pos)?;
    let hidden = shape.attn.hidden;
    let embed =
        bf16_tensor(map, EMBED_TOKENS, &[shape.vocab, hidden]).map_err(BlockError::Weight)?;
    embed_gather_bf16(&mut work.x, embed.bytes, embed.shape, token).map_err(map_op)?;

    for layer in 0..shape.num_layers {
        let ln =
            bf16_tensor(map, &input_layernorm_key(layer), &[hidden]).map_err(BlockError::Weight)?;
        rmsnorm_bf16_weight(&mut work.normed, ln.bytes, &work.x, shape.rms_eps).map_err(map_op)?;
        attention_layer(
            &mut work.x,
            &work.normed,
            map,
            layer,
            pos,
            shape.attn,
            cache,
            &mut work.attn,
        )
        .map_err(BlockError::Attn)?;

        let post = bf16_tensor(map, &post_attention_layernorm_key(layer), &[hidden])
            .map_err(BlockError::Weight)?;
        rmsnorm_bf16_weight(&mut work.normed, post.bytes, &work.x, shape.rms_eps)
            .map_err(map_op)?;
        let gate_w = bf16_tensor(
            map,
            &gate_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(BlockError::Weight)?;
        let up_w = bf16_tensor(
            map,
            &up_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(BlockError::Weight)?;
        let down_w = bf16_tensor(
            map,
            &down_proj_weight_key(layer),
            &[hidden, shape.intermediate],
        )
        .map_err(BlockError::Weight)?;
        swiglu_bf16(
            &mut work.mlp,
            &work.normed,
            gate_w.bytes,
            up_w.bytes,
            down_w.bytes,
            &mut work.gate,
            &mut work.up,
            &mut work.act,
            SwigluShape {
                hidden,
                intermediate: shape.intermediate,
            },
        )
        .map_err(|_| BlockError::Shape)?;
        add_assign(&mut work.x, &work.mlp);
    }

    let final_ln = bf16_tensor(map, FINAL_NORM, &[hidden]).map_err(BlockError::Weight)?;
    rmsnorm_bf16_weight(&mut work.normed, final_ln.bytes, &work.x, shape.rms_eps)
        .map_err(map_op)?;
    // Tied head: same bytes as the embedding gather. No `lm_head` key.
    gemv_bf16(
        &mut work.logits,
        embed.bytes,
        &work.normed,
        shape.vocab,
        hidden,
    )
    .map_err(map_op)?;
    cache.advance().map_err(BlockError::Cache)?;
    Ok(&work.logits)
}

/// Run every prompt id as one chunk, then advance the cache once per token.
///
/// This is not [`forward_token`] in a loop. One layer projects Q, K, V, the
/// output, and the SwiGLU matrices for the whole chunk ([`gemm_bf16`]),
/// writes that layer's K and V, then scores each query with
/// [`gqa_attn_f32_strided`] and [`GqaMask::Causal`] on the cache stride.
/// Each `(token, row)` dot is still
/// `j = 0..k-1`, multiply then add, so the chunk matches a serial
/// [`forward_token`] on the success path: same KV slots, and the same logits
/// for the last prompt id.
///
/// The tied head runs only on that last id. Earlier prompt logits are unused.
/// Decode stays on [`forward_token`].
///
/// Positions are `cache.len()..`. An empty prompt, a token outside `vocab`,
/// or a missing weight is an error and does not move the cursor or write a
/// slot. The cursor advances only after every layer and the tied head have
/// finished.
pub fn prefill<'a>(
    tokens: &[u32],
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &mut KvCache,
    work: &'a mut BlockWorkspace,
) -> Result<&'a [f32], BlockError> {
    if tokens.is_empty() {
        return Err(BlockError::Shape);
    }
    let n = tokens.len();
    let start = cache.len();
    ready(shape, cache, work, start)?;
    let end = start.checked_add(n).ok_or(BlockError::Shape)?;
    if end > cache.max_seq() {
        return Err(BlockError::Cache(KvError::Full));
    }
    let hidden = shape.attn.hidden;
    let q_dim = shape
        .attn
        .num_q_heads
        .checked_mul(shape.attn.head_dim)
        .ok_or(BlockError::Shape)?;
    let kv_dim = shape
        .attn
        .num_kv_heads
        .checked_mul(shape.attn.head_dim)
        .ok_or(BlockError::Shape)?;
    if shape.attn.head_dim == 0 || shape.attn.head_dim % 2 != 0 {
        return Err(BlockError::Shape);
    }
    for &token in tokens {
        if token as usize >= shape.vocab {
            return Err(BlockError::Token);
        }
    }
    let _ = u32::try_from(end - 1).map_err(|_| BlockError::Position)?;
    preflight_weights(map, shape, hidden, q_dim, kv_dim)?;

    let mut buf = PrefillBuf::new(shape, n, end, hidden, q_dim, kv_dim)?;
    let embed =
        bf16_tensor(map, EMBED_TOKENS, &[shape.vocab, hidden]).map_err(BlockError::Weight)?;
    for (i, &token) in tokens.iter().enumerate() {
        let s = i * hidden;
        embed_gather_bf16(&mut buf.x[s..s + hidden], embed.bytes, embed.shape, token)
            .map_err(map_op)?;
    }

    for layer in 0..shape.num_layers {
        let ln =
            bf16_tensor(map, &input_layernorm_key(layer), &[hidden]).map_err(BlockError::Weight)?;
        norm_all(&mut buf.normed, &buf.x, ln.bytes, n, hidden, shape.rms_eps)?;
        let q_w = bf16_tensor(map, &q_proj_weight_key(layer), &[q_dim, hidden])
            .map_err(BlockError::Weight)?;
        let k_w = bf16_tensor(map, &k_proj_weight_key(layer), &[kv_dim, hidden])
            .map_err(BlockError::Weight)?;
        let v_w = bf16_tensor(map, &v_proj_weight_key(layer), &[kv_dim, hidden])
            .map_err(BlockError::Weight)?;
        let q_b =
            bf16_tensor(map, &q_proj_bias_key(layer), &[q_dim]).map_err(BlockError::Weight)?;
        let k_b =
            bf16_tensor(map, &k_proj_bias_key(layer), &[kv_dim]).map_err(BlockError::Weight)?;
        let v_b =
            bf16_tensor(map, &v_proj_bias_key(layer), &[kv_dim]).map_err(BlockError::Weight)?;
        let o_w = bf16_tensor(map, &o_proj_weight_key(layer), &[hidden, q_dim])
            .map_err(BlockError::Weight)?;
        project(
            &mut buf.q,
            q_w.bytes,
            &buf.normed,
            n,
            q_dim,
            hidden,
            &mut buf.gemm,
        )?;
        project(
            &mut buf.k,
            k_w.bytes,
            &buf.normed,
            n,
            kv_dim,
            hidden,
            &mut buf.gemm,
        )?;
        project(
            &mut buf.v,
            v_w.bytes,
            &buf.normed,
            n,
            kv_dim,
            hidden,
            &mut buf.gemm,
        )?;
        bias_rope_store(
            &mut buf.q,
            &mut buf.k,
            &mut buf.v,
            q_b.bytes,
            k_b.bytes,
            v_b.bytes,
            n,
            q_dim,
            kv_dim,
            start,
            shape.attn.head_dim,
            shape.attn.rope_theta,
            layer,
            cache,
        )?;
        attend_chunk(
            &mut buf.att,
            &buf.q,
            &mut buf.probs,
            cache,
            layer,
            start,
            end,
            n,
            shape,
            q_dim,
        )?;
        project(
            &mut buf.proj,
            o_w.bytes,
            &buf.att,
            n,
            hidden,
            q_dim,
            &mut buf.gemm,
        )?;
        add_tokens(&mut buf.x, &buf.proj, n, hidden);

        let post = bf16_tensor(map, &post_attention_layernorm_key(layer), &[hidden])
            .map_err(BlockError::Weight)?;
        norm_all(
            &mut buf.normed,
            &buf.x,
            post.bytes,
            n,
            hidden,
            shape.rms_eps,
        )?;
        let gate_w = bf16_tensor(
            map,
            &gate_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(BlockError::Weight)?;
        let up_w = bf16_tensor(
            map,
            &up_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(BlockError::Weight)?;
        let down_w = bf16_tensor(
            map,
            &down_proj_weight_key(layer),
            &[hidden, shape.intermediate],
        )
        .map_err(BlockError::Weight)?;
        project(
            &mut buf.gate,
            gate_w.bytes,
            &buf.normed,
            n,
            shape.intermediate,
            hidden,
            &mut buf.gemm,
        )?;
        project(
            &mut buf.up,
            up_w.bytes,
            &buf.normed,
            n,
            shape.intermediate,
            hidden,
            &mut buf.gemm,
        )?;
        for i in 0..n {
            let s = i * shape.intermediate;
            let e = s + shape.intermediate;
            swiglu_mul_f32(&mut buf.act[s..e], &buf.gate[s..e], &buf.up[s..e])
                .map_err(|_| BlockError::Shape)?;
        }
        project(
            &mut buf.proj,
            down_w.bytes,
            &buf.act,
            n,
            hidden,
            shape.intermediate,
            &mut buf.gemm,
        )?;
        add_tokens(&mut buf.x, &buf.proj, n, hidden);
    }

    let final_ln = bf16_tensor(map, FINAL_NORM, &[hidden]).map_err(BlockError::Weight)?;
    let last = (n - 1) * hidden;
    rmsnorm_bf16_weight(
        &mut work.normed,
        final_ln.bytes,
        &buf.x[last..last + hidden],
        shape.rms_eps,
    )
    .map_err(map_op)?;
    work.x.copy_from_slice(&buf.x[last..last + hidden]);
    gemv_bf16(
        &mut work.logits,
        embed.bytes,
        &work.normed,
        shape.vocab,
        hidden,
    )
    .map_err(map_op)?;
    for _ in 0..n {
        cache.advance().map_err(BlockError::Cache)?;
    }
    Ok(&work.logits)
}

struct PrefillBuf {
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    probs: Vec<f32>,
    gemm: Vec<f32>,
}

impl PrefillBuf {
    fn new(
        shape: BlockShape,
        n: usize,
        seq: usize,
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<Self, BlockError> {
        let k_max = hidden.max(q_dim).max(shape.intermediate);
        let gemm_len =
            gemm_bf16_scratch_elems(n, k_max, gemv_bf16_threads()).ok_or(BlockError::Shape)?;
        Ok(Self {
            x: zeros(n.checked_mul(hidden).ok_or(BlockError::Shape)?),
            normed: zeros(n.checked_mul(hidden).ok_or(BlockError::Shape)?),
            q: zeros(n.checked_mul(q_dim).ok_or(BlockError::Shape)?),
            k: zeros(n.checked_mul(kv_dim).ok_or(BlockError::Shape)?),
            v: zeros(n.checked_mul(kv_dim).ok_or(BlockError::Shape)?),
            att: zeros(n.checked_mul(q_dim).ok_or(BlockError::Shape)?),
            proj: zeros(n.checked_mul(hidden).ok_or(BlockError::Shape)?),
            gate: zeros(n.checked_mul(shape.intermediate).ok_or(BlockError::Shape)?),
            up: zeros(n.checked_mul(shape.intermediate).ok_or(BlockError::Shape)?),
            act: zeros(n.checked_mul(shape.intermediate).ok_or(BlockError::Shape)?),
            probs: zeros(seq),
            gemm: zeros(gemm_len),
        })
    }
}

fn preflight_weights(
    map: &impl WeightMap,
    shape: BlockShape,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<(), BlockError> {
    bf16_tensor(map, EMBED_TOKENS, &[shape.vocab, hidden]).map_err(BlockError::Weight)?;
    bf16_tensor(map, FINAL_NORM, &[hidden]).map_err(BlockError::Weight)?;
    for layer in 0..shape.num_layers {
        bf16_tensor(map, &input_layernorm_key(layer), &[hidden]).map_err(BlockError::Weight)?;
        bf16_tensor(map, &q_proj_weight_key(layer), &[q_dim, hidden])
            .map_err(BlockError::Weight)?;
        bf16_tensor(map, &q_proj_bias_key(layer), &[q_dim]).map_err(BlockError::Weight)?;
        bf16_tensor(map, &k_proj_weight_key(layer), &[kv_dim, hidden])
            .map_err(BlockError::Weight)?;
        bf16_tensor(map, &k_proj_bias_key(layer), &[kv_dim]).map_err(BlockError::Weight)?;
        bf16_tensor(map, &v_proj_weight_key(layer), &[kv_dim, hidden])
            .map_err(BlockError::Weight)?;
        bf16_tensor(map, &v_proj_bias_key(layer), &[kv_dim]).map_err(BlockError::Weight)?;
        bf16_tensor(map, &o_proj_weight_key(layer), &[hidden, q_dim])
            .map_err(BlockError::Weight)?;
        bf16_tensor(map, &post_attention_layernorm_key(layer), &[hidden])
            .map_err(BlockError::Weight)?;
        bf16_tensor(
            map,
            &gate_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(BlockError::Weight)?;
        bf16_tensor(
            map,
            &up_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(BlockError::Weight)?;
        bf16_tensor(
            map,
            &down_proj_weight_key(layer),
            &[hidden, shape.intermediate],
        )
        .map_err(BlockError::Weight)?;
    }
    Ok(())
}

fn project(
    y: &mut [f32],
    w: &[u8],
    x: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    scratch: &mut [f32],
) -> Result<(), BlockError> {
    gemm_bf16(y, w, x, batch, m, k, scratch).map_err(map_op)
}

fn norm_all(
    dst: &mut [f32],
    src: &[f32],
    gamma: &[u8],
    n: usize,
    dim: usize,
    eps: f32,
) -> Result<(), BlockError> {
    for i in 0..n {
        let s = i * dim;
        rmsnorm_bf16_weight(&mut dst[s..s + dim], gamma, &src[s..s + dim], eps).map_err(map_op)?;
    }
    Ok(())
}

fn add_tokens(y: &mut [f32], x: &[f32], n: usize, dim: usize) {
    for i in 0..n {
        let s = i * dim;
        add_assign(&mut y[s..s + dim], &x[s..s + dim]);
    }
}

fn bias_rope_store(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    q_b: &[u8],
    k_b: &[u8],
    v_b: &[u8],
    n: usize,
    q_dim: usize,
    kv_dim: usize,
    start: usize,
    head_dim: usize,
    theta: f32,
    layer: usize,
    cache: &mut KvCache,
) -> Result<(), BlockError> {
    for i in 0..n {
        let qs = i * q_dim;
        let ks = i * kv_dim;
        add_bf16_bias(&mut q[qs..qs + q_dim], q_b).map_err(map_op)?;
        add_bf16_bias(&mut k[ks..ks + kv_dim], k_b).map_err(map_op)?;
        add_bf16_bias(&mut v[ks..ks + kv_dim], v_b).map_err(map_op)?;
        let pos = start + i;
        let pos_u = u32::try_from(pos).map_err(|_| BlockError::Position)?;
        rope_f32_inplace(&mut q[qs..qs + q_dim], pos_u, head_dim, theta)
            .map_err(|_| BlockError::Shape)?;
        rope_f32_inplace(&mut k[ks..ks + kv_dim], pos_u, head_dim, theta)
            .map_err(|_| BlockError::Shape)?;
        cache
            .store_at(layer, pos, &k[ks..ks + kv_dim], &v[ks..ks + kv_dim])
            .map_err(BlockError::Cache)?;
    }
    Ok(())
}

fn attend_chunk(
    att: &mut [f32],
    q: &[f32],
    probs: &mut [f32],
    cache: &KvCache,
    layer: usize,
    start: usize,
    end: usize,
    n: usize,
    shape: BlockShape,
    q_dim: usize,
) -> Result<(), BlockError> {
    let layer_k = cache.layer_keys(layer).map_err(BlockError::Cache)?;
    let layer_v = cache.layer_values(layer).map_err(BlockError::Cache)?;
    let stride = cache.max_seq();
    for i in 0..n {
        let s = i * q_dim;
        gqa_attn_f32_strided(
            &mut att[s..s + q_dim],
            &q[s..s + q_dim],
            layer_k,
            layer_v,
            None,
            None,
            &mut probs[..end],
            GqaShape {
                num_q_heads: shape.attn.num_q_heads,
                num_kv_heads: shape.attn.num_kv_heads,
                head_dim: shape.attn.head_dim,
                seq_len: end,
            },
            GqaMask::Causal {
                query_pos: start + i,
            },
            stride,
        )
        .map_err(|_| BlockError::Shape)?;
    }
    Ok(())
}

/// Greedy-decode `steps` tokens from the logits already in `work`.
///
/// Each chosen id is appended to `out` only after its [`forward_token`]
/// succeeds. The next position is `cache.len()` (the prompt length, after
/// [`prefill`]). This does not advance the cache a second time.
pub fn greedy_decode(
    steps: usize,
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &mut KvCache,
    work: &mut BlockWorkspace,
    out: &mut Vec<u32>,
) -> Result<(), BlockError> {
    for _ in 0..steps {
        let token = greedy_token(&work.logits);
        let pos = cache.len();
        forward_token(token, pos, map, shape, cache, work)?;
        out.push(token);
    }
    Ok(())
}

/// One-layer fixture: hidden 2, vocab 4, eps 0.
///
/// Not a Qwen checkpoint. [`forward_token`] of token 0 yields logits
/// `[1, 2, 0, -1]` (greedy id 1). See the unit test for the arithmetic.
/// [`BlockShape::qwen25_1_5b`] is the real width; this is the tiny path the
/// runtime flag runs instead of downloading 1.5B.
pub const fn fixture_shape() -> BlockShape {
    BlockShape {
        attn: AttnShape {
            hidden: 2,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rope_theta: tensor_core::ROPE_THETA_QWEN25,
        },
        intermediate: 2,
        vocab: 4,
        num_layers: 1,
        rms_eps: 0.0,
    }
}

/// In-memory BF16 weights for [`fixture_shape`].
///
/// Token 0's row is `[1, 1]`. The layer's Q/K/V weights are zero, V bias is
/// `[1, 0]`, and `o_proj` copies that V lane into both residual lanes, so the
/// stream becomes `[2, 2]`. Gate weights are zero, so SwiGLU adds nothing.
/// The final gamma is `[1, 0]`, so the tied head sees `[1, 0]` and the logits
/// are the embedding's first column: `[1, 2, 0, -1]`.
pub fn block_fixture() -> Result<MmapWeights, BlockError> {
    let shape = fixture_shape();
    let mut map = MmapWeights::new();
    insert_embed_and_final(&mut map, shape)?;
    fill_layer(
        &mut map,
        0,
        &[0.5, 0.0, 0.0, 0.25],
        &[1.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
    )?;
    Ok(map)
}

fn insert_embed_and_final(map: &mut MmapWeights, shape: BlockShape) -> Result<(), BlockError> {
    let hidden = shape.attn.hidden;
    debug_assert_eq!(hidden, 2);
    debug_assert_eq!(shape.vocab, 4);
    put(
        map,
        EMBED_TOKENS,
        &[shape.vocab, hidden],
        &[1.0, 1.0, 2.0, 0.0, 0.0, 4.0, -1.0, 0.0],
    )?;
    put(map, FINAL_NORM, &[hidden], &[1.0, 0.0])?;
    Ok(())
}

/// Q bias, V bias, `o_proj`, then gate / up / down. Norms are `[1, 1]`.
/// Q/K/V weights and K bias are zero, so V stored in the cache is `v_b`.
fn fill_layer(
    map: &mut MmapWeights,
    layer: usize,
    q_b: &[f32],
    v_b: &[f32],
    o_w: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
) -> Result<(), BlockError> {
    let shape = fixture_shape();
    let hidden = shape.attn.hidden;
    let q_dim = shape.attn.num_q_heads * shape.attn.head_dim;
    let kv_dim = shape.attn.num_kv_heads * shape.attn.head_dim;
    let inter = shape.intermediate;
    put(map, &input_layernorm_key(layer), &[hidden], &[1.0, 1.0])?;
    put(
        map,
        &crate::qwen25_layout::q_proj_weight_key(layer),
        &[q_dim, hidden],
        &zeros(q_dim * hidden),
    )?;
    put(
        map,
        &crate::qwen25_layout::q_proj_bias_key(layer),
        &[q_dim],
        q_b,
    )?;
    put(
        map,
        &crate::qwen25_layout::k_proj_weight_key(layer),
        &[kv_dim, hidden],
        &zeros(kv_dim * hidden),
    )?;
    put(
        map,
        &crate::qwen25_layout::k_proj_bias_key(layer),
        &[kv_dim],
        &zeros(kv_dim),
    )?;
    put(
        map,
        &crate::qwen25_layout::v_proj_weight_key(layer),
        &[kv_dim, hidden],
        &zeros(kv_dim * hidden),
    )?;
    put(
        map,
        &crate::qwen25_layout::v_proj_bias_key(layer),
        &[kv_dim],
        v_b,
    )?;
    put(
        map,
        &crate::qwen25_layout::o_proj_weight_key(layer),
        &[hidden, q_dim],
        o_w,
    )?;
    put(
        map,
        &post_attention_layernorm_key(layer),
        &[hidden],
        &[1.0, 1.0],
    )?;
    put(map, &gate_proj_weight_key(layer), &[inter, hidden], gate)?;
    put(map, &up_proj_weight_key(layer), &[inter, hidden], up)?;
    put(map, &down_proj_weight_key(layer), &[hidden, inter], down)?;
    Ok(())
}

fn put(
    map: &mut MmapWeights,
    name: &str,
    shape: &[usize],
    values: &[f32],
) -> Result<(), BlockError> {
    map.insert_bf16(name, shape, &pack_bf16(values))
        .map_err(|_| BlockError::Shape)
}

/// Little-endian BF16. Fixture values are exact (`f32` low 16 bits are zero).
fn pack_bf16(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for &value in values {
        let bits = value.to_bits();
        debug_assert_eq!(
            bits & 0xFFFF,
            0,
            "fixture value {value} is not exact in bf16"
        );
        let hi = (bits >> 16) as u16;
        bytes.extend_from_slice(&hi.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KvShape;
    use crate::qwen25_layout::{EMBED_TOKENS, HIDDEN_SIZE, TIE_WORD_EMBEDDINGS};
    use tensor_core::{gemv_bf16, rmsnorm_bf16_weight, silu_f32, swiglu_bf16, SwigluShape};
    use weight_loader::{load_f32_tensor, WeightError};

    fn cache_for(shape: BlockShape, max_seq: usize) -> KvCache {
        KvCache::new(KvShape {
            num_layers: shape.num_layers,
            num_kv_heads: shape.attn.num_kv_heads,
            head_dim: shape.attn.head_dim,
            max_seq,
        })
        .unwrap()
    }

    fn run<'a>(
        token: u32,
        pos: usize,
        map: &MmapWeights,
        shape: BlockShape,
        cache: &mut KvCache,
        work: &'a mut BlockWorkspace,
    ) -> &'a [f32] {
        forward_token(token, pos, map, shape, cache, work).unwrap()
    }

    #[test]
    fn qwen25_block_shape_matches_layout() {
        let shape = BlockShape::qwen25_1_5b();
        assert_eq!(shape.num_layers, 28);
        assert_eq!(shape.vocab, VOCAB_SIZE);
        assert_eq!(shape.intermediate, INTERMEDIATE_SIZE);
        assert_eq!(shape.attn.hidden, HIDDEN_SIZE);
        assert_eq!(shape.attn, AttnShape::qwen25_1_5b());
        assert_eq!(shape.rms_eps, RMS_NORM_EPS_QWEN25);
        assert_eq!(shape.rms_eps, 1e-6);
        assert!(TIE_WORD_EMBEDDINGS);
        assert_eq!(EMBED_TOKENS, "model.embed_tokens.weight");
        assert!(!EMBED_TOKENS.contains("lm_head"));

        let tiny = fixture_shape();
        assert_eq!(tiny.num_layers, 1);
        assert_eq!(tiny.vocab, 4);
        assert_eq!(tiny.attn.hidden, 2);
        assert_eq!(tiny.rms_eps, 0.0);
        assert!(BlockWorkspace::new(tiny, 0).is_err());
        assert!(BlockWorkspace::new(
            BlockShape {
                num_layers: 0,
                ..tiny
            },
            2
        )
        .is_err());
        assert!(BlockWorkspace::new(
            BlockShape {
                attn: AttnShape {
                    head_dim: 1,
                    ..tiny.attn
                },
                ..tiny
            },
            2
        )
        .is_err());
    }

    #[test]
    fn one_block_forward_numbers_and_tied_embed_bytes() {
        // Token 0 embeds to [1, 1]. eps 0 and gamma [1, 1] leave that vector
        // (mean square is 1). Q/K/V weights are 0, so V is the bias [1, 0]
        // and the single key's softmax copies it to both query heads.
        // o_proj's first column is 1, so the residual add is [1, 1] and the
        // stream is [2, 2]. Gate weights are 0, so SiLU(gate)⊙up is 0 and
        // the down projection (and any MLP bias) adds nothing.
        // Final gamma [1, 0]: mean square of [2, 2] is 4, inv_rms is 1/2,
        // normed is [1, 0]. Logits are column 0 of the embedding.
        let shape = fixture_shape();
        let mut weights = block_fixture().unwrap();
        // A decoy head and the biases this checkpoint does not use. 8 in the
        // first lane would move [2, 2] off the diagonal and change the logits.
        weights
            .insert_bf16(
                "lm_head.weight",
                &[4, 2],
                &pack_bf16(&[0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        weights
            .insert_bf16(
                "model.layers.0.self_attn.o_proj.bias",
                &[2],
                &pack_bf16(&[8.0, 0.0]),
            )
            .unwrap();
        weights
            .insert_bf16(
                "model.layers.0.mlp.down_proj.bias",
                &[2],
                &pack_bf16(&[8.0, 0.0]),
            )
            .unwrap();
        assert_eq!(
            load_f32_tensor(&weights, EMBED_TOKENS).unwrap_err(),
            WeightError::DTypeMismatch
        );

        let embed = bf16_tensor(&weights, EMBED_TOKENS, &[4, 2]).unwrap();
        assert_eq!(
            embed.bytes.as_ptr(),
            weights.tensor_bytes(EMBED_TOKENS).unwrap().as_ptr()
        );
        let mut from_bytes = [0.0f32; 4];
        gemv_bf16(&mut from_bytes, embed.bytes, &[1.0, 0.0], 4, 2).unwrap();
        assert_eq!(from_bytes, [1.0, 2.0, 0.0, -1.0]);

        let mut cache = cache_for(shape, 4);
        let mut work = BlockWorkspace::new(shape, 4).unwrap();
        let logits = run(0, 0, &weights, shape, &mut cache, &mut work);
        assert_eq!(logits, &[1.0, 2.0, 0.0, -1.0]);
        assert_eq!(logits, &from_bytes);
        assert_eq!(greedy_token(logits), 1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[0.0, 0.0]);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[1.0, 0.0]);
        // One advance, not one per projection: the next slot is still empty.
        assert_eq!(cache.value_head(0, 0, 1).unwrap(), &[0.0, 0.0]);
        assert_eq!(cache.key_head(0, 0, 1).unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn swiglu_residual_changes_the_logits() {
        // Same attention path, so the stream is [2, 2] before the MLP and the
        // post-norm of that vector is [1, 1]. Gate and up each read the first
        // lane only, so h = [silu(1), 0] and down (identity) leaves that.
        // The residual is no longer a scaled [1, 1], and the tied head moves.
        let shape = fixture_shape();
        let mut weights = MmapWeights::new();
        insert_embed_and_final(&mut weights, shape).unwrap();
        fill_layer(
            &mut weights,
            0,
            &[0.5, 0.0, 0.0, 0.25],
            &[1.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

        let mut gate = [0.0f32; 2];
        let mut up = [0.0f32; 2];
        let mut act = [0.0f32; 2];
        let mut mlp = [0.0f32; 2];
        swiglu_bf16(
            &mut mlp,
            &[1.0, 1.0],
            &pack_bf16(&[1.0, 0.0, 0.0, 0.0]),
            &pack_bf16(&[1.0, 0.0, 0.0, 0.0]),
            &pack_bf16(&[1.0, 0.0, 0.0, 1.0]),
            &mut gate,
            &mut up,
            &mut act,
            SwigluShape {
                hidden: 2,
                intermediate: 2,
            },
        )
        .unwrap();
        assert_eq!(mlp[0], silu_f32(1.0));
        assert_eq!(mlp[1], 0.0);
        let residual = [2.0 + mlp[0], 2.0 + mlp[1]];
        let mut normed = [0.0f32; 2];
        rmsnorm_bf16_weight(&mut normed, &pack_bf16(&[1.0, 0.0]), &residual, 0.0).unwrap();
        let embed = weights.tensor_bytes(EMBED_TOKENS).unwrap();
        let mut expect = [0.0f32; 4];
        gemv_bf16(&mut expect, embed, &normed, 4, 2).unwrap();
        assert_ne!(expect, [1.0, 2.0, 0.0, -1.0]);

        let mut cache = cache_for(shape, 2);
        let mut work = BlockWorkspace::new(shape, 2).unwrap();
        let logits = run(0, 0, &weights, shape, &mut cache, &mut work);
        assert_eq!(logits, &expect);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[1.0, 0.0]);
    }

    #[test]
    fn two_layers_advance_once_per_token() {
        // o_proj is zero, so neither layer moves the residual. V biases differ,
        // and both are written at the same cursor. One forward publishes one
        // token: len is 1, and position 1 is still empty on both layers.
        let shape = BlockShape {
            num_layers: 2,
            ..fixture_shape()
        };
        let mut weights = MmapWeights::new();
        insert_embed_and_final(&mut weights, shape).unwrap();
        let o_zero = [0.0f32; 8];
        let mlp_zero = [0.0f32; 4];
        fill_layer(
            &mut weights,
            0,
            &[0.0; 4],
            &[2.0, 0.0],
            &o_zero,
            &mlp_zero,
            &mlp_zero,
            &mlp_zero,
        )
        .unwrap();
        fill_layer(
            &mut weights,
            1,
            &[0.0; 4],
            &[0.0, 8.0],
            &o_zero,
            &mlp_zero,
            &mlp_zero,
            &mlp_zero,
        )
        .unwrap();

        let mut cache = cache_for(shape, 4);
        let mut work = BlockWorkspace::new(shape, 4).unwrap();
        let logits = run(0, 0, &weights, shape, &mut cache, &mut work);
        // Residual stayed on the embedding [1, 1]. Final gamma [1, 0] still
        // yields normed [1, 0], so the tied head matches the one-layer fixture.
        assert_eq!(logits, &[1.0, 2.0, 0.0, -1.0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[2.0, 0.0]);
        assert_eq!(cache.value_head(1, 0, 0).unwrap(), &[0.0, 8.0]);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[0.0, 0.0]);
        assert_eq!(cache.key_head(1, 0, 0).unwrap(), &[0.0, 0.0]);
        assert_eq!(cache.value_head(0, 0, 1).unwrap(), &[0.0, 0.0]);
        assert_eq!(cache.value_head(1, 0, 1).unwrap(), &[0.0, 0.0]);

        let logits = run(1, 1, &weights, shape, &mut cache, &mut work);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[2.0, 0.0]);
        assert_eq!(cache.value_head(1, 0, 0).unwrap(), &[0.0, 8.0]);
        assert_eq!(cache.value_head(0, 0, 1).unwrap(), &[2.0, 0.0]);
        assert_eq!(cache.value_head(1, 0, 1).unwrap(), &[0.0, 8.0]);
        // Token 1 embeds to [2, 0]. Attention and SwiGLU add 0, so the stream
        // stays [2, 0]. Final gamma [1, 0] keeps only the first lane.
        let mut normed = [0.0f32; 2];
        rmsnorm_bf16_weight(&mut normed, &pack_bf16(&[1.0, 0.0]), &[2.0, 0.0], 0.0).unwrap();
        let mut expect = [0.0f32; 4];
        let embed = weights.tensor_bytes(EMBED_TOKENS).unwrap();
        gemv_bf16(&mut expect, embed, &normed, 4, 2).unwrap();
        assert_eq!(logits, &expect);
        assert!(expect[1] > expect[0]);
        assert_eq!(greedy_token(logits), 1);
    }

    #[test]
    fn prefill_and_greedy_decode_on_the_fixture() {
        let shape = fixture_shape();
        let weights = block_fixture().unwrap();
        let mut cache = cache_for(shape, 8);
        let mut work = BlockWorkspace::new(shape, 8).unwrap();
        assert_eq!(
            prefill(&[], &weights, shape, &mut cache, &mut work).unwrap_err(),
            BlockError::Shape
        );
        assert_eq!(cache.len(), 0);

        let logits = prefill(&[0], &weights, shape, &mut cache, &mut work).unwrap();
        assert_eq!(logits, &[1.0, 2.0, 0.0, -1.0]);
        assert_eq!(cache.len(), 1);
        let mut decoded = Vec::new();
        greedy_decode(4, &weights, shape, &mut cache, &mut work, &mut decoded).unwrap();
        assert_eq!(decoded, vec![1, 1, 1, 1]);
        // Prompt plus four generated tokens. Not four advances per layer.
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[1.0, 0.0]);
        assert_eq!(cache.value_head(0, 0, 4).unwrap(), &[1.0, 0.0]);
    }

    #[test]
    fn errors_do_not_advance() {
        let shape = fixture_shape();
        let mut cache = cache_for(shape, 2);
        let mut work = BlockWorkspace::new(shape, 2).unwrap();
        let empty = MmapWeights::new();
        assert_eq!(
            forward_token(0, 0, &empty, shape, &mut cache, &mut work).unwrap_err(),
            BlockError::Weight(QwenBf16Error::Missing)
        );
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[0.0, 0.0]);
        assert_eq!(work.logits(), &[0.0, 0.0, 0.0, 0.0]);

        let weights = block_fixture().unwrap();
        assert_eq!(
            forward_token(0, 1, &weights, shape, &mut cache, &mut work).unwrap_err(),
            BlockError::Position
        );
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[0.0, 0.0]);

        assert_eq!(
            forward_token(9, 0, &weights, shape, &mut cache, &mut work).unwrap_err(),
            BlockError::Token
        );
        assert_eq!(cache.len(), 0);

        let mut wide = BlockWorkspace::new(shape, 4).unwrap();
        assert_eq!(
            forward_token(0, 0, &weights, shape, &mut cache, &mut wide).unwrap_err(),
            BlockError::Shape
        );
        assert_eq!(cache.len(), 0);

        // Layer 0 is present. Layer 1 is not. The first layer stores, the
        // cursor stays, and a complete two-layer map can overwrite that slot.
        let mut partial = MmapWeights::new();
        insert_embed_and_final(&mut partial, shape).unwrap();
        fill_layer(
            &mut partial,
            0,
            &[0.0; 4],
            &[1.0, 0.0],
            &[0.0; 8],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
        )
        .unwrap();
        let two = BlockShape {
            num_layers: 2,
            ..shape
        };
        let mut cache2 = cache_for(two, 2);
        let mut work2 = BlockWorkspace::new(two, 2).unwrap();
        assert_eq!(
            forward_token(0, 0, &partial, two, &mut cache2, &mut work2).unwrap_err(),
            BlockError::Weight(QwenBf16Error::Missing)
        );
        assert_eq!(cache2.len(), 0);
        assert_eq!(cache2.value_head(0, 0, 0).unwrap(), &[1.0, 0.0]);
        assert_eq!(cache2.value_head(1, 0, 0).unwrap(), &[0.0, 0.0]);
        assert_eq!(work2.logits(), &[0.0; 4]);

        let mut full = cache_for(shape, 1);
        let mut work_full = BlockWorkspace::new(shape, 1).unwrap();
        run(0, 0, &weights, shape, &mut full, &mut work_full);
        assert_eq!(full.len(), 1);
        let before = work_full.logits().to_vec();
        assert_eq!(
            forward_token(1, 1, &weights, shape, &mut full, &mut work_full).unwrap_err(),
            BlockError::Cache(KvError::Full)
        );
        assert_eq!(full.len(), 1);
        assert_eq!(full.value_head(0, 0, 0).unwrap(), &[1.0, 0.0]);
        assert_eq!(work_full.logits(), &before);
    }

    fn pat(i: usize) -> f32 {
        const V: [f32; 8] = [0.5, -0.25, 1.0, 0.0, -1.0, 0.125, 2.0, -0.5];
        V[i % 8]
    }

    fn put_pat(map: &mut MmapWeights, name: &str, shape: &[usize], salt: usize) {
        let n = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| pat(i + salt)).collect();
        put(map, name, shape, &vals).unwrap();
    }

    fn rich_map(shape: BlockShape) -> MmapWeights {
        use crate::qwen25_layout::{
            down_proj_weight_key, gate_proj_weight_key, input_layernorm_key, k_proj_bias_key,
            k_proj_weight_key, o_proj_weight_key, post_attention_layernorm_key, q_proj_bias_key,
            q_proj_weight_key, up_proj_weight_key, v_proj_bias_key, v_proj_weight_key, FINAL_NORM,
        };
        let hidden = shape.attn.hidden;
        let q_dim = shape.attn.num_q_heads * shape.attn.head_dim;
        let kv_dim = shape.attn.num_kv_heads * shape.attn.head_dim;
        let inter = shape.intermediate;
        let mut map = MmapWeights::new();
        put_pat(&mut map, EMBED_TOKENS, &[shape.vocab, hidden], 1);
        put_pat(&mut map, FINAL_NORM, &[hidden], 2);
        for layer in 0..shape.num_layers {
            let salt = 10 + layer * 20;
            put_pat(&mut map, &input_layernorm_key(layer), &[hidden], salt);
            put_pat(
                &mut map,
                &q_proj_weight_key(layer),
                &[q_dim, hidden],
                salt + 1,
            );
            put_pat(&mut map, &q_proj_bias_key(layer), &[q_dim], salt + 2);
            put_pat(
                &mut map,
                &k_proj_weight_key(layer),
                &[kv_dim, hidden],
                salt + 3,
            );
            put_pat(&mut map, &k_proj_bias_key(layer), &[kv_dim], salt + 4);
            put_pat(
                &mut map,
                &v_proj_weight_key(layer),
                &[kv_dim, hidden],
                salt + 5,
            );
            put_pat(&mut map, &v_proj_bias_key(layer), &[kv_dim], salt + 6);
            put_pat(
                &mut map,
                &o_proj_weight_key(layer),
                &[hidden, q_dim],
                salt + 7,
            );
            put_pat(
                &mut map,
                &post_attention_layernorm_key(layer),
                &[hidden],
                salt + 8,
            );
            put_pat(
                &mut map,
                &gate_proj_weight_key(layer),
                &[inter, hidden],
                salt + 9,
            );
            put_pat(
                &mut map,
                &up_proj_weight_key(layer),
                &[inter, hidden],
                salt + 10,
            );
            put_pat(
                &mut map,
                &down_proj_weight_key(layer),
                &[hidden, inter],
                salt + 11,
            );
        }
        map
    }

    fn logit_bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn kv_bits(cache: &KvCache) -> Vec<u32> {
        let shape = cache.shape();
        let mut bits = Vec::new();
        for layer in 0..shape.num_layers {
            for head in 0..shape.num_kv_heads {
                for pos in 0..cache.len() {
                    for lane in cache.key_head(layer, head, pos).unwrap() {
                        bits.push(lane.to_bits());
                    }
                    for lane in cache.value_head(layer, head, pos).unwrap() {
                        bits.push(lane.to_bits());
                    }
                }
            }
        }
        bits
    }

    fn serial_prefill(
        tokens: &[u32],
        map: &MmapWeights,
        shape: BlockShape,
        cache: &mut KvCache,
        work: &mut BlockWorkspace,
    ) -> Vec<u32> {
        let start = cache.len();
        let mut last = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let logits = forward_token(token, start + i, map, shape, cache, work).unwrap();
            last = logit_bits(logits);
        }
        last
    }

    #[test]
    fn chunk_prefill_matches_serial_forward_token_bits() {
        let shape = BlockShape {
            attn: AttnShape {
                hidden: 8,
                num_q_heads: 4,
                num_kv_heads: 2,
                head_dim: 2,
                rope_theta: tensor_core::ROPE_THETA_QWEN25,
            },
            intermediate: 6,
            vocab: 5,
            num_layers: 2,
            rms_eps: 1e-6,
        };
        let map = rich_map(shape);
        let prompt = [1u32, 3, 0, 4, 2];
        let mut cache_s = cache_for(shape, 16);
        let mut work_s = BlockWorkspace::new(shape, 16).unwrap();
        let mut cache_c = cache_for(shape, 16);
        let mut work_c = BlockWorkspace::new(shape, 16).unwrap();
        // A committed prefix, so the chunk does not start at position 0.
        forward_token(2, 0, &map, shape, &mut cache_s, &mut work_s).unwrap();
        forward_token(2, 0, &map, shape, &mut cache_c, &mut work_c).unwrap();
        let serial_logits = serial_prefill(&prompt, &map, shape, &mut cache_s, &mut work_s);
        let chunk_logits =
            logit_bits(prefill(&prompt, &map, shape, &mut cache_c, &mut work_c).unwrap());
        assert_eq!(chunk_logits, serial_logits);
        assert_eq!(cache_c.len(), cache_s.len());
        assert_eq!(kv_bits(&cache_c), kv_bits(&cache_s));

        let mut out_s = Vec::new();
        let mut out_c = Vec::new();
        greedy_decode(3, &map, shape, &mut cache_s, &mut work_s, &mut out_s).unwrap();
        greedy_decode(3, &map, shape, &mut cache_c, &mut work_c, &mut out_c).unwrap();
        assert_eq!(out_c, out_s);
        assert_eq!(logit_bits(work_c.logits()), logit_bits(work_s.logits()));
        assert_eq!(kv_bits(&cache_c), kv_bits(&cache_s));

        // Batch tails: 33 is not a multiple of 8, 32, or 64.
        let wide = BlockShape {
            num_layers: 1,
            ..shape
        };
        let map33 = rich_map(wide);
        let tokens: Vec<u32> = (0..33).map(|i| (i % 5) as u32).collect();
        let mut cache_s = cache_for(wide, 40);
        let mut work_s = BlockWorkspace::new(wide, 40).unwrap();
        let mut cache_c = cache_for(wide, 40);
        let mut work_c = BlockWorkspace::new(wide, 40).unwrap();
        let serial_logits = serial_prefill(&tokens, &map33, wide, &mut cache_s, &mut work_s);
        let chunk_logits =
            logit_bits(prefill(&tokens, &map33, wide, &mut cache_c, &mut work_c).unwrap());
        assert_eq!(chunk_logits, serial_logits);
        assert_eq!(cache_c.len(), 33);
        assert_eq!(kv_bits(&cache_c), kv_bits(&cache_s));
    }

    #[test]
    fn chunk_prefill_rejects_before_any_store() {
        let shape = fixture_shape();
        let weights = block_fixture().unwrap();
        let mut cache = cache_for(shape, 4);
        let mut work = BlockWorkspace::new(shape, 4).unwrap();
        assert_eq!(
            prefill(&[0, 9], &weights, shape, &mut cache, &mut work).unwrap_err(),
            BlockError::Token
        );
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[0.0, 0.0]);
        assert_eq!(work.logits(), &[0.0; 4]);

        let mut small = cache_for(shape, 2);
        let mut work_small = BlockWorkspace::new(shape, 2).unwrap();
        assert_eq!(
            prefill(&[0, 0, 0], &weights, shape, &mut small, &mut work_small).unwrap_err(),
            BlockError::Cache(KvError::Full)
        );
        assert_eq!(small.len(), 0);
        assert_eq!(small.value_head(0, 0, 0).unwrap(), &[0.0, 0.0]);

        let mut partial = MmapWeights::new();
        insert_embed_and_final(&mut partial, shape).unwrap();
        let mut cache2 = cache_for(shape, 4);
        let mut work2 = BlockWorkspace::new(shape, 4).unwrap();
        assert_eq!(
            prefill(&[0, 1], &partial, shape, &mut cache2, &mut work2).unwrap_err(),
            BlockError::Weight(QwenBf16Error::Missing)
        );
        assert_eq!(cache2.len(), 0);
        assert_eq!(cache2.value_head(0, 0, 0).unwrap(), &[0.0, 0.0]);
    }
}
