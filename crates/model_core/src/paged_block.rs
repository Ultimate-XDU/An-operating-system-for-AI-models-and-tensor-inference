//! Paged prefill and decode for one sequence.
//!
//! Projections, bias, NeoX RoPE, residual, SwiGLU, and the tied embedding are
//! the same operators as [`crate::prefill`] and [`crate::forward_token`]. The
//! only substitution is where K and V live: a [`PagedCache`] block table, read
//! by [`PagedCache::paged_gqa`]. A full page hit reruns only the last token
//! and does not write. Decode writes the new token into its private slot
//! before scoring it. `len` moves only in [`PagedCache::publish`], after every
//! layer of that chunk has succeeded.
//!
//! [`PagedError::Conflict`] is fatal. The caller [`PagedCache::finish`]es and
//! does not retry, so `admit`'s lock can fall and the prefix can be evicted.
//! There is no speculative decode here.

use alloc::vec::Vec;
use core::fmt;

use tensor_core::{
    add_bf16_bias, embed_gather_bf16, gemm_bf16_scratch_elems, gemv_bf16, gemv_bf16_threads,
    rmsnorm_bf16_weight, rope_f32_inplace, swiglu_bf16, swiglu_mul_f32, Bf16OpError, GqaMask,
    SwigluShape,
};
use weight_loader::WeightMap;

use crate::greedy_token;
use crate::paged_cache::{PagedCache, PagedError, PagedReq};
use crate::qwen25_bf16::{bf16_tensor, QwenBf16Error};
use crate::qwen25_block::{
    add_tokens, norm_all, preflight_weights, project, BlockError, BlockShape, BlockWorkspace,
};
use crate::qwen25_layout::{
    down_proj_weight_key, gate_proj_weight_key, input_layernorm_key, k_proj_bias_key,
    k_proj_weight_key, o_proj_weight_key, post_attention_layernorm_key, q_proj_bias_key,
    q_proj_weight_key, up_proj_weight_key, v_proj_bias_key, v_proj_weight_key, EMBED_TOKENS,
    FINAL_NORM,
};

/// Failure from [`paged_prefill`], [`paged_decode`], or [`paged_greedy`].
///
/// [`PagedBlockError::Cache`] of [`PagedError::Conflict`] means the caller
/// must [`PagedCache::finish`] and must not retry. Other errors leave `len`
/// at the last successful publish. A block this attempt allocated that holds
/// no published token has already been detached, its `refcount` set to 0, and
/// only then pushed on the free list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedBlockError {
    Shape,
    Position,
    Token,
    Weight(QwenBf16Error),
    Cache(PagedError),
}

impl fmt::Display for PagedBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PagedBlockError::Shape => write!(f, "paged block shape does not match the buffers"),
            PagedBlockError::Position => write!(f, "paged block position does not fit"),
            PagedBlockError::Token => write!(f, "token id is out of range"),
            PagedBlockError::Weight(err) => write!(f, "paged block weight: {err}"),
            PagedBlockError::Cache(err) => write!(f, "paged block cache: {err}"),
        }
    }
}

enum KvCommit {
    /// Store this token's K and V, then score the slot just written.
    Write,
    /// Score the slot already in the pool. A bit mismatch is `Conflict`.
    Check,
}

/// `req` is [`PagedCache::admit`] of `tokens`. This allocates the unpublished
/// suffix, runs it in chunks of `chunk_tokens`, and publishes each chunk after
/// that chunk's layers succeed. [`crate::PREFILL_CHUNK`] is the production
/// chunk. A test may pass a smaller count, including the whole suffix.
/// `chunk_tokens == 0` is [`PagedBlockError::Shape`].
///
/// When `req.len()` is already `tokens.len()`, every page is cached. Only the
/// last token is rerun, at its absolute position, and nothing is written. A
/// KV mismatch returns [`PagedError::Conflict`] without allocating. The caller
/// finishes the request and does not retry.
///
/// The returned slice is `work.logits()` and stays valid until the next call
/// that borrows `work` mutably. The tied head runs only on the last prompt token.
pub fn paged_prefill<'a>(
    tokens: &[u32],
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &mut PagedCache,
    req: &mut PagedReq,
    work: &'a mut BlockWorkspace,
    chunk_tokens: usize,
) -> Result<&'a [f32], PagedBlockError> {
    check_prompt(tokens, map, shape, cache, req, work, chunk_tokens)?;
    if req.len() == tokens.len() {
        let pos = req.len() - 1;
        token_forward(
            tokens[pos],
            pos,
            map,
            shape,
            cache,
            req,
            work,
            KvCommit::Check,
        )?;
        return Ok(work.logits());
    }
    reserve_suffix(cache, req, tokens.len())?;
    while req.len() < tokens.len() {
        let start = req.len();
        let n = chunk_tokens.min(tokens.len() - start);
        let last = start + n == tokens.len();
        run_chunk(
            &tokens[start..start + n],
            start,
            last,
            map,
            shape,
            cache,
            req,
            work,
        )?;
    }
    Ok(work.logits())
}

/// One decode step at `req.len()`.
///
/// Allocates a private tail page when `len` is a multiple of the block size.
/// Writes K and V, scores `0..=pos`, then publishes that one token after the
/// tied head. A failure before publish leaves `len` alone and detaches a new
/// block that holds nothing published.
pub fn paged_decode<'a>(
    token: u32,
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &mut PagedCache,
    req: &mut PagedReq,
    work: &'a mut BlockWorkspace,
) -> Result<&'a [f32], PagedBlockError> {
    let pos = req.len();
    check_step(pos, shape, cache, req, work)?;
    if token as usize >= shape.vocab {
        return Err(PagedBlockError::Token);
    }
    let b = cache.shape().block_tokens;
    if pos % b == 0 {
        let blocks = cache.alloc_blocks(1).map_err(PagedBlockError::Cache)?;
        cache.attach(req, &blocks);
    } else if req.blocks().len() <= pos / b {
        return Err(PagedBlockError::Cache(PagedError::Range));
    }
    if let Err(err) = token_forward(token, pos, map, shape, cache, req, work, KvCommit::Write) {
        cache.abort_chunk(req).map_err(PagedBlockError::Cache)?;
        return Err(err);
    }
    match cache.publish(req, &[token]) {
        Ok(()) => Ok(work.logits()),
        Err(PagedError::Conflict) => Err(PagedBlockError::Cache(PagedError::Conflict)),
        Err(err) => {
            cache.abort_chunk(req).map_err(PagedBlockError::Cache)?;
            Err(PagedBlockError::Cache(err))
        }
    }
}

/// Greedy-decode `steps` tokens from the logits already in `work`.
///
/// Each chosen id is appended to `out` only after [`paged_decode`] succeeds.
/// This does not publish a second time.
pub fn paged_greedy(
    steps: usize,
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &mut PagedCache,
    req: &mut PagedReq,
    work: &mut BlockWorkspace,
    out: &mut Vec<u32>,
) -> Result<(), PagedBlockError> {
    for _ in 0..steps {
        let token = greedy_token(work.logits());
        paged_decode(token, map, shape, cache, req, work)?;
        out.push(token);
    }
    Ok(())
}

fn check_prompt(
    tokens: &[u32],
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &PagedCache,
    req: &PagedReq,
    work: &BlockWorkspace,
    chunk_tokens: usize,
) -> Result<(), PagedBlockError> {
    if tokens.is_empty() || chunk_tokens == 0 {
        return Err(PagedBlockError::Shape);
    }
    check_step(tokens.len() - 1, shape, cache, req, work)?;
    if tokens.len() > work.max_seq() {
        return Err(PagedBlockError::Cache(PagedError::Full));
    }
    if req.len() > tokens.len() || req.tokens() != &tokens[..req.len()] {
        return Err(PagedBlockError::Shape);
    }
    let b = cache.shape().block_tokens;
    let pages = if req.len() == 0 { 0 } else { req.len() / b };
    if req.len() % b != 0 || req.blocks().len() != pages {
        return Err(PagedBlockError::Shape);
    }
    for &token in tokens {
        if token as usize >= shape.vocab {
            return Err(PagedBlockError::Token);
        }
    }
    let _ = u32::try_from(tokens.len() - 1).map_err(|_| PagedBlockError::Position)?;
    let (q_dim, kv_dim) = dims(shape)?;
    preflight_weights(map, shape, shape.attn.hidden, q_dim, kv_dim).map_err(map_block)
}

fn check_step(
    pos: usize,
    shape: BlockShape,
    cache: &PagedCache,
    req: &PagedReq,
    work: &BlockWorkspace,
) -> Result<(), PagedBlockError> {
    if work.shape() != shape {
        return Err(PagedBlockError::Shape);
    }
    let page = cache.shape();
    if page.num_layers != shape.num_layers
        || page.num_kv_heads != shape.attn.num_kv_heads
        || page.head_dim != shape.attn.head_dim
        || page.block_tokens == 0
    {
        return Err(PagedBlockError::Shape);
    }
    if shape.attn.head_dim % 2 != 0 || shape.attn.num_q_heads % shape.attn.num_kv_heads != 0 {
        return Err(PagedBlockError::Shape);
    }
    if req.tokens().len() != req.len() {
        return Err(PagedBlockError::Shape);
    }
    if pos >= work.max_seq() {
        return Err(PagedBlockError::Cache(PagedError::Full));
    }
    let _ = u32::try_from(pos).map_err(|_| PagedBlockError::Position)?;
    dims(shape)?;
    Ok(())
}

fn dims(shape: BlockShape) -> Result<(usize, usize), PagedBlockError> {
    let q_dim = shape
        .attn
        .num_q_heads
        .checked_mul(shape.attn.head_dim)
        .ok_or(PagedBlockError::Shape)?;
    let kv_dim = shape
        .attn
        .num_kv_heads
        .checked_mul(shape.attn.head_dim)
        .ok_or(PagedBlockError::Shape)?;
    Ok((q_dim, kv_dim))
}

fn reserve_suffix(
    cache: &mut PagedCache,
    req: &mut PagedReq,
    total: usize,
) -> Result<(), PagedBlockError> {
    let pages = total.div_ceil(cache.shape().block_tokens);
    let have = req.blocks().len();
    if have > pages {
        return Err(PagedBlockError::Shape);
    }
    if have == pages {
        return Ok(());
    }
    let extra = cache
        .alloc_blocks(pages - have)
        .map_err(PagedBlockError::Cache)?;
    cache.attach(req, &extra);
    Ok(())
}

fn run_chunk(
    new_tokens: &[u32],
    start: usize,
    last: bool,
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &mut PagedCache,
    req: &mut PagedReq,
    work: &mut BlockWorkspace,
) -> Result<(), PagedBlockError> {
    let result = run_chunk_body(new_tokens, start, last, map, shape, cache, req, work);
    if result.is_err() && !is_conflict(&result) {
        cache.abort_chunk(req).map_err(PagedBlockError::Cache)?;
    }
    result
}

fn is_conflict(result: &Result<(), PagedBlockError>) -> bool {
    matches!(result, Err(PagedBlockError::Cache(PagedError::Conflict)))
}

fn run_chunk_body(
    new_tokens: &[u32],
    start: usize,
    last: bool,
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &mut PagedCache,
    req: &mut PagedReq,
    work: &mut BlockWorkspace,
) -> Result<(), PagedBlockError> {
    let n = new_tokens.len();
    let hidden = shape.attn.hidden;
    let (q_dim, kv_dim) = dims(shape)?;
    let end = start + n;
    let mut buf = ChunkBuf::new(shape, n, end, hidden, q_dim, kv_dim)?;
    let embed =
        bf16_tensor(map, EMBED_TOKENS, &[shape.vocab, hidden]).map_err(PagedBlockError::Weight)?;
    for (i, &token) in new_tokens.iter().enumerate() {
        let s = i * hidden;
        embed_gather_bf16(&mut buf.x[s..s + hidden], embed.bytes, embed.shape, token)
            .map_err(map_op)?;
    }

    for layer in 0..shape.num_layers {
        let ln = bf16_tensor(map, &input_layernorm_key(layer), &[hidden])
            .map_err(PagedBlockError::Weight)?;
        norm_all(&mut buf.normed, &buf.x, ln.bytes, n, hidden, shape.rms_eps).map_err(map_block)?;
        let q_w = bf16_tensor(map, &q_proj_weight_key(layer), &[q_dim, hidden])
            .map_err(PagedBlockError::Weight)?;
        let k_w = bf16_tensor(map, &k_proj_weight_key(layer), &[kv_dim, hidden])
            .map_err(PagedBlockError::Weight)?;
        let v_w = bf16_tensor(map, &v_proj_weight_key(layer), &[kv_dim, hidden])
            .map_err(PagedBlockError::Weight)?;
        let q_b =
            bf16_tensor(map, &q_proj_bias_key(layer), &[q_dim]).map_err(PagedBlockError::Weight)?;
        let k_b = bf16_tensor(map, &k_proj_bias_key(layer), &[kv_dim])
            .map_err(PagedBlockError::Weight)?;
        let v_b = bf16_tensor(map, &v_proj_bias_key(layer), &[kv_dim])
            .map_err(PagedBlockError::Weight)?;
        let o_w = bf16_tensor(map, &o_proj_weight_key(layer), &[hidden, q_dim])
            .map_err(PagedBlockError::Weight)?;
        project(
            &mut buf.q,
            q_w.bytes,
            &buf.normed,
            n,
            q_dim,
            hidden,
            &mut buf.gemm,
        )
        .map_err(map_block)?;
        project(
            &mut buf.k,
            k_w.bytes,
            &buf.normed,
            n,
            kv_dim,
            hidden,
            &mut buf.gemm,
        )
        .map_err(map_block)?;
        project(
            &mut buf.v,
            v_w.bytes,
            &buf.normed,
            n,
            kv_dim,
            hidden,
            &mut buf.gemm,
        )
        .map_err(map_block)?;
        bias_rope_write(
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
            req,
        )?;
        attend_chunk(
            &mut buf.att,
            &buf.q,
            &mut buf.probs,
            cache,
            req,
            layer,
            start,
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
        )
        .map_err(map_block)?;
        add_tokens(&mut buf.x, &buf.proj, n, hidden);

        let post = bf16_tensor(map, &post_attention_layernorm_key(layer), &[hidden])
            .map_err(PagedBlockError::Weight)?;
        norm_all(
            &mut buf.normed,
            &buf.x,
            post.bytes,
            n,
            hidden,
            shape.rms_eps,
        )
        .map_err(map_block)?;
        let gate_w = bf16_tensor(
            map,
            &gate_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(PagedBlockError::Weight)?;
        let up_w = bf16_tensor(
            map,
            &up_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(PagedBlockError::Weight)?;
        let down_w = bf16_tensor(
            map,
            &down_proj_weight_key(layer),
            &[hidden, shape.intermediate],
        )
        .map_err(PagedBlockError::Weight)?;
        project(
            &mut buf.gate,
            gate_w.bytes,
            &buf.normed,
            n,
            shape.intermediate,
            hidden,
            &mut buf.gemm,
        )
        .map_err(map_block)?;
        project(
            &mut buf.up,
            up_w.bytes,
            &buf.normed,
            n,
            shape.intermediate,
            hidden,
            &mut buf.gemm,
        )
        .map_err(map_block)?;
        for i in 0..n {
            let s = i * shape.intermediate;
            let e = s + shape.intermediate;
            swiglu_mul_f32(&mut buf.act[s..e], &buf.gate[s..e], &buf.up[s..e])
                .map_err(|_| PagedBlockError::Shape)?;
        }
        project(
            &mut buf.proj,
            down_w.bytes,
            &buf.act,
            n,
            hidden,
            shape.intermediate,
            &mut buf.gemm,
        )
        .map_err(map_block)?;
        add_tokens(&mut buf.x, &buf.proj, n, hidden);
    }

    if last {
        let final_ln = bf16_tensor(map, FINAL_NORM, &[hidden]).map_err(PagedBlockError::Weight)?;
        let at = (n - 1) * hidden;
        write_tied(
            work,
            embed.bytes,
            final_ln.bytes,
            shape.rms_eps,
            &buf.x[at..at + hidden],
            shape.vocab,
            hidden,
        )?;
    }
    cache
        .publish(req, new_tokens)
        .map_err(PagedBlockError::Cache)?;
    Ok(())
}

fn bias_rope_write(
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
    cache: &mut PagedCache,
    req: &PagedReq,
) -> Result<(), PagedBlockError> {
    for i in 0..n {
        let qs = i * q_dim;
        let ks = i * kv_dim;
        add_bf16_bias(&mut q[qs..qs + q_dim], q_b).map_err(map_op)?;
        add_bf16_bias(&mut k[ks..ks + kv_dim], k_b).map_err(map_op)?;
        add_bf16_bias(&mut v[ks..ks + kv_dim], v_b).map_err(map_op)?;
        let pos = start + i;
        let pos_u = u32::try_from(pos).map_err(|_| PagedBlockError::Position)?;
        rope_f32_inplace(&mut q[qs..qs + q_dim], pos_u, head_dim, theta)
            .map_err(|_| PagedBlockError::Shape)?;
        rope_f32_inplace(&mut k[ks..ks + kv_dim], pos_u, head_dim, theta)
            .map_err(|_| PagedBlockError::Shape)?;
        cache
            .write(req, layer, pos, &k[ks..ks + kv_dim], &v[ks..ks + kv_dim])
            .map_err(PagedBlockError::Cache)?;
    }
    Ok(())
}

fn attend_chunk(
    att: &mut [f32],
    q: &[f32],
    probs: &mut [f32],
    cache: &PagedCache,
    req: &PagedReq,
    layer: usize,
    start: usize,
    n: usize,
    shape: BlockShape,
    q_dim: usize,
) -> Result<(), PagedBlockError> {
    for i in 0..n {
        let pos = start + i;
        let seq = pos + 1;
        let s = i * q_dim;
        // `seq = query_pos + 1` and `Full` is the same visible set as a longer
        // buffer with `Causal { query_pos }`. Keys past `pos` are not loaded.
        cache
            .paged_gqa(
                req,
                layer,
                &q[s..s + q_dim],
                &mut att[s..s + q_dim],
                &mut probs[..seq],
                shape.attn.num_q_heads,
                seq,
                GqaMask::Full,
            )
            .map_err(PagedBlockError::Cache)?;
    }
    Ok(())
}

fn token_forward(
    token: u32,
    pos: usize,
    map: &impl WeightMap,
    shape: BlockShape,
    cache: &mut PagedCache,
    req: &mut PagedReq,
    work: &mut BlockWorkspace,
    commit: KvCommit,
) -> Result<(), PagedBlockError> {
    let hidden = shape.attn.hidden;
    let (q_dim, kv_dim) = dims(shape)?;
    let pos_u = u32::try_from(pos).map_err(|_| PagedBlockError::Position)?;
    let seq = pos + 1;
    let embed =
        bf16_tensor(map, EMBED_TOKENS, &[shape.vocab, hidden]).map_err(PagedBlockError::Weight)?;
    let (blk, attn) = work.parts();
    embed_gather_bf16(blk.x, embed.bytes, embed.shape, token).map_err(map_op)?;

    for layer in 0..shape.num_layers {
        let ln = bf16_tensor(map, &input_layernorm_key(layer), &[hidden])
            .map_err(PagedBlockError::Weight)?;
        rmsnorm_bf16_weight(blk.normed, ln.bytes, blk.x, shape.rms_eps).map_err(map_op)?;
        let q_w = bf16_tensor(map, &q_proj_weight_key(layer), &[q_dim, hidden])
            .map_err(PagedBlockError::Weight)?;
        let k_w = bf16_tensor(map, &k_proj_weight_key(layer), &[kv_dim, hidden])
            .map_err(PagedBlockError::Weight)?;
        let v_w = bf16_tensor(map, &v_proj_weight_key(layer), &[kv_dim, hidden])
            .map_err(PagedBlockError::Weight)?;
        let q_b =
            bf16_tensor(map, &q_proj_bias_key(layer), &[q_dim]).map_err(PagedBlockError::Weight)?;
        let k_b = bf16_tensor(map, &k_proj_bias_key(layer), &[kv_dim])
            .map_err(PagedBlockError::Weight)?;
        let v_b = bf16_tensor(map, &v_proj_bias_key(layer), &[kv_dim])
            .map_err(PagedBlockError::Weight)?;
        let o_w = bf16_tensor(map, &o_proj_weight_key(layer), &[hidden, q_dim])
            .map_err(PagedBlockError::Weight)?;
        gemv_bf16(attn.q, q_w.bytes, blk.normed, q_dim, hidden).map_err(map_op)?;
        add_bf16_bias(attn.q, q_b.bytes).map_err(map_op)?;
        gemv_bf16(attn.k, k_w.bytes, blk.normed, kv_dim, hidden).map_err(map_op)?;
        add_bf16_bias(attn.k, k_b.bytes).map_err(map_op)?;
        gemv_bf16(attn.v, v_w.bytes, blk.normed, kv_dim, hidden).map_err(map_op)?;
        add_bf16_bias(attn.v, v_b.bytes).map_err(map_op)?;
        rope_f32_inplace(attn.q, pos_u, shape.attn.head_dim, shape.attn.rope_theta)
            .map_err(|_| PagedBlockError::Shape)?;
        rope_f32_inplace(attn.k, pos_u, shape.attn.head_dim, shape.attn.rope_theta)
            .map_err(|_| PagedBlockError::Shape)?;
        match commit {
            KvCommit::Write => {
                cache
                    .write(req, layer, pos, attn.k, attn.v)
                    .map_err(PagedBlockError::Cache)?;
            }
            KvCommit::Check => {
                if !kv_matches(cache, req, layer, pos, attn.k, attn.v)? {
                    return Err(PagedBlockError::Cache(PagedError::Conflict));
                }
            }
        }
        cache
            .paged_gqa(
                req,
                layer,
                attn.q,
                attn.att,
                &mut attn.probs[..seq],
                shape.attn.num_q_heads,
                seq,
                GqaMask::Full,
            )
            .map_err(PagedBlockError::Cache)?;
        gemv_bf16(attn.proj, o_w.bytes, attn.att, hidden, q_dim).map_err(map_op)?;
        add_assign(blk.x, attn.proj);

        let post = bf16_tensor(map, &post_attention_layernorm_key(layer), &[hidden])
            .map_err(PagedBlockError::Weight)?;
        rmsnorm_bf16_weight(blk.normed, post.bytes, blk.x, shape.rms_eps).map_err(map_op)?;
        let gate_w = bf16_tensor(
            map,
            &gate_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(PagedBlockError::Weight)?;
        let up_w = bf16_tensor(
            map,
            &up_proj_weight_key(layer),
            &[shape.intermediate, hidden],
        )
        .map_err(PagedBlockError::Weight)?;
        let down_w = bf16_tensor(
            map,
            &down_proj_weight_key(layer),
            &[hidden, shape.intermediate],
        )
        .map_err(PagedBlockError::Weight)?;
        swiglu_bf16(
            blk.mlp,
            blk.normed,
            gate_w.bytes,
            up_w.bytes,
            down_w.bytes,
            blk.gate,
            blk.up,
            blk.act,
            SwigluShape {
                hidden,
                intermediate: shape.intermediate,
            },
        )
        .map_err(|_| PagedBlockError::Shape)?;
        add_assign(blk.x, blk.mlp);
    }

    let final_ln = bf16_tensor(map, FINAL_NORM, &[hidden]).map_err(PagedBlockError::Weight)?;
    rmsnorm_bf16_weight(blk.normed, final_ln.bytes, blk.x, shape.rms_eps).map_err(map_op)?;
    gemv_bf16(blk.logits, embed.bytes, blk.normed, shape.vocab, hidden).map_err(map_op)?;
    Ok(())
}

fn kv_matches(
    cache: &PagedCache,
    req: &PagedReq,
    layer: usize,
    pos: usize,
    k: &[f32],
    v: &[f32],
) -> Result<bool, PagedBlockError> {
    let kv_heads = cache.shape().num_kv_heads;
    let head_dim = cache.shape().head_dim;
    for head in 0..kv_heads {
        let s = head * head_dim;
        let got_k = cache
            .token_k(req, layer, head, pos)
            .map_err(PagedBlockError::Cache)?;
        let got_v = cache
            .token_v(req, layer, head, pos)
            .map_err(PagedBlockError::Cache)?;
        if !same_bits(got_k, &k[s..s + head_dim]) || !same_bits(got_v, &v[s..s + head_dim]) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn write_tied(
    work: &mut BlockWorkspace,
    embed: &[u8],
    gamma: &[u8],
    eps: f32,
    residual: &[f32],
    vocab: usize,
    hidden: usize,
) -> Result<(), PagedBlockError> {
    let (blk, _) = work.parts();
    blk.x.copy_from_slice(residual);
    rmsnorm_bf16_weight(blk.normed, gamma, blk.x, eps).map_err(map_op)?;
    gemv_bf16(blk.logits, embed, blk.normed, vocab, hidden).map_err(map_op)?;
    Ok(())
}

fn add_assign(y: &mut [f32], x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    for i in 0..y.len() {
        y[i] += x[i];
    }
}

fn map_op(err: Bf16OpError) -> PagedBlockError {
    match err {
        Bf16OpError::TokenOutOfBounds => PagedBlockError::Token,
        Bf16OpError::Shape | Bf16OpError::RowOutOfBounds => PagedBlockError::Shape,
    }
}

fn map_block(err: BlockError) -> PagedBlockError {
    match err {
        BlockError::Shape | BlockError::Attn(_) | BlockError::Cache(_) => PagedBlockError::Shape,
        BlockError::Position => PagedBlockError::Position,
        BlockError::Token => PagedBlockError::Token,
        BlockError::Weight(err) => PagedBlockError::Weight(err),
    }
}

struct ChunkBuf {
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

impl ChunkBuf {
    fn new(
        shape: BlockShape,
        n: usize,
        seq: usize,
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<Self, PagedBlockError> {
        let k_max = hidden.max(q_dim).max(shape.intermediate);
        let gemm_len =
            gemm_bf16_scratch_elems(n, k_max, gemv_bf16_threads()).ok_or(PagedBlockError::Shape)?;
        Ok(Self {
            x: zeros(n.checked_mul(hidden).ok_or(PagedBlockError::Shape)?),
            normed: zeros(n.checked_mul(hidden).ok_or(PagedBlockError::Shape)?),
            q: zeros(n.checked_mul(q_dim).ok_or(PagedBlockError::Shape)?),
            k: zeros(n.checked_mul(kv_dim).ok_or(PagedBlockError::Shape)?),
            v: zeros(n.checked_mul(kv_dim).ok_or(PagedBlockError::Shape)?),
            att: zeros(n.checked_mul(q_dim).ok_or(PagedBlockError::Shape)?),
            proj: zeros(n.checked_mul(hidden).ok_or(PagedBlockError::Shape)?),
            gate: zeros(
                n.checked_mul(shape.intermediate)
                    .ok_or(PagedBlockError::Shape)?,
            ),
            up: zeros(
                n.checked_mul(shape.intermediate)
                    .ok_or(PagedBlockError::Shape)?,
            ),
            act: zeros(
                n.checked_mul(shape.intermediate)
                    .ok_or(PagedBlockError::Shape)?,
            ),
            probs: zeros(seq),
            gemm: zeros(gemm_len),
        })
    }
}

fn zeros(n: usize) -> Vec<f32> {
    let mut out = Vec::new();
    out.resize(n, 0.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{KvCache, KvShape};
    use crate::paged_cache::PagedShape;
    use crate::qwen25_block::{
        fill_layer, fill_layer_kv, fixture_shape, greedy_decode, insert_embed_and_final, prefill,
        BlockShape,
    };
    use crate::qwen25_layout::input_layernorm_key;
    use core::cell::Cell;
    use weight_loader::{MmapWeights, WeightError, WeightMap};

    fn pshape(shape: BlockShape, block: usize) -> PagedShape {
        PagedShape {
            num_layers: shape.num_layers,
            num_kv_heads: shape.attn.num_kv_heads,
            head_dim: shape.attn.head_dim,
            block_tokens: block,
        }
    }

    fn assert_bits(got: &[f32], expect: &[f32]) {
        assert_eq!(got.len(), expect.len());
        for (i, (x, y)) in got.iter().zip(expect).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "lane {i}: {x} vs {y}");
        }
    }

    fn assert_kv(cache: &PagedCache, req: &PagedReq, dense: &KvCache, seq: usize) {
        let layers = cache.shape().num_layers;
        let heads = cache.shape().num_kv_heads;
        for layer in 0..layers {
            for t in 0..seq {
                for head in 0..heads {
                    assert_bits(
                        cache.token_k(req, layer, head, t).unwrap(),
                        dense.key_head(layer, head, t).unwrap(),
                    );
                    assert_bits(
                        cache.token_v(req, layer, head, t).unwrap(),
                        dense.value_head(layer, head, t).unwrap(),
                    );
                }
            }
        }
    }

    fn dense_run(
        tokens: &[u32],
        map: &MmapWeights,
        shape: BlockShape,
        max_seq: usize,
    ) -> (Vec<f32>, KvCache, BlockWorkspace) {
        let mut cache = KvCache::new(KvShape {
            num_layers: shape.num_layers,
            num_kv_heads: shape.attn.num_kv_heads,
            head_dim: shape.attn.head_dim,
            max_seq,
        })
        .unwrap();
        let mut work = BlockWorkspace::new(shape, max_seq).unwrap();
        let logits = prefill(tokens, map, shape, &mut cache, &mut work)
            .unwrap()
            .to_vec();
        (logits, cache, work)
    }

    fn open(
        tokens: &[u32],
        map: &MmapWeights,
        shape: BlockShape,
        cache: &mut PagedCache,
        work: &mut BlockWorkspace,
        chunk: usize,
    ) -> (Vec<f32>, PagedReq) {
        let mut req = cache.admit(tokens).unwrap();
        let logits = paged_prefill(tokens, map, shape, cache, &mut req, work, chunk)
            .unwrap()
            .to_vec();
        (logits, req)
    }

    #[test]
    fn paged_prefill_matches_dense_including_a_one_token_chunk() {
        let shape = fixture_shape();
        let weights = crate::qwen25_block::block_fixture().unwrap();
        let tokens = [0u32, 1, 0];
        let (dense_logits, dense, _) = dense_run(&tokens, &weights, shape, 8);
        assert_eq!(dense_logits, [1.0, 2.0, 0.0, -1.0]);

        for chunk in [1usize, 3, 8] {
            let mut cache = PagedCache::new(pshape(shape, 2), 4).unwrap();
            let mut work = BlockWorkspace::new(shape, 8).unwrap();
            let (logits, req) = open(&tokens, &weights, shape, &mut cache, &mut work, chunk);
            assert_bits(&logits, &dense_logits);
            assert_kv(&cache, &req, &dense, tokens.len());
            cache.finish(req).unwrap();
        }
    }

    #[test]
    fn prefix_hit_and_full_page_rerun_match_dense_greedy() {
        let shape = fixture_shape();
        let weights = crate::qwen25_block::block_fixture().unwrap();
        // B = 2. `[0, 1]` is one full page. `[0, 1, 0]` leaves a private tail
        // that finish drops; the second admit recomputes that token.
        for tokens in [vec![0u32, 1], vec![0, 1, 0]] {
            let (dense_logits, mut dense, mut dense_work) = dense_run(&tokens, &weights, shape, 8);
            let mut decoded = Vec::new();
            greedy_decode(
                3,
                &weights,
                shape,
                &mut dense,
                &mut dense_work,
                &mut decoded,
            )
            .unwrap();

            let mut cache = PagedCache::new(pshape(shape, 2), 8).unwrap();
            let mut work = BlockWorkspace::new(shape, 8).unwrap();
            let (logits, req) = open(&tokens, &weights, shape, &mut cache, &mut work, 8);
            assert_bits(&logits, &dense_logits);
            cache.finish(req).unwrap();

            let mut req = cache.admit(&tokens).unwrap();
            let again = paged_prefill(&tokens, &weights, shape, &mut cache, &mut req, &mut work, 8)
                .unwrap();
            assert_bits(again, &dense_logits);
            // Rebuild a dense cache of the prompt alone for the KV compare.
            // Greedy above advanced the first dense cache past the prompt.
            let (_, prompt_dense, _) = dense_run(&tokens, &weights, shape, 8);
            assert_kv(&cache, &req, &prompt_dense, tokens.len());

            let mut got = Vec::new();
            paged_greedy(
                3, &weights, shape, &mut cache, &mut req, &mut work, &mut got,
            )
            .unwrap();
            assert_eq!(got, decoded);
            assert_bits(work.logits(), dense_work.logits());
            cache.finish(req).unwrap();
        }
    }

    #[test]
    fn two_fresh_pools_match_logits_tokens_and_block_ids() {
        let shape = fixture_shape();
        let weights = crate::qwen25_block::block_fixture().unwrap();
        let tokens = [0u32, 1, 0];
        let mut ids = Vec::new();
        let mut logits = Vec::new();
        let mut generated = Vec::new();
        for _ in 0..2 {
            let mut cache = PagedCache::new(pshape(shape, 2), 8).unwrap();
            let mut work = BlockWorkspace::new(shape, 8).unwrap();
            let (first, mut req) = open(&tokens, &weights, shape, &mut cache, &mut work, 2);
            logits.push(first);
            ids.push(req.blocks().to_vec());
            let mut out = Vec::new();
            paged_greedy(
                3, &weights, shape, &mut cache, &mut req, &mut work, &mut out,
            )
            .unwrap();
            generated.push(out);
            cache.finish(req).unwrap();
        }
        assert_bits(&logits[0], &logits[1]);
        assert_eq!(ids[0], ids[1]);
        assert_eq!(generated[0], generated[1]);
    }

    #[test]
    fn rope_after_a_page_hit_uses_the_absolute_position() {
        let shape = fixture_shape();
        let mut weights = MmapWeights::new();
        insert_embed_and_final(&mut weights, shape).unwrap();
        fill_layer_kv(
            &mut weights,
            0,
            &[0.5, 0.0, 0.0, 0.25],
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
        )
        .unwrap();
        let tokens = [0u32, 1, 2];
        let (dense_logits, dense, _) = dense_run(&tokens, &weights, shape, 8);
        assert_ne!(
            dense.key_head(0, 0, 0).unwrap(),
            dense.key_head(0, 0, 2).unwrap(),
            "position 0 and 2 must rotate the non-zero K differently"
        );

        let mut cache = PagedCache::new(pshape(shape, 2), 4).unwrap();
        let mut work = BlockWorkspace::new(shape, 8).unwrap();
        let (_prefix, prefix_req) = open(&[0, 1], &weights, shape, &mut cache, &mut work, 2);
        cache.finish(prefix_req).unwrap();
        let mut req = cache.admit(&tokens).unwrap();
        assert_eq!(req.len(), 2);
        let logits =
            paged_prefill(&tokens, &weights, shape, &mut cache, &mut req, &mut work, 2).unwrap();
        assert_bits(logits, &dense_logits);
        assert_kv(&cache, &req, &dense, tokens.len());
        cache.finish(req).unwrap();
    }

    #[test]
    fn third_layer_failure_detaches_the_new_block_and_retry_matches() {
        let shape = BlockShape {
            num_layers: 3,
            ..fixture_shape()
        };
        let mut weights = MmapWeights::new();
        insert_embed_and_final(&mut weights, shape).unwrap();
        let o_zero = [0.0f32; 8];
        let mlp_zero = [0.0f32; 4];
        for (layer, v_b) in [(0usize, [1.0, 0.0]), (1, [0.0, 1.0]), (2, [0.5, 0.0])] {
            fill_layer(
                &mut weights,
                layer,
                &[0.0; 4],
                &v_b,
                &o_zero,
                &mlp_zero,
                &mlp_zero,
                &mlp_zero,
            )
            .unwrap();
        }
        let prompt = [0u32, 1, 0, 1];
        let mut cache = PagedCache::new(pshape(shape, 2), 4).unwrap();
        let mut work = BlockWorkspace::new(shape, 8).unwrap();
        let (_first, first) = open(&[0, 1], &weights, shape, &mut cache, &mut work, 4);
        let prefix = first.blocks()[0];
        cache.finish(first).unwrap();

        let mut req = cache.admit(&prompt).unwrap();
        assert_eq!(req.len(), 2);
        assert_eq!(req.blocks(), &[prefix]);
        let snap: Vec<(Vec<u32>, Vec<u32>)> = (0..shape.num_layers)
            .map(|layer| {
                let k = cache
                    .token_k(&req, layer, 0, 0)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let v = cache
                    .token_v(&req, layer, 0, 0)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                (k, v)
            })
            .collect();
        let free_before = cache.free_len();
        let held = req.blocks().to_vec();

        struct FailSecond<'a> {
            inner: &'a MmapWeights,
            key: String,
            hits: Cell<u32>,
        }
        impl WeightMap for FailSecond<'_> {
            fn tensor_bytes(&self, name: &str) -> Result<&[u8], WeightError> {
                if name == self.key {
                    let n = self.hits.get();
                    self.hits.set(n + 1);
                    if n >= 1 {
                        return Err(WeightError::NotFound);
                    }
                }
                self.inner.tensor_bytes(name)
            }
            fn tensor_info(&self, name: &str) -> Result<&weight_loader::TensorInfo, WeightError> {
                self.inner.tensor_info(name)
            }
        }
        let fail = FailSecond {
            inner: &weights,
            key: input_layernorm_key(2),
            hits: Cell::new(0),
        };
        let err = paged_prefill(&prompt, &fail, shape, &mut cache, &mut req, &mut work, 4);
        assert_eq!(err, Err(PagedBlockError::Weight(QwenBf16Error::Missing)));
        assert_eq!(req.len(), 2);
        assert_eq!(req.blocks(), &held);
        assert_eq!(cache.free_len(), free_before);
        for layer in 0..shape.num_layers {
            let k: Vec<u32> = cache
                .token_k(&req, layer, 0, 0)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let v: Vec<u32> = cache
                .token_v(&req, layer, 0, 0)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(k, snap[layer].0);
            assert_eq!(v, snap[layer].1);
        }
        // The returned block is at refcount 0, so this alloc can write it.
        // Put it back by finishing a throwaway request after the retry's own alloc.
        let logits = paged_prefill(&prompt, &weights, shape, &mut cache, &mut req, &mut work, 4)
            .unwrap()
            .to_vec();

        let mut clean = PagedCache::new(pshape(shape, 2), 4).unwrap();
        let mut clean_work = BlockWorkspace::new(shape, 8).unwrap();
        let (clean_logits, clean_req) =
            open(&prompt, &weights, shape, &mut clean, &mut clean_work, 4);
        assert_bits(&logits, &clean_logits);
        let (_, dense, _) = dense_run(&prompt, &weights, shape, 8);
        assert_kv(&cache, &req, &dense, prompt.len());
        assert_kv(&clean, &clean_req, &dense, prompt.len());
        cache.finish(req).unwrap();
        clean.finish(clean_req).unwrap();
    }

    #[test]
    fn full_page_mismatch_stays_locked_until_finish() {
        let shape = fixture_shape();
        let weights = crate::qwen25_block::block_fixture().unwrap();
        let tokens = [0u32, 1];
        let mut cache = PagedCache::new(pshape(shape, 2), 2).unwrap();
        let mut work = BlockWorkspace::new(shape, 4).unwrap();
        let (_logits, req) = open(&tokens, &weights, shape, &mut cache, &mut work, 2);
        let block = req.blocks()[0];
        cache.finish(req).unwrap();

        let mut req = cache.admit(&tokens).unwrap();
        assert_eq!(req.len(), 2);
        let node = req.last_node();
        cache.debug_add_k(&req, 0, 0, 1, 1.0).unwrap();
        let poked = cache.token_k(&req, 0, 0, 1).unwrap().to_vec();
        assert_eq!(cache.lock_ref(node).unwrap(), 1);
        assert!(!cache.is_evictable(node));
        let err = paged_prefill(&tokens, &weights, shape, &mut cache, &mut req, &mut work, 2);
        assert_eq!(err, Err(PagedBlockError::Cache(PagedError::Conflict)));
        assert_eq!(req.len(), 2);
        assert_eq!(cache.lock_ref(node).unwrap(), 1);
        assert!(!cache.is_evictable(node));
        assert_eq!(cache.token_k(&req, 0, 0, 1).unwrap(), &poked);
        cache.finish(req).unwrap();
        assert_eq!(cache.lock_ref(node).unwrap(), 0);
        assert!(cache.is_evictable(node));
        assert_eq!(cache.refcount(block).unwrap(), 1);
    }

    #[test]
    fn prefill_full_does_not_touch_a_pinned_page() {
        let shape = fixture_shape();
        let weights = crate::qwen25_block::block_fixture().unwrap();
        let mut cache = PagedCache::new(pshape(shape, 2), 1).unwrap();
        let mut work = BlockWorkspace::new(shape, 4).unwrap();
        let (_logits, pinned) = open(&[0, 1], &weights, shape, &mut cache, &mut work, 2);
        let block = pinned.blocks()[0];
        let before = cache.token_v(&pinned, 0, 0, 0).unwrap().to_vec();
        let mut other = cache.admit(&[2, 3]).unwrap();
        assert!(other.blocks().is_empty());
        let root = other.last_node();
        assert_eq!(cache.lock_ref(root).unwrap(), 2);
        let err = paged_prefill(
            &[2, 3],
            &weights,
            shape,
            &mut cache,
            &mut other,
            &mut work,
            2,
        );
        assert_eq!(err, Err(PagedBlockError::Cache(PagedError::Full)));
        assert!(other.blocks().is_empty());
        assert_eq!(other.len(), 0);
        assert_eq!(cache.token_v(&pinned, 0, 0, 0).unwrap(), &before);
        assert_eq!(cache.refcount(block).unwrap(), 2);
        cache.finish(other).unwrap();
        assert_eq!(cache.lock_ref(root).unwrap(), 1);
        let page = pinned.last_node();
        cache.finish(pinned).unwrap();
        assert_eq!(cache.lock_ref(root).unwrap(), 0);
        assert_eq!(cache.node_tokens(page).unwrap(), &[0, 1]);
        assert!(cache.is_evictable(page));
    }
}
