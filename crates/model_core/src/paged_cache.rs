//! Paged KV pool, radix prefix cache, and paged attention.
//!
//! One physical block holds every layer's f32 K and V. Layout inside a block:
//!
//! ```text
//! block_stride = num_layers * 2 * num_kv_heads * block_tokens * head_dim
//! base(block)  = block_index * block_stride
//! offset(layer, plane, head, slot) =
//!     ((((layer * 2 + plane) * num_kv_heads + head) * block_tokens + slot) * head_dim
//! ```
//!
//! `plane` 0 is K and 1 is V. `slot` is `0 .. block_tokens`. Logical token `t`
//! lives at `block_table[t / block_tokens]`, slot `t % block_tokens`, on every
//! layer. Default `block_tokens` is [`BLOCK_TOKENS`] (16). A test may pass 2
//! or 4.
//!
//! The radix indexes token ids, not KV bytes. Only a full page is admitted.
//! A partial tail stays on the request: it is not shared and it is not
//! copy-on-write. `refcount` counts owners of a physical block (the tree, if
//! the page is sealed, plus each live block table that points at it).
//! `lock_ref` counts in-flight requests whose sealed prefix includes that
//! node. Eviction takes an unlocked leaf, least-recently touched, and frees
//! its blocks only when each `refcount` is 1.
//!
//! Paged attention is [`tensor_core::gqa_attn_f32`] with a different address.
//! The dot, the scale, and the softmax order are the same, so the bits match
//! a dense pack. The dense [`crate::KvCache`] stays in place as that reference.
//!
//! There is no speculative decode, beam, or KV quantisation here.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use tensor_core::{attn_scale_f32, gqa_attn_f32, GqaError, GqaMask, GqaShape};

use crate::qwen25_layout::{HEAD_DIM, NUM_HIDDEN_LAYERS, NUM_KEY_VALUE_HEADS};

/// Production page size. Tests pass a smaller `block_tokens` on [`PagedShape`].
pub const BLOCK_TOKENS: usize = 16;

/// Default prefill chunk, a multiple of [`BLOCK_TOKENS`].
pub const PREFILL_CHUNK: usize = 256;

/// Empty block-table slot. Every id in `0 .. capacity` is a real block.
pub const BLOCK_NONE: u32 = u32::MAX;

/// Physical block index.
pub type BlockId = u32;

/// Index into the radix node arena. A live node's id does not change.
pub type NodeId = usize;

const ROOT: NodeId = 0;
const PLANE_K: usize = 0;
const PLANE_V: usize = 1;

/// Geometry of one [`PagedCache`]. Qwen2.5-1.5B uses [`PagedShape::qwen25_1_5b`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedShape {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_tokens: usize,
}

impl PagedShape {
    /// 28 layers, 2 KV heads, head dim 128, block size 16.
    pub const fn qwen25_1_5b() -> Self {
        Self {
            num_layers: NUM_HIDDEN_LAYERS,
            num_kv_heads: NUM_KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            block_tokens: BLOCK_TOKENS,
        }
    }

    fn block_stride(self) -> Option<usize> {
        self.num_layers
            .checked_mul(2)?
            .checked_mul(self.num_kv_heads)?
            .checked_mul(self.block_tokens)?
            .checked_mul(self.head_dim)
    }
}

/// Failure from the pool, the tree, or paged attention.
///
/// `Conflict` is a refused write, a KV mismatch on a duplicate page, or a
/// broken refcount. The bytes of a shared block are left as they were.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedError {
    Shape,
    Range,
    Full,
    Conflict,
    Position,
    Token,
}

impl fmt::Display for PagedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PagedError::Shape => write!(f, "paged shape does not match the buffers"),
            PagedError::Range => write!(f, "paged index is out of range"),
            PagedError::Full => write!(f, "paged cache is full"),
            PagedError::Conflict => write!(f, "paged write conflicts with a sealed block"),
            PagedError::Position => write!(f, "paged position does not fit"),
            PagedError::Token => write!(f, "token id is out of range"),
        }
    }
}

/// Page-aligned hit from [`PagedCache::match_prefix`]. Does not lock or allocate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    pub node: NodeId,
    pub len: usize,
    pub blocks: Vec<BlockId>,
}

/// One in-flight sequence. `len` is the published token count.
///
/// Other requests and the radix see `len` only. Slots at `t >= len` may
/// already hold this chunk's K/V. [`PagedCache::finish`] consumes the request;
/// the block ids are then invalid.
#[derive(Debug)]
pub struct PagedReq {
    blocks: Vec<BlockId>,
    len: usize,
    last_node: NodeId,
    tokens: Vec<u32>,
}

impl PagedReq {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    pub fn last_node(&self) -> NodeId {
        self.last_node
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

struct Node {
    tokens: Vec<u32>,
    blocks: Vec<BlockId>,
    children: BTreeMap<u32, NodeId>,
    parent: Option<NodeId>,
    lock_ref: u32,
    last_access: u64,
    alive: bool,
}

/// Outcome of comparing one existing edge with a request's private pages.
enum Adopt {
    /// Every page matched. The request now points at the authoritative blocks.
    Sealed,
    /// The first page disagreed. Nothing on this edge was sealed.
    ConflictAtStart,
    /// A proper prefix of the edge matched and is sealed. `len` stops there.
    ConflictAfterPrefix,
}

/// Fixed pool plus the radix of sealed pages.
///
/// The pool is one `Vec<f32>` sized at construction and never grown. Free
/// blocks are a stack: allocate pops, free pushes. Releasing a block does not
/// zero it. Attention simply does not load slots at or past the caller's
/// sequence length.
pub struct PagedCache {
    shape: PagedShape,
    capacity: usize,
    stride: usize,
    kv: Vec<f32>,
    refcount: Vec<u32>,
    /// `true` once a radix node owns the block.
    sealed: Vec<bool>,
    free: Vec<BlockId>,
    nodes: Vec<Node>,
    /// Unlocked leaves, ordered by `(last_access, node id)`.
    evictable: BTreeMap<(u64, NodeId), ()>,
    /// Logical clock. Starts at 0; each touch adds 1, so the first stamp is 1.
    clock: u64,
}

impl PagedCache {
    /// Allocate `capacity` zeroed blocks. `capacity == 0` or a zero dimension
    /// is [`PagedError::Shape`].
    pub fn new(shape: PagedShape, capacity: usize) -> Result<Self, PagedError> {
        if shape.num_layers == 0
            || shape.num_kv_heads == 0
            || shape.head_dim == 0
            || shape.block_tokens == 0
            || capacity == 0
        {
            return Err(PagedError::Shape);
        }
        let stride = shape.block_stride().ok_or(PagedError::Shape)?;
        let total = capacity.checked_mul(stride).ok_or(PagedError::Shape)?;
        let mut kv = Vec::new();
        kv.resize(total, 0.0);
        let mut free = Vec::new();
        for id in (0..capacity).rev() {
            free.push(id as BlockId);
        }
        let root = Node {
            tokens: Vec::new(),
            blocks: Vec::new(),
            children: BTreeMap::new(),
            parent: None,
            lock_ref: 0,
            last_access: 0,
            alive: true,
        };
        Ok(Self {
            shape,
            capacity,
            stride,
            kv,
            refcount: alloc_zeros(capacity),
            sealed: alloc_bools(capacity),
            free,
            nodes: alloc_nodes(root),
            evictable: BTreeMap::new(),
            clock: 0,
        })
    }

    pub fn shape(&self) -> PagedShape {
        self.shape
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn block_stride(&self) -> usize {
        self.stride
    }

    pub fn free_len(&self) -> usize {
        self.free.len()
    }

    pub fn refcount(&self, block: BlockId) -> Result<u32, PagedError> {
        self.block_index(block).map(|i| self.refcount[i])
    }

    pub fn is_sealed(&self, block: BlockId) -> Result<bool, PagedError> {
        self.block_index(block).map(|i| self.sealed[i])
    }

    pub fn lock_ref(&self, node: NodeId) -> Result<u32, PagedError> {
        self.node(node).map(|n| n.lock_ref)
    }

    pub fn node_tokens(&self, node: NodeId) -> Result<&[u32], PagedError> {
        self.node(node).map(|n| n.tokens.as_slice())
    }

    pub fn node_blocks(&self, node: NodeId) -> Result<&[BlockId], PagedError> {
        self.node(node).map(|n| n.blocks.as_slice())
    }

    pub fn is_evictable(&self, node: NodeId) -> bool {
        let Some(n) = self.nodes.get(node) else {
            return false;
        };
        self.evictable.contains_key(&(n.last_access, node))
    }

    /// Page-aligned match. May split a compressed edge. Does not lock or
    /// change a refcount. Tokens past the last full page are ignored.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> Result<PrefixMatch, PagedError> {
        if tokens.is_empty() {
            return Err(PagedError::Shape);
        }
        let (node, len, blocks) = self.walk(tokens);
        Ok(PrefixMatch { node, len, blocks })
    }

    /// Match `tokens`, lock that node up through the root, and take one
    /// refcount on each hit block. The tail is not allocated.
    pub fn admit(&mut self, tokens: &[u32]) -> Result<PagedReq, PagedError> {
        let hit = self.match_prefix(tokens)?;
        self.inc_lock(hit.node);
        for &block in &hit.blocks {
            let i = self.block_index(block)?;
            self.refcount[i] = self.refcount[i]
                .checked_add(1)
                .ok_or(PagedError::Conflict)?;
        }
        Ok(PagedReq {
            blocks: hit.blocks,
            len: hit.len,
            last_node: hit.node,
            tokens: tokens[..hit.len].to_vec(),
        })
    }

    /// Pop `n` blocks, evicting unlocked leaves until the free list can supply
    /// them. Each returned block has `refcount == 1` and is not sealed.
    ///
    /// On [`PagedError::Full`] every block popped by this call is pushed back,
    /// and its `refcount` is written to 0 before that push. Those ids were
    /// never attached to a block table. Evictions that already happened stay
    /// evicted. The caller's block table and locks are not touched.
    pub fn alloc_blocks(&mut self, n: usize) -> Result<Vec<BlockId>, PagedError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut popped = Vec::new();
        loop {
            if popped.len() == n {
                return Ok(popped);
            }
            if let Some(id) = self.free.pop() {
                let i = id as usize;
                if self.refcount[i] != 0 {
                    self.rewind(popped);
                    return Err(PagedError::Conflict);
                }
                self.refcount[i] = 1;
                self.sealed[i] = false;
                popped.push(id);
                continue;
            }
            match self.evict_one() {
                Ok(true) => {}
                Ok(false) => {
                    self.rewind(popped);
                    return Err(PagedError::Full);
                }
                Err(err) => {
                    self.rewind(popped);
                    return Err(err);
                }
            }
        }
    }

    /// Append blocks from [`PagedCache::alloc_blocks`] onto `req`.
    pub fn attach(&mut self, req: &mut PagedReq, blocks: &[BlockId]) {
        req.blocks.extend_from_slice(blocks);
    }

    /// Write one token of one layer. `key` and `value` are
    /// `[num_kv_heads, head_dim]`.
    ///
    /// The slot must be unpublished (`pos >= req.len`) and the block must be
    /// private (`refcount == 1` and not sealed). Otherwise [`PagedError::Conflict`]
    /// and the block is unchanged.
    pub fn write(
        &mut self,
        req: &PagedReq,
        layer: usize,
        pos: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), PagedError> {
        let kv_dim = self.kv_dim()?;
        if key.len() != kv_dim || value.len() != kv_dim {
            return Err(PagedError::Shape);
        }
        if layer >= self.shape.num_layers {
            return Err(PagedError::Range);
        }
        if pos < req.len {
            return Err(PagedError::Conflict);
        }
        let (block, slot) = self.locate(req, pos)?;
        if self.refcount[block] != 1 || self.sealed[block] {
            return Err(PagedError::Conflict);
        }
        let hd = self.shape.head_dim;
        for head in 0..self.shape.num_kv_heads {
            let src = head * hd;
            let k_at = self.offset(block, layer, PLANE_K, head, slot);
            let v_at = self.offset(block, layer, PLANE_V, head, slot);
            for d in 0..hd {
                self.kv[k_at + d] = key[src + d];
                self.kv[v_at + d] = value[src + d];
            }
        }
        Ok(())
    }

    /// Seal full pages in `new_tokens`, then publish those tokens plus any
    /// partial tail. `req.tokens` must already stop at `req.len`.
    ///
    /// A duplicate page with the same f32 is dropped and the table retargets
    /// to the sealed block. A duplicate page with different f32 is
    /// [`PagedError::Conflict`]: earlier pages of this call stay sealed, and
    /// `len` stops at the first conflicting page.
    pub fn publish(&mut self, req: &mut PagedReq, new_tokens: &[u32]) -> Result<(), PagedError> {
        if new_tokens.is_empty() || req.tokens.len() != req.len {
            return Err(PagedError::Shape);
        }
        let new_len = req
            .len
            .checked_add(new_tokens.len())
            .ok_or(PagedError::Shape)?;
        let need = new_len.div_ceil(self.shape.block_tokens);
        if req.blocks.len() < need {
            return Err(PagedError::Range);
        }
        req.tokens.extend_from_slice(new_tokens);
        self.seal(req, new_len)
    }

    /// Drop blocks this chunk allocated that contain no published token.
    ///
    /// `len` and the locks stay. Detach and `refcount = 0` are one step: after
    /// that the block is on no table and not in the tree. Only then does it
    /// join the free list. A private tail that already holds published tokens
    /// stays, dirty slots and all. [`PagedCache::finish`] releases whatever
    /// is still in the table.
    pub fn abort_chunk(&mut self, req: &mut PagedReq) -> Result<(), PagedError> {
        let b = self.shape.block_tokens;
        let keep = if req.len == 0 { 0 } else { req.len.div_ceil(b) };
        while req.blocks.len() > keep {
            self.detach_unpublished(req)?;
        }
        let cap = keep.checked_mul(b).ok_or(PagedError::Shape)?;
        if req.tokens.len() > cap {
            req.tokens.truncate(cap);
        }
        Ok(())
    }

    /// Pop one unpublished block, zero its refcount, then free it.
    ///
    /// The block must be a private tail (`refcount == 1`, not sealed). The
    /// free-list push happens only after the id is off the table.
    fn detach_unpublished(&mut self, req: &mut PagedReq) -> Result<(), PagedError> {
        let id = req.blocks.pop().ok_or(PagedError::Range)?;
        let i = match self.block_index(id) {
            Ok(i) => i,
            Err(err) => {
                req.blocks.push(id);
                return Err(err);
            }
        };
        if self.refcount[i] != 1 || self.sealed[i] {
            req.blocks.push(id);
            return Err(PagedError::Conflict);
        }
        self.refcount[i] = 0;
        self.free.push(id);
        Ok(())
    }

    /// Drop this request's blocks, then unlock from `last_node` up to the root.
    ///
    /// The table is cleared first. Each pop and its `refcount -= 1` are one
    /// step. A private tail hits 0 and only then joins the free list. A sealed
    /// page stays with the tree, at `refcount == 1`. `dec_lock` runs after
    /// that, so a leaf enters the evict set only once every one of its blocks
    /// is owned by the tree alone. Block ids on `req` are then dead.
    pub fn finish(&mut self, mut req: PagedReq) -> Result<(), PagedError> {
        let last = req.last_node;
        while let Some(id) = req.blocks.pop() {
            self.release(id)?;
        }
        self.dec_lock(last)?;
        Ok(())
    }

    /// Copy logical K/V into the dense `[num_kv_heads, seq, head_dim]` layout
    /// [`tensor_core::gqa_attn_f32`] reads. `seq` may pass `req.len` when this
    /// request has already written those slots.
    pub fn pack_logical(
        &self,
        req: &PagedReq,
        layer: usize,
        seq: usize,
        dst_k: &mut [f32],
        dst_v: &mut [f32],
    ) -> Result<(), PagedError> {
        let kv = self.shape.num_kv_heads;
        let hd = self.shape.head_dim;
        let need = kv
            .checked_mul(seq)
            .and_then(|n| n.checked_mul(hd))
            .ok_or(PagedError::Shape)?;
        if dst_k.len() < need || dst_v.len() < need || layer >= self.shape.num_layers {
            return Err(PagedError::Shape);
        }
        for t in 0..seq {
            let (block, slot) = self.locate(req, t)?;
            for head in 0..kv {
                let src_k = self.offset(block, layer, PLANE_K, head, slot);
                let src_v = self.offset(block, layer, PLANE_V, head, slot);
                let dst = (head * seq + t) * hd;
                dst_k[dst..dst + hd].copy_from_slice(&self.kv[src_k..src_k + hd]);
                dst_v[dst..dst + hd].copy_from_slice(&self.kv[src_v..src_v + hd]);
            }
        }
        Ok(())
    }

    /// One query, one layer, addressed through `req`'s block table.
    ///
    /// `seq_len` is the number of logical keys. [`GqaMask::Full`] with
    /// `seq_len = query_pos + 1` is the causal set: keys past that point are
    /// not loaded. The arithmetic matches [`gqa_attn_f32`].
    pub fn paged_gqa(
        &self,
        req: &PagedReq,
        layer: usize,
        q: &[f32],
        y: &mut [f32],
        probs: &mut [f32],
        num_q_heads: usize,
        seq_len: usize,
        mask: GqaMask,
    ) -> Result<(), PagedError> {
        self.paged_gqa_inner(req, layer, q, y, probs, num_q_heads, seq_len, mask)
    }

    /// K lanes for one head at logical `t`. The slice length is `head_dim`.
    pub fn token_k(
        &self,
        req: &PagedReq,
        layer: usize,
        head: usize,
        t: usize,
    ) -> Result<&[f32], PagedError> {
        self.token_plane(req, layer, PLANE_K, head, t)
    }

    /// V lanes for one head at logical `t`.
    pub fn token_v(
        &self,
        req: &PagedReq,
        layer: usize,
        head: usize,
        t: usize,
    ) -> Result<&[f32], PagedError> {
        self.token_plane(req, layer, PLANE_V, head, t)
    }

    fn paged_gqa_inner(
        &self,
        req: &PagedReq,
        layer: usize,
        q: &[f32],
        y: &mut [f32],
        probs: &mut [f32],
        num_q_heads: usize,
        seq_len: usize,
        mask: GqaMask,
    ) -> Result<(), PagedError> {
        let kv_heads = self.shape.num_kv_heads;
        let hd = self.shape.head_dim;
        if num_q_heads == 0
            || kv_heads == 0
            || hd == 0
            || seq_len == 0
            || layer >= self.shape.num_layers
        {
            return Err(PagedError::Shape);
        }
        if num_q_heads % kv_heads != 0 {
            return Err(PagedError::Shape);
        }
        let q_elems = num_q_heads.checked_mul(hd).ok_or(PagedError::Shape)?;
        if y.len() != q_elems || q.len() != q_elems || probs.len() != seq_len {
            return Err(PagedError::Shape);
        }
        let q_per_kv = num_q_heads / kv_heads;
        for t in 0..seq_len {
            if key_kept(t, mask) {
                self.locate(req, t)?;
            }
        }
        let scale = attn_scale_f32(hd).map_err(|_| PagedError::Shape)?;
        for q_head in 0..num_q_heads {
            let kv_head = q_head / q_per_kv;
            let q_row = &q[q_head * hd..(q_head + 1) * hd];
            for t in 0..seq_len {
                if !key_kept(t, mask) {
                    continue;
                }
                let (block, slot) = self.locate(req, t)?;
                let k_at = self.offset(block, layer, PLANE_K, kv_head, slot);
                probs[t] = dot_ltr(q_row, &self.kv[k_at..k_at + hd]) * scale;
            }
            softmax_kept(probs, mask);
            let y_row = &mut y[q_head * hd..(q_head + 1) * hd];
            for (d, out) in y_row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (t, prob) in probs.iter().enumerate() {
                    if !key_kept(t, mask) {
                        continue;
                    }
                    let (block, slot) = self.locate(req, t)?;
                    let v_at = self.offset(block, layer, PLANE_V, kv_head, slot) + d;
                    acc += *prob * self.kv[v_at];
                }
                *out = acc;
            }
        }
        Ok(())
    }

    fn seal(&mut self, req: &mut PagedReq, new_len: usize) -> Result<(), PagedError> {
        let b = self.shape.block_tokens;
        let mut node = req.last_node;
        let mut pos = self.path_len(node);
        while pos + b <= new_len {
            let first = req.tokens[pos];
            let Some(child) = self.nodes[node].children.get(&first).copied() else {
                let pages = (new_len - pos) / b;
                let take = pages * b;
                let tokens = req.tokens[pos..pos + take].to_vec();
                let blocks: Vec<BlockId> = (0..pages).map(|i| req.blocks[pos / b + i]).collect();
                node = self.insert_new(node, tokens, &blocks)?;
                break;
            };
            let take = self.page_prefix(child, &req.tokens[pos..new_len]);
            if take == 0 {
                break;
            }
            let child_len = self.nodes[child].tokens.len();
            let landed = if take < child_len {
                self.split(child, take)
            } else {
                child
            };
            match self.adopt(req, landed, pos)? {
                Adopt::Sealed => {
                    node = landed;
                    pos += take;
                }
                // No page of this edge was sealed. `node` is the parent, and
                // `pos` already includes pages sealed earlier in this publish.
                Adopt::ConflictAtStart => {
                    req.last_node = node;
                    req.len = pos;
                    return Err(PagedError::Conflict);
                }
                // A matching prefix of this edge is sealed. `adopt` recorded
                // `last_node` and `len`; do not move them back to the parent.
                Adopt::ConflictAfterPrefix => return Err(PagedError::Conflict),
            }
        }
        req.last_node = node;
        req.len = new_len;
        Ok(())
    }

    /// Compare every page of `node` with the private blocks at `pos`.
    /// On a mid-edge f32 mismatch, seal the equal prefix and stop `len` there.
    fn adopt(&mut self, req: &mut PagedReq, node: NodeId, pos: usize) -> Result<Adopt, PagedError> {
        let b = self.shape.block_tokens;
        let pages = self.nodes[node].blocks.len();
        for i in 0..pages {
            let auth = self.nodes[node].blocks[i];
            let mine = req.blocks[pos / b + i];
            if mine != auth && !self.blocks_equal(mine, auth) {
                if i == 0 {
                    return Ok(Adopt::ConflictAtStart);
                }
                let prefix = self.split(node, i * b);
                self.retarget(req, prefix, pos)?;
                self.lock_existing(prefix);
                self.touch(prefix);
                req.last_node = prefix;
                req.len = pos + i * b;
                return Ok(Adopt::ConflictAfterPrefix);
            }
        }
        self.retarget(req, node, pos)?;
        self.lock_existing(node);
        self.touch(node);
        Ok(Adopt::Sealed)
    }

    fn retarget(&mut self, req: &mut PagedReq, node: NodeId, pos: usize) -> Result<(), PagedError> {
        let b = self.shape.block_tokens;
        let pages = self.nodes[node].blocks.len();
        for i in 0..pages {
            let auth = self.nodes[node].blocks[i];
            let idx = pos / b + i;
            let mine = req.blocks[idx];
            if mine == auth {
                continue;
            }
            req.blocks[idx] = auth;
            let ai = self.block_index(auth)?;
            self.refcount[ai] = self.refcount[ai]
                .checked_add(1)
                .ok_or(PagedError::Conflict)?;
            self.release(mine)?;
        }
        Ok(())
    }

    fn insert_new(
        &mut self,
        parent: NodeId,
        tokens: Vec<u32>,
        blocks: &[BlockId],
    ) -> Result<NodeId, PagedError> {
        let first = *tokens.first().ok_or(PagedError::Shape)?;
        if self.nodes[parent].children.contains_key(&first) {
            return Err(PagedError::Conflict);
        }
        self.clock += 1;
        let id = self.nodes.len();
        self.nodes.push(Node {
            tokens,
            blocks: blocks.to_vec(),
            children: BTreeMap::new(),
            parent: Some(parent),
            lock_ref: 1,
            last_access: self.clock,
            alive: true,
        });
        self.nodes[parent].children.insert(first, id);
        self.remove_evictable(parent);
        for &block in blocks {
            let i = self.block_index(block)?;
            self.refcount[i] = self.refcount[i]
                .checked_add(1)
                .ok_or(PagedError::Conflict)?;
            self.sealed[i] = true;
        }
        Ok(id)
    }

    fn walk(&mut self, tokens: &[u32]) -> (NodeId, usize, Vec<BlockId>) {
        let b = self.shape.block_tokens;
        let mut node = ROOT;
        let mut matched = 0;
        let mut blocks = Vec::new();
        let mut touched = Vec::new();
        touched.push(ROOT);
        while matched < tokens.len() {
            let first = tokens[matched];
            let Some(child) = self.nodes[node].children.get(&first).copied() else {
                break;
            };
            let edge_len = self.nodes[child].tokens.len();
            let n_raw = common_prefix(&self.nodes[child].tokens, &tokens[matched..]);
            let n = n_raw - n_raw % b;
            if n == 0 {
                break;
            }
            if n < edge_len {
                let prefix = self.split(child, n);
                blocks.extend_from_slice(&self.nodes[prefix].blocks);
                matched += n;
                node = prefix;
                touched.push(prefix);
                break;
            }
            blocks.extend_from_slice(&self.nodes[child].blocks);
            matched += edge_len;
            node = child;
            touched.push(child);
        }
        for id in touched {
            self.touch(id);
        }
        (node, matched, blocks)
    }

    fn split(&mut self, child: NodeId, n: usize) -> NodeId {
        let b = self.shape.block_tokens;
        debug_assert!(n > 0 && n % b == 0);
        debug_assert!(n < self.nodes[child].tokens.len());
        let prefix_tokens = self.nodes[child].tokens[..n].to_vec();
        let n_blocks = n / b;
        let prefix_blocks = self.nodes[child].blocks[..n_blocks].to_vec();
        let suffix_first = self.nodes[child].tokens[n];
        let parent = self.nodes[child].parent.expect("root has no edge to split");
        let lock_ref = self.nodes[child].lock_ref;
        let last_access = self.nodes[child].last_access;
        let prefix = self.nodes.len();
        self.nodes.push(Node {
            tokens: prefix_tokens,
            blocks: prefix_blocks,
            children: BTreeMap::new(),
            parent: Some(parent),
            lock_ref,
            last_access,
            alive: true,
        });
        let suffix_tokens = self.nodes[child].tokens[n..].to_vec();
        let suffix_blocks = self.nodes[child].blocks[n_blocks..].to_vec();
        self.nodes[child].tokens = suffix_tokens;
        self.nodes[child].blocks = suffix_blocks;
        self.nodes[child].parent = Some(prefix);
        let first = self.nodes[prefix].tokens[0];
        self.nodes[parent].children.insert(first, prefix);
        self.nodes[prefix].children.insert(suffix_first, child);
        prefix
    }

    fn page_prefix(&self, child: NodeId, rest: &[u32]) -> usize {
        let n = common_prefix(&self.nodes[child].tokens, rest);
        let b = self.shape.block_tokens;
        n - n % b
    }

    fn path_len(&self, mut node: NodeId) -> usize {
        let mut n = 0;
        loop {
            n += self.nodes[node].tokens.len();
            match self.nodes[node].parent {
                Some(parent) => node = parent,
                None => break,
            }
        }
        n
    }

    fn inc_lock(&mut self, mut node: NodeId) {
        loop {
            self.nodes[node].lock_ref = self.nodes[node].lock_ref.saturating_add(1);
            self.remove_evictable(node);
            match self.nodes[node].parent {
                Some(parent) => node = parent,
                None => break,
            }
        }
    }

    fn dec_lock(&mut self, mut node: NodeId) -> Result<(), PagedError> {
        loop {
            if self.nodes[node].lock_ref == 0 {
                return Err(PagedError::Conflict);
            }
            self.nodes[node].lock_ref -= 1;
            // The table was already dropped. An unlocked leaf's blocks are the
            // tree's alone; a 2 here means `finish` unlocked before releasing.
            if self.is_unlocked_leaf(node) {
                self.leaf_blocks_cached(node)?;
            }
            self.maybe_evictable(node);
            match self.nodes[node].parent {
                Some(parent) => node = parent,
                None => break,
            }
        }
        Ok(())
    }

    fn lock_existing(&mut self, node: NodeId) {
        self.nodes[node].lock_ref = self.nodes[node].lock_ref.saturating_add(1);
        self.remove_evictable(node);
    }

    fn touch(&mut self, node: NodeId) {
        self.remove_evictable(node);
        self.clock += 1;
        self.nodes[node].last_access = self.clock;
        self.maybe_evictable(node);
    }

    fn is_unlocked_leaf(&self, node: NodeId) -> bool {
        let n = &self.nodes[node];
        n.alive && n.parent.is_some() && n.lock_ref == 0 && n.children.is_empty()
    }

    /// `lock_ref == 0` and a leaf: every block is sealed and owned only by the tree.
    fn leaf_blocks_cached(&self, node: NodeId) -> Result<(), PagedError> {
        for &block in &self.nodes[node].blocks {
            let i = self.block_index(block)?;
            if self.refcount[i] != 1 || !self.sealed[i] {
                return Err(PagedError::Conflict);
            }
        }
        Ok(())
    }

    fn maybe_evictable(&mut self, node: NodeId) {
        let n = &self.nodes[node];
        if n.alive && n.parent.is_some() && n.lock_ref == 0 && n.children.is_empty() {
            self.evictable.insert((n.last_access, node), ());
        }
    }

    fn remove_evictable(&mut self, node: NodeId) {
        let access = self.nodes[node].last_access;
        self.evictable.remove(&(access, node));
    }

    fn evict_one(&mut self) -> Result<bool, PagedError> {
        let Some((access, id)) = self.evictable.keys().next().copied() else {
            return Ok(false);
        };
        let parent = match self.nodes[id].parent {
            Some(parent) => parent,
            None => return Err(PagedError::Conflict),
        };
        if self.nodes[id].lock_ref != 0
            || !self.nodes[id].children.is_empty()
            || !self.nodes[id].alive
        {
            return Err(PagedError::Conflict);
        }
        let blocks = self.nodes[id].blocks.clone();
        let first = self.nodes[id].tokens[0];
        for &block in &blocks {
            if self.refcount[block as usize] != 1 || !self.sealed[block as usize] {
                return Err(PagedError::Conflict);
            }
        }
        self.evictable.remove(&(access, id));
        for block in blocks {
            let i = block as usize;
            self.refcount[i] = 0;
            self.sealed[i] = false;
            self.free.push(block);
        }
        self.nodes[parent].children.remove(&first);
        self.nodes[id].alive = false;
        self.nodes[id].tokens.clear();
        self.nodes[id].blocks.clear();
        self.nodes[id].children.clear();
        self.maybe_evictable(parent);
        Ok(true)
    }

    fn rewind(&mut self, popped: Vec<BlockId>) {
        // Reverse keeps the free stack in the order it had before this call.
        // `refcount` is 0 before the id is free again. It was set to 1 on pop.
        for id in popped.into_iter().rev() {
            let i = id as usize;
            self.refcount[i] = 0;
            self.sealed[i] = false;
            self.free.push(id);
        }
    }

    fn release(&mut self, block: BlockId) -> Result<(), PagedError> {
        let i = self.block_index(block)?;
        if self.refcount[i] == 0 {
            return Err(PagedError::Range);
        }
        self.refcount[i] -= 1;
        if self.refcount[i] == 0 {
            self.sealed[i] = false;
            self.free.push(block);
        }
        Ok(())
    }

    fn blocks_equal(&self, a: BlockId, b: BlockId) -> bool {
        let a = a as usize * self.stride;
        let b = b as usize * self.stride;
        self.kv[a..a + self.stride] == self.kv[b..b + self.stride]
    }

    fn token_plane(
        &self,
        req: &PagedReq,
        layer: usize,
        plane: usize,
        head: usize,
        t: usize,
    ) -> Result<&[f32], PagedError> {
        if layer >= self.shape.num_layers || head >= self.shape.num_kv_heads {
            return Err(PagedError::Range);
        }
        let (block, slot) = self.locate(req, t)?;
        let at = self.offset(block, layer, plane, head, slot);
        let hd = self.shape.head_dim;
        Ok(&self.kv[at..at + hd])
    }

    fn locate(&self, req: &PagedReq, t: usize) -> Result<(usize, usize), PagedError> {
        let b = self.shape.block_tokens;
        let logical = t / b;
        let block = *req.blocks.get(logical).ok_or(PagedError::Range)?;
        let i = self.block_index(block)?;
        Ok((i, t % b))
    }

    fn offset(&self, block: usize, layer: usize, plane: usize, head: usize, slot: usize) -> usize {
        let b = self.shape.block_tokens;
        let kv = self.shape.num_kv_heads;
        let hd = self.shape.head_dim;
        let within = (((layer * 2 + plane) * kv + head) * b + slot) * hd;
        block * self.stride + within
    }

    fn kv_dim(&self) -> Result<usize, PagedError> {
        self.shape
            .num_kv_heads
            .checked_mul(self.shape.head_dim)
            .ok_or(PagedError::Shape)
    }

    fn block_index(&self, block: BlockId) -> Result<usize, PagedError> {
        let i = block as usize;
        if block == BLOCK_NONE || i >= self.capacity {
            return Err(PagedError::Range);
        }
        Ok(i)
    }

    fn node(&self, node: NodeId) -> Result<&Node, PagedError> {
        self.nodes
            .get(node)
            .filter(|n| n.alive)
            .ok_or(PagedError::Range)
    }

    /// Add `delta` to one K lane. Tests use this to force a full-page mismatch.
    #[cfg(test)]
    pub(crate) fn debug_add_k(
        &mut self,
        req: &PagedReq,
        layer: usize,
        head: usize,
        t: usize,
        delta: f32,
    ) -> Result<(), PagedError> {
        let (block, slot) = self.locate(req, t)?;
        let at = self.offset(block, layer, PLANE_K, head, slot);
        self.kv[at] += delta;
        Ok(())
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn key_kept(t: usize, mask: GqaMask) -> bool {
    match mask {
        GqaMask::Full => true,
        GqaMask::Causal { query_pos } => t <= query_pos,
    }
}

fn dot_ltr(q: &[f32], k: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for d in 0..q.len() {
        acc += q[d] * k[d];
    }
    acc
}

fn softmax_kept(scores: &mut [f32], mask: GqaMask) {
    debug_assert!(key_kept(0, mask));
    let mut max_score = scores[0];
    for (t, score) in scores.iter().enumerate().skip(1) {
        if key_kept(t, mask) && *score > max_score {
            max_score = *score;
        }
    }
    let mut sum = 0.0f32;
    for (t, score) in scores.iter_mut().enumerate() {
        if !key_kept(t, mask) {
            continue;
        }
        let e = exp_f32(*score - max_score);
        *score = e;
        sum += e;
    }
    for (t, score) in scores.iter_mut().enumerate() {
        if key_kept(t, mask) {
            *score /= sum;
        } else {
            *score = 0.0;
        }
    }
}

#[cfg_attr(unix, link(name = "m"))]
extern "C" {
    fn expf(x: f32) -> f32;
}

fn exp_f32(x: f32) -> f32 {
    // SAFETY: expf is defined for every f32 bit pattern.
    unsafe { expf(x) }
}

fn alloc_zeros(n: usize) -> Vec<u32> {
    let mut v = Vec::new();
    v.resize(n, 0);
    v
}

fn alloc_bools(n: usize) -> Vec<bool> {
    let mut v = Vec::new();
    v.resize(n, false);
    v
}

fn alloc_nodes(root: Node) -> Vec<Node> {
    let mut v = Vec::new();
    v.push(root);
    v
}

/// Pack then score, so a test can require the fused kernel and the dense
/// reference to share bits. Not the production read path.
pub fn paged_gqa_via_pack(
    cache: &PagedCache,
    req: &PagedReq,
    layer: usize,
    q: &[f32],
    y: &mut [f32],
    probs: &mut [f32],
    num_q_heads: usize,
    seq_len: usize,
    mask: GqaMask,
    pack_k: &mut [f32],
    pack_v: &mut [f32],
) -> Result<(), PagedError> {
    cache.pack_logical(req, layer, seq_len, pack_k, pack_v)?;
    let kv = cache.shape().num_kv_heads;
    let hd = cache.shape().head_dim;
    let packed = kv * seq_len * hd;
    gqa_attn_f32(
        y,
        q,
        &pack_k[..packed],
        &pack_v[..packed],
        probs,
        GqaShape {
            num_q_heads,
            num_kv_heads: kv,
            head_dim: hd,
            seq_len,
        },
        mask,
    )
    .map_err(|err| match err {
        GqaError::Shape => PagedError::Shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{KvCache, KvShape};

    fn shape(layers: usize, kv: usize, dim: usize, block: usize) -> PagedShape {
        PagedShape {
            num_layers: layers,
            num_kv_heads: kv,
            head_dim: dim,
            block_tokens: block,
        }
    }

    fn cache(block: usize, cap: usize) -> PagedCache {
        PagedCache::new(shape(1, 1, 2, block), cap).unwrap()
    }

    fn kv(pos: usize, token: u32) -> ([f32; 2], [f32; 2]) {
        ([(pos + 1) as f32, token as f32], [10.0 + pos as f32, 0.5])
    }

    fn write_all(cache: &mut PagedCache, req: &PagedReq, tokens: &[u32], from: usize) {
        for (i, &token) in tokens.iter().enumerate() {
            let (k, v) = kv(from + i, token);
            cache.write(req, 0, from + i, &k, &v).unwrap();
        }
    }

    fn seal(cache: &mut PagedCache, tokens: &[u32]) -> PagedReq {
        let mut req = cache.admit(tokens).unwrap();
        assert_eq!(req.len(), 0);
        let n = tokens.len().div_ceil(cache.shape.block_tokens);
        let blocks = cache.alloc_blocks(n).unwrap();
        cache.attach(&mut req, &blocks);
        write_all(cache, &req, tokens, 0);
        cache.publish(&mut req, tokens).unwrap();
        req
    }

    #[test]
    fn qwen_block_is_16_tokens_and_896_kib() {
        let shape = PagedShape::qwen25_1_5b();
        assert_eq!(shape.block_tokens, 16);
        assert_eq!(shape.num_layers, 28);
        assert_eq!(shape.num_kv_heads, 2);
        assert_eq!(shape.head_dim, 128);
        assert_eq!(BLOCK_TOKENS, 16);
        assert_eq!(PREFILL_CHUNK, 256);
        assert_eq!(PREFILL_CHUNK % BLOCK_TOKENS, 0);
        let stride = shape.block_stride().unwrap();
        assert_eq!(stride, 28 * 2 * 2 * 16 * 128);
        assert_eq!(stride * 4, 917_504);
        assert!(PagedCache::new(shape, 0).is_err());
        assert!(PagedCache::new(
            PagedShape {
                block_tokens: 0,
                ..shape
            },
            1
        )
        .is_err());
    }

    #[test]
    fn logical_address_matches_dense_kv() {
        let layers = 2;
        let kv_heads = 2;
        let dim = 4;
        let b = 4;
        let seq = 6;
        let mut dense = KvCache::new(KvShape {
            num_layers: layers,
            num_kv_heads: kv_heads,
            head_dim: dim,
            max_seq: seq,
        })
        .unwrap();
        let mut paged = PagedCache::new(shape(layers, kv_heads, dim, b), 4).unwrap();
        let tokens: Vec<u32> = (0..seq as u32).collect();
        let mut req = paged.admit(&tokens).unwrap();
        let blocks = paged.alloc_blocks(2).unwrap();
        paged.attach(&mut req, &blocks);
        for t in 0..seq {
            for layer in 0..layers {
                let mut key = vec![0.0f32; kv_heads * dim];
                let mut value = vec![0.0f32; kv_heads * dim];
                for head in 0..kv_heads {
                    for d in 0..dim {
                        key[head * dim + d] =
                            (layer * 100 + head * 10 + t) as f32 + d as f32 * 0.25;
                        value[head * dim + d] = 50.0 + key[head * dim + d];
                    }
                }
                dense.store_at(layer, t, &key, &value).unwrap();
                paged.write(&req, layer, t, &key, &value).unwrap();
                for head in 0..kv_heads {
                    assert_eq!(
                        paged.token_k(&req, layer, head, t).unwrap(),
                        dense.key_head(layer, head, t).unwrap()
                    );
                    assert_eq!(
                        paged.token_v(&req, layer, head, t).unwrap(),
                        dense.value_head(layer, head, t).unwrap()
                    );
                }
            }
        }
        // Mid-block (seq 6, B 4) and a boundary-sized prefix of 4.
        for (layer, len) in [(0, 6), (1, 4), (0, 4), (1, 6)] {
            let mut dk = vec![0.0f32; kv_heads * len * dim];
            let mut dv = vec![0.0f32; kv_heads * len * dim];
            let mut pk = vec![0.0f32; kv_heads * len * dim];
            let mut pv = vec![0.0f32; kv_heads * len * dim];
            dense.pack_prefix(layer, len, &mut dk, &mut dv).unwrap();
            paged
                .pack_logical(&req, layer, len, &mut pk, &mut pv)
                .unwrap();
            assert_eq!(pk, dk);
            assert_eq!(pv, dv);
        }
    }

    #[test]
    fn stale_tail_and_paged_attn_match_gqa() {
        let b = 4;
        let mut paged = cache(b, 2);
        let tokens = [1u32, 2, 3, 4];
        let mut req = paged.admit(&tokens).unwrap();
        let blocks = paged.alloc_blocks(1).unwrap();
        paged.attach(&mut req, &blocks);
        let block = blocks[0] as usize;
        for lane in &mut paged.kv[block * paged.stride..(block + 1) * paged.stride] {
            *lane = f32::NAN;
        }
        paged.abort_chunk(&mut req).unwrap();
        assert_eq!(paged.free_len(), 2);
        let again = paged.alloc_blocks(1).unwrap();
        assert_eq!(again[0], blocks[0]);
        paged.attach(&mut req, &again);
        // The recycled page still holds NaN past the one token we write.
        let (k, v) = kv(0, 1);
        paged.write(&req, 0, 0, &k, &v).unwrap();
        assert!(paged.kv[paged.offset(again[0] as usize, 0, 0, 0, 1)].is_nan());
        let q = [1.0f32, 0.0];
        let mut y = [0.0f32; 2];
        let mut probs = [0.0f32; 1];
        paged
            .paged_gqa(&req, 0, &q, &mut y, &mut probs, 1, 1, GqaMask::Full)
            .unwrap();
        let mut y_ref = [0.0f32; 2];
        let mut p_ref = [0.0f32; 1];
        let mut pack_k = [0.0f32; 2];
        let mut pack_v = [0.0f32; 2];
        paged_gqa_via_pack(
            &paged,
            &req,
            0,
            &q,
            &mut y_ref,
            &mut p_ref,
            1,
            1,
            GqaMask::Full,
            &mut pack_k,
            &mut pack_v,
        )
        .unwrap();
        assert_eq!(y, y_ref);
        assert_eq!(probs, p_ref);
        assert!(y[0].is_finite());
    }

    #[test]
    fn paged_attn_matches_packed_gqa_on_lengths_masks_and_groups() {
        // B = 4. Lengths cover a short tail, a boundary, and past two pages.
        // Query heads 4, KV heads 2: groups are contiguous, not h % 2.
        let b = 4;
        let kv_heads = 2;
        let q_heads = 4;
        let dim = 4;
        let seq = 3 * b + 5;
        let mut paged = PagedCache::new(shape(1, kv_heads, dim, b), 8).unwrap();
        let tokens: Vec<u32> = (0..seq as u32).collect();
        let mut req = paged.admit(&tokens).unwrap();
        let blocks = paged.alloc_blocks(seq.div_ceil(b)).unwrap();
        // Scramble the physical order by reversing the attach list.
        let mut scrambled = blocks.clone();
        scrambled.reverse();
        paged.attach(&mut req, &scrambled);
        for t in 0..seq {
            let mut key = vec![0.0f32; kv_heads * dim];
            let mut value = vec![0.0f32; kv_heads * dim];
            for head in 0..kv_heads {
                for d in 0..dim {
                    key[head * dim + d] = ((t + 1) * (head + 1)) as f32 + d as f32 * 0.5;
                    value[head * dim + d] = 3.0 + head as f32 + (t % 5) as f32;
                }
            }
            paged.write(&req, 0, t, &key, &value).unwrap();
        }
        let mut q = vec![0.0f32; q_heads * dim];
        for h in 0..q_heads {
            q[h * dim] = 1.0 + h as f32;
            q[h * dim + 1] = if h / 2 == 0 { 0.25 } else { -0.5 };
        }
        let lengths = [1usize, b - 1, b, b + 1, 3 * b, 3 * b + 5];
        for seq_len in lengths {
            let queries = [0usize, b - 1, b, seq_len / 2, seq_len - 1];
            for query_pos in queries {
                if query_pos >= seq_len {
                    continue;
                }
                let masks = [GqaMask::Full, GqaMask::Causal { query_pos }];
                for mask in masks {
                    let mut y = vec![7.0f32; q.len()];
                    let mut probs = vec![7.0f32; seq_len];
                    paged
                        .paged_gqa(&req, 0, &q, &mut y, &mut probs, q_heads, seq_len, mask)
                        .unwrap();
                    let mut y2 = vec![0.0f32; q.len()];
                    let mut probs2 = vec![0.0f32; seq_len];
                    let packed = kv_heads * seq_len * dim;
                    let mut pk = vec![0.0f32; packed];
                    let mut pv = vec![0.0f32; packed];
                    paged_gqa_via_pack(
                        &paged,
                        &req,
                        0,
                        &q,
                        &mut y2,
                        &mut probs2,
                        q_heads,
                        seq_len,
                        mask,
                        &mut pk,
                        &mut pv,
                    )
                    .unwrap();
                    assert_eq!(y, y2, "seq {seq_len} query {query_pos} {mask:?}");
                    assert_eq!(probs, probs2);
                }
            }
        }
        // Full on seq = q+1 matches Causal on a longer prefix. The extra
        // packed keys are invisible, and the fused kernel never loads them.
        let qpos = 5usize;
        let mut y_full = vec![0.0f32; q.len()];
        let mut p_full = vec![0.0f32; qpos + 1];
        paged
            .paged_gqa(
                &req,
                0,
                &q,
                &mut y_full,
                &mut p_full,
                q_heads,
                qpos + 1,
                GqaMask::Full,
            )
            .unwrap();
        let mut y_c = vec![0.0f32; q.len()];
        let mut p_c = vec![0.0f32; seq];
        let packed = kv_heads * seq * dim;
        let mut pk = vec![0.0f32; packed];
        let mut pv = vec![0.0f32; packed];
        paged_gqa_via_pack(
            &paged,
            &req,
            0,
            &q,
            &mut y_c,
            &mut p_c,
            q_heads,
            seq,
            GqaMask::Causal { query_pos: qpos },
            &mut pk,
            &mut pv,
        )
        .unwrap();
        assert_eq!(y_full, y_c);
    }

    #[test]
    fn write_refuses_a_sealed_block_and_bytes_stay() {
        let mut paged = cache(4, 4);
        let tokens = [7u32, 8, 9, 10];
        let req = seal(&mut paged, &tokens);
        let before = paged.token_k(&req, 0, 0, 0).unwrap().to_vec();
        let err = paged.write(&req, 0, 0, &[9.0, 9.0], &[9.0, 9.0]);
        assert_eq!(err, Err(PagedError::Conflict));
        assert_eq!(paged.token_k(&req, 0, 0, 0).unwrap(), &before);
        // A second request sharing the page cannot write it either.
        let mut other = paged.admit(&tokens).unwrap();
        assert_eq!(other.blocks(), req.blocks());
        assert_eq!(paged.refcount(req.blocks()[0]).unwrap(), 3);
        let err = paged.write(&other, 0, 4, &[1.0, 1.0], &[1.0, 1.0]);
        assert_eq!(err, Err(PagedError::Range));
        other.tokens.clear();
        // pos 0 is published.
        let err = paged.write(&other, 0, 0, &[1.0, 1.0], &[1.0, 1.0]);
        assert_eq!(err, Err(PagedError::Conflict));
        assert_eq!(paged.token_k(&req, 0, 0, 0).unwrap(), &before);
        paged.finish(req).unwrap();
        paged.finish(other).unwrap();
    }

    #[test]
    fn refcount_script_then_evict_and_recompute() {
        // Capacity 1: after both requests finish, the only block is the cached
        // page, so the next alloc has to evict it and hand that id back.
        let mut paged = cache(4, 1);
        let tokens = [10u32, 11, 12, 13];
        let a = seal(&mut paged, &tokens);
        let block = a.blocks()[0];
        assert_eq!(paged.refcount(block).unwrap(), 2);
        assert_eq!(paged.lock_ref(a.last_node()).unwrap(), 1);
        let b = paged.admit(&tokens).unwrap();
        assert_eq!(b.blocks(), &[block]);
        assert_eq!(paged.refcount(block).unwrap(), 3);
        assert_eq!(paged.lock_ref(a.last_node()).unwrap(), 2);
        let node = a.last_node();
        paged.finish(a).unwrap();
        assert_eq!(paged.refcount(block).unwrap(), 2);
        assert_eq!(paged.lock_ref(node).unwrap(), 1);
        assert!(!paged.is_evictable(node));
        paged.finish(b).unwrap();
        assert_eq!(paged.refcount(block).unwrap(), 1);
        assert_eq!(paged.lock_ref(node).unwrap(), 0);
        assert!(paged.is_evictable(node));
        let saved =
            paged.kv[block as usize * paged.stride..(block as usize + 1) * paged.stride].to_vec();
        let got = paged.alloc_blocks(1).unwrap();
        assert_eq!(got, vec![block]);
        assert_eq!(paged.refcount(block).unwrap(), 1);
        assert!(!paged.is_sealed(block).unwrap());
        assert!(paged.free.contains(&block) == false);
        // C misses and writes the same tokens into the recycled block.
        let mut c = paged.admit(&tokens).unwrap();
        assert_eq!(c.len(), 0);
        paged.attach(&mut c, &got);
        write_all(&mut paged, &c, &tokens, 0);
        let now =
            paged.kv[block as usize * paged.stride..(block as usize + 1) * paged.stride].to_vec();
        assert_eq!(now, saved);
        paged.finish(c).unwrap();
    }

    #[test]
    fn pinned_request_is_not_evicted() {
        let mut paged = cache(4, 1);
        let tokens = [1u32, 2, 3, 4];
        let req = seal(&mut paged, &tokens);
        let block = req.blocks()[0];
        let before = paged.token_k(&req, 0, 0, 0).unwrap().to_vec();
        assert_eq!(paged.free_len(), 0);
        assert_eq!(paged.alloc_blocks(1).unwrap_err(), PagedError::Full);
        assert_eq!(paged.refcount(block).unwrap(), 2);
        assert_eq!(paged.token_k(&req, 0, 0, 0).unwrap(), &before);
        assert!(!paged.free.contains(&block));
        paged.finish(req).unwrap();
    }

    #[test]
    fn short_free_list_evicts_then_still_returns_full() {
        let mut paged = cache(2, 3);
        let _hot = seal(&mut paged, &[1, 2]);
        let _also = seal(&mut paged, &[3, 4]);
        // One running request holds a private block as well? Both sealed and
        // finished would be evictable. Keep `hot` running so its block is pinned,
        // and finish the other by rebuilding: seal() returns a live request.
        // Finish the second only.
        let cold_tokens = [5u32, 6];
        let _cold = seal(&mut paged, &cold_tokens);
        // Capacity 3: hot, also, cold. Finish `also` and `cold` so two leaves
        // are freeable, while `hot` stays locked. We need `also`.
        // seal() three times used 3 blocks. hot is first.
        // Identify: finish everyone except hot, then alloc that needs more
        // blocks than the freeable ones.
        // Easier reconstruction:
        drop(paged);
        let mut paged = cache(2, 3);
        let pinned = seal(&mut paged, &[1, 2]);
        let leaf = seal(&mut paged, &[3, 4]);
        let leaf_block = leaf.blocks()[0];
        paged.finish(leaf).unwrap();
        assert_eq!(paged.free_len(), 1);
        // Need 2 blocks. One is free, one leaf can be evicted (already finished,
        // its block is NOT free — refcount 1 sealed). Free len is 1 (the
        // untouched third block). Evicting the leaf yields a second block.
        // That is enough for 2. Use a case that is still short:
        // capacity 3, pinned holds 1, one leaf holds 1, one free. alloc(3)
        // evicts the leaf (now 2 free) and still needs 3, so Full, and the
        // popped blocks return to the free list.
        let caller = paged.admit(&[9, 9]).unwrap();
        let before = caller.blocks().to_vec();
        let err = paged.alloc_blocks(3).unwrap_err();
        assert_eq!(err, PagedError::Full);
        assert_eq!(caller.blocks(), &before);
        assert_eq!(paged.refcount(pinned.blocks()[0]).unwrap(), 2);
        assert_eq!(paged.is_sealed(leaf_block).unwrap(), false);
        assert_eq!(paged.free_len(), 2);
        paged.finish(pinned).unwrap();
        paged.finish(caller).unwrap();
    }

    #[test]
    fn lru_evicts_the_cold_page_and_the_child_before_the_parent() {
        let mut paged = cache(2, 2);
        let x = seal(&mut paged, &[1, 2]);
        let y = seal(&mut paged, &[3, 4]);
        let x_block = x.blocks()[0];
        let y_block = y.blocks()[0];
        paged.finish(x).unwrap();
        paged.finish(y).unwrap();
        paged.match_prefix(&[1, 2]).unwrap();
        let got = paged.alloc_blocks(1).unwrap();
        assert_eq!(got, vec![y_block]);
        assert!(paged.is_sealed(x_block).unwrap());
        assert_eq!(paged.match_prefix(&[1, 2]).unwrap().len, 2);
        assert_eq!(paged.match_prefix(&[3, 4]).unwrap().len, 0);
        // Put Y's block back so the chain test has room. Drop the popped ref.
        paged.release(got[0]).unwrap();

        let mut chain = cache(2, 2);
        let mut req = chain.admit(&[1, 2, 3, 4]).unwrap();
        let blocks = chain.alloc_blocks(2).unwrap();
        chain.attach(&mut req, &blocks);
        write_all(&mut chain, &req, &[1, 2], 0);
        chain.publish(&mut req, &[1, 2]).unwrap();
        write_all(&mut chain, &req, &[3, 4], 2);
        chain.publish(&mut req, &[3, 4]).unwrap();
        let parent = chain.nodes[req.last_node()].parent.unwrap();
        assert_ne!(parent, ROOT);
        assert_eq!(chain.node_tokens(parent).unwrap(), &[1, 2]);
        assert_eq!(chain.node_tokens(req.last_node()).unwrap(), &[3, 4]);
        chain.finish(req).unwrap();
        assert!(chain.is_evictable(chain.nodes.len() - 1) || chain.evictable.len() == 1);
        let child = chain.evictable.keys().next().copied().unwrap().1;
        assert_eq!(chain.node_tokens(child).unwrap(), &[3, 4]);
        assert!(chain.evict_one().unwrap());
        assert!(chain.is_evictable(parent));
        assert_eq!(chain.node_tokens(parent).unwrap(), &[1, 2]);
    }

    #[test]
    fn partial_page_is_private_and_a_full_second_page_seals() {
        let mut paged = cache(4, 4);
        let mut a = paged.admit(&[10, 11, 12, 13, 14]).unwrap();
        let blocks = paged.alloc_blocks(2).unwrap();
        paged.attach(&mut a, &blocks);
        write_all(&mut paged, &a, &[10, 11, 12, 13, 14], 0);
        paged.publish(&mut a, &[10, 11, 12, 13, 14]).unwrap();
        assert_eq!(a.len(), 5);
        assert_eq!(paged.node_tokens(a.last_node()).unwrap(), &[10, 11, 12, 13]);
        assert!(!paged.is_sealed(a.blocks()[1]).unwrap());
        paged.finish(a).unwrap();
        let mut b = paged.admit(&[10, 11, 12, 13, 14]).unwrap();
        assert_eq!(b.len(), 4);
        let tail = paged.alloc_blocks(1).unwrap();
        assert_ne!(tail[0], b.blocks()[0]);
        paged.attach(&mut b, &tail);
        write_all(&mut paged, &b, &[14], 4);
        paged.publish(&mut b, &[14]).unwrap();
        assert_eq!(b.len(), 5);
        paged.finish(b).unwrap();
        // Only the full page remains.
        let hit = paged.match_prefix(&[10, 11, 12, 13, 14]).unwrap();
        assert_eq!(hit.len, 4);
        assert_eq!(hit.blocks.len(), 1);

        // Decode fills the next page and a later request hits both pages.
        let mut d = paged.admit(&[10, 11, 12, 13]).unwrap();
        assert_eq!(d.len(), 4);
        let next = [14u32, 16, 17, 18];
        for (i, token) in next.iter().enumerate() {
            if d.len() % 4 == 0 {
                let blk = paged.alloc_blocks(1).unwrap();
                paged.attach(&mut d, &blk);
            }
            let (k, v) = kv(4 + i, *token);
            paged.write(&d, 0, d.tokens.len(), &k, &v).unwrap();
            paged.publish(&mut d, &[*token]).unwrap();
        }
        assert_eq!(d.len(), 8);
        assert_eq!(paged.node_tokens(d.last_node()).unwrap(), &next);
        let parent = paged.nodes[d.last_node()].parent.unwrap();
        assert_eq!(paged.node_tokens(parent).unwrap(), &[10, 11, 12, 13]);
        paged.finish(d).unwrap();
        let hit2 = paged
            .match_prefix(&[10, 11, 12, 13, 14, 16, 17, 18])
            .unwrap();
        assert_eq!(hit2.len, 8);
    }

    #[test]
    fn shared_prefix_is_not_copied_or_mutated() {
        let mut paged = cache(4, 6);
        let a = seal(&mut paged, &[1, 2, 3, 4, 5, 6, 7, 8]);
        // seal() publishes everything, including both pages, and keeps the req.
        let prefix = a.blocks()[..2].to_vec();
        let snap: Vec<f32> = prefix
            .iter()
            .flat_map(|b| {
                let i = *b as usize * paged.stride;
                paged.kv[i..i + paged.stride].to_vec()
            })
            .collect();
        paged.finish(a).unwrap();
        let mut b = paged.admit(&[1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        assert_eq!(b.len(), 8);
        assert_eq!(&b.blocks()[..2], &prefix[..]);
        let tail = paged.alloc_blocks(1).unwrap();
        paged.attach(&mut b, &tail);
        let (k, v) = kv(8, 9);
        paged.write(&b, 0, 8, &k, &v).unwrap();
        let after: Vec<f32> = prefix
            .iter()
            .flat_map(|blk| {
                let i = *blk as usize * paged.stride;
                paged.kv[i..i + paged.stride].to_vec()
            })
            .collect();
        assert_eq!(after, snap);
        paged.finish(b).unwrap();
    }

    #[test]
    fn unpublished_slot_stays_out_of_match_and_shorter_attention() {
        let mut paged = cache(4, 2);
        let mut req = paged.admit(&[1, 2, 3, 4, 5]).unwrap();
        let blocks = paged.alloc_blocks(2).unwrap();
        paged.attach(&mut req, &blocks);
        write_all(&mut paged, &req, &[1, 2, 3, 4], 0);
        paged.publish(&mut req, &[1, 2, 3, 4]).unwrap();
        let (k, v) = kv(4, 5);
        paged.write(&req, 0, 4, &k, &v).unwrap();
        assert_eq!(req.len(), 4);
        // A different cache walk must not observe token 5: it is not sealed.
        // This request still owns the private tail, so match on a fresh token
        // list of length 5 hits only the full page.
        let hit = paged.match_prefix(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(hit.len, 4);
        let q = [1.0f32, 0.0];
        let mut y_short = [0.0f32; 2];
        let mut p_short = [0.0f32; 4];
        paged
            .paged_gqa(&req, 0, &q, &mut y_short, &mut p_short, 1, 4, GqaMask::Full)
            .unwrap();
        let mut y_long = [0.0f32; 2];
        let mut p_long = [0.0f32; 5];
        paged
            .paged_gqa(&req, 0, &q, &mut y_long, &mut p_long, 1, 5, GqaMask::Full)
            .unwrap();
        assert_ne!(y_short, y_long);
        let mut y_pack = [0.0f32; 2];
        let mut p_pack = [0.0f32; 4];
        let mut pk = [0.0f32; 8];
        let mut pv = [0.0f32; 8];
        paged_gqa_via_pack(
            &paged,
            &req,
            0,
            &q,
            &mut y_pack,
            &mut p_pack,
            1,
            4,
            GqaMask::Full,
            &mut pk,
            &mut pv,
        )
        .unwrap();
        assert_eq!(y_short, y_pack);
        paged.finish(req).unwrap();
    }

    #[test]
    fn dedup_drops_the_copy_and_a_changed_byte_conflicts() {
        let mut paged = cache(2, 4);
        let mut a = paged.admit(&[4, 5]).unwrap();
        let ba = paged.alloc_blocks(1).unwrap();
        paged.attach(&mut a, &ba);
        write_all(&mut paged, &a, &[4, 5], 0);
        // B admits before A publishes, so it computes its own page.
        let mut b = paged.admit(&[4, 5]).unwrap();
        assert_eq!(b.len(), 0);
        let bb = paged.alloc_blocks(1).unwrap();
        assert_ne!(bb[0], ba[0]);
        paged.attach(&mut b, &bb);
        write_all(&mut paged, &b, &[4, 5], 0);
        paged.publish(&mut a, &[4, 5]).unwrap();
        let auth = a.blocks()[0];
        let before =
            paged.kv[auth as usize * paged.stride..(auth as usize + 1) * paged.stride].to_vec();
        paged.publish(&mut b, &[4, 5]).unwrap();
        assert_eq!(b.blocks(), &[auth]);
        assert_eq!(paged.refcount(bb[0]).unwrap(), 0);
        assert!(paged.free.contains(&bb[0]));
        assert_eq!(paged.node_blocks(b.last_node()).unwrap(), &[auth]);
        assert_eq!(
            &paged.kv[auth as usize * paged.stride..(auth as usize + 1) * paged.stride],
            &before
        );
        // Tree owns it once, plus A and B.
        assert_eq!(paged.refcount(auth).unwrap(), 3);
        paged.finish(a).unwrap();
        paged.finish(b).unwrap();
        assert_eq!(paged.refcount(auth).unwrap(), 1);

        // A second pool: C computes the page before it is sealed, then flips
        // one lane. Publishing after A has sealed the honest page conflicts,
        // and the tree is not split or extended.
        let mut paged = cache(2, 4);
        let mut a = paged.admit(&[4, 5]).unwrap();
        let ba = paged.alloc_blocks(1).unwrap();
        paged.attach(&mut a, &ba);
        write_all(&mut paged, &a, &[4, 5], 0);
        let mut c = paged.admit(&[4, 5]).unwrap();
        assert_eq!(c.len(), 0);
        let bc = paged.alloc_blocks(1).unwrap();
        paged.attach(&mut c, &bc);
        write_all(&mut paged, &c, &[4, 5], 0);
        let at = paged.offset(bc[0] as usize, 0, PLANE_K, 0, 0);
        paged.kv[at] += 1.0;
        paged.publish(&mut a, &[4, 5]).unwrap();
        let auth = a.blocks()[0];
        let before =
            paged.kv[auth as usize * paged.stride..(auth as usize + 1) * paged.stride].to_vec();
        let nodes_before = paged.nodes.len();
        let err = paged.publish(&mut c, &[4, 5]);
        assert_eq!(err, Err(PagedError::Conflict));
        assert_eq!(paged.nodes.len(), nodes_before);
        assert_eq!(c.len(), 0);
        assert_eq!(c.last_node(), ROOT);
        assert_eq!(
            &paged.kv[auth as usize * paged.stride..(auth as usize + 1) * paged.stride],
            &before
        );
        paged.finish(a).unwrap();
        paged.finish(c).unwrap();
    }

    #[test]
    fn split_copies_the_lock_and_keeps_the_suffix_id() {
        let mut paged = cache(2, 4);
        let mut a = paged.admit(&[1, 2, 3, 4]).unwrap();
        let blocks = paged.alloc_blocks(2).unwrap();
        paged.attach(&mut a, &blocks);
        write_all(&mut paged, &a, &[1, 2, 3, 4], 0);
        paged.publish(&mut a, &[1, 2, 3, 4]).unwrap();
        let suffix = a.last_node();
        assert_eq!(paged.node_tokens(suffix).unwrap(), &[1, 2, 3, 4]);
        assert_eq!(paged.lock_ref(suffix).unwrap(), 1);
        // B hits only the first page and splits the compressed edge.
        let b = paged.admit(&[1, 2]).unwrap();
        assert_eq!(a.last_node(), suffix);
        assert_eq!(paged.node_tokens(suffix).unwrap(), &[3, 4]);
        let prefix = b.last_node();
        assert_ne!(prefix, suffix);
        assert_eq!(paged.node_tokens(prefix).unwrap(), &[1, 2]);
        assert_eq!(paged.nodes[suffix].parent, Some(prefix));
        // Copied lock from A, plus B's admit.
        assert_eq!(paged.lock_ref(prefix).unwrap(), 2);
        assert_eq!(paged.lock_ref(suffix).unwrap(), 1);
        paged.finish(a).unwrap();
        assert_eq!(paged.lock_ref(prefix).unwrap(), 1);
        assert!(!paged.is_evictable(prefix));
        paged.finish(b).unwrap();
        assert_eq!(paged.lock_ref(prefix).unwrap(), 0);
    }

    #[test]
    fn abort_chunk_releases_blocks_that_hold_nothing_published() {
        let mut paged = cache(4, 4);
        let mut req = paged.admit(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let blocks = paged.alloc_blocks(2).unwrap();
        paged.attach(&mut req, &blocks);
        write_all(&mut paged, &req, &[1, 2, 3, 4], 0);
        paged.publish(&mut req, &[1, 2, 3, 4]).unwrap();
        assert_eq!(req.len(), 4);
        write_all(&mut paged, &req, &[5, 6], 4);
        let published = req.blocks()[0];
        let tail = req.blocks()[1];
        paged.abort_chunk(&mut req).unwrap();
        assert_eq!(req.len(), 4);
        assert_eq!(req.blocks(), &[published]);
        assert_eq!(paged.refcount(tail).unwrap(), 0);
        assert!(paged.free.contains(&tail));
        assert_eq!(paged.token_k(&req, 0, 0, 0).unwrap()[0], 1.0);
        // The freed id is not still owned by this table. The next alloc can
        // write it. `refcount` was 0 before the free-list push.
        assert!(!req.blocks().contains(&tail));
        let mut other = paged.admit(&[9, 9, 9, 9]).unwrap();
        let got = paged.alloc_blocks(1).unwrap();
        assert_eq!(got[0], tail);
        paged.attach(&mut other, &got);
        let (k, v) = kv(0, 9);
        paged.write(&other, 0, 0, &k, &v).unwrap();
        paged.finish(req).unwrap();
        paged.finish(other).unwrap();
    }

    #[test]
    fn alloc_failure_writes_refcount_back_to_zero() {
        let mut paged = cache(2, 3);
        let pinned = seal(&mut paged, &[1, 2]);
        let leaf = seal(&mut paged, &[3, 4]);
        let leaf_block = leaf.blocks()[0];
        let pinned_block = pinned.blocks()[0];
        let pinned_bytes = paged.kv
            [pinned_block as usize * paged.stride..(pinned_block as usize + 1) * paged.stride]
            .to_vec();
        paged.finish(leaf).unwrap();
        let caller = paged.admit(&[9, 9]).unwrap();
        let before_blocks = caller.blocks().to_vec();
        let before_lock = paged.lock_ref(caller.last_node()).unwrap();
        assert_eq!(paged.alloc_blocks(3).unwrap_err(), PagedError::Full);
        assert_eq!(caller.blocks(), &before_blocks);
        assert_eq!(paged.lock_ref(caller.last_node()).unwrap(), before_lock);
        assert_eq!(paged.refcount(pinned_block).unwrap(), 2);
        assert_eq!(
            &paged.kv
                [pinned_block as usize * paged.stride..(pinned_block as usize + 1) * paged.stride],
            &pinned_bytes
        );
        assert!(!paged.free.contains(&pinned_block));
        assert!(paged.free.contains(&leaf_block));
        assert!(!paged.free.is_empty());
        for &id in &paged.free {
            assert_eq!(paged.refcount[id as usize], 0);
            assert!(!paged.sealed[id as usize]);
        }
        // A later alloc of a returned block is writable. refcount is 1 only
        // once the caller receives it.
        let mut next = paged.admit(&[8, 8]).unwrap();
        let got = paged.alloc_blocks(1).unwrap();
        assert_eq!(paged.refcount(got[0]).unwrap(), 1);
        paged.attach(&mut next, &got);
        let (k, v) = kv(0, 8);
        paged.write(&next, 0, 0, &k, &v).unwrap();
        paged.finish(pinned).unwrap();
        paged.finish(caller).unwrap();
        paged.finish(next).unwrap();
    }

    #[test]
    fn mid_edge_mismatch_seals_the_equal_prefix_only() {
        let mut paged = cache(2, 4);
        let mut a = paged.admit(&[1, 2, 3, 4]).unwrap();
        let ba = paged.alloc_blocks(2).unwrap();
        paged.attach(&mut a, &ba);
        write_all(&mut paged, &a, &[1, 2, 3, 4], 0);
        // B computes the same tokens before they are in the tree.
        let mut b = paged.admit(&[1, 2, 3, 4]).unwrap();
        let bb = paged.alloc_blocks(2).unwrap();
        paged.attach(&mut b, &bb);
        write_all(&mut paged, &b, &[1, 2, 3, 4], 0);
        let flipped = paged.offset(bb[1] as usize, 0, PLANE_K, 0, 0);
        paged.kv[flipped] += 1.0;
        paged.publish(&mut a, &[1, 2, 3, 4]).unwrap();
        let suffix = a.last_node();
        let err = paged.publish(&mut b, &[1, 2, 3, 4]);
        assert_eq!(err, Err(PagedError::Conflict));
        // Page 0 matched and is sealed. Page 1 stays private. The suffix id
        // is still the node A holds.
        assert_eq!(b.len(), 2);
        assert_eq!(a.last_node(), suffix);
        assert_eq!(paged.node_tokens(suffix).unwrap(), &[3, 4]);
        let prefix = b.last_node();
        assert_ne!(prefix, suffix);
        assert_eq!(paged.node_tokens(prefix).unwrap(), &[1, 2]);
        assert_eq!(b.blocks()[0], a.blocks()[0]);
        assert_eq!(b.blocks()[1], bb[1]);
        assert_eq!(paged.refcount(bb[0]).unwrap(), 0);
        assert!(paged.free.contains(&bb[0]));
        paged.finish(a).unwrap();
        paged.finish(b).unwrap();
    }
}
