//! ModelCore — unified MinimalistModel trait (Phase 1).
#![cfg_attr(not(test), no_std)]

extern crate alloc;

/// Phase-1 model surface: one token forward.
pub trait MinimalistModel {
    type KvCache;
    fn forward_token(&mut self, token: u32, pos: usize, kv: &mut Self::KvCache) -> &[f32];
}
