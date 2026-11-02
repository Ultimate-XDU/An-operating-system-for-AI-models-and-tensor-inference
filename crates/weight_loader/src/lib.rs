//! WeightLoaderCore — SafeTensors / GGUF zero-copy mapping (Phase 1).
#![cfg_attr(not(test), no_std)]

extern crate alloc;

/// Placeholder: maps a weight file into a contiguous view.
pub trait WeightMap {
    fn tensor_bytes(&self, name: &str) -> Option<&[u8]>;
}
