//! Read-only host map of a SafeTensors file.
//!
//! Compiled with the `host` feature and during this crate's tests. The default
//! `no_std` build does not include it. [`SafeTensorsMmap`] owns the mapping;
//! [`SafeTensorsWeights`](crate::SafeTensorsWeights) borrows tensor payloads
//! from those pages.

use std::fmt;
use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::Mmap;

use crate::{SafeTensorsWeights, WeightError};

/// Failure opening or parsing a SafeTensors file on the host.
#[derive(Debug)]
pub enum WeightFileError {
    /// The path could not be opened or mapped.
    Io(io::Error),
    /// The mapped bytes are not a SafeTensors file this loader accepts.
    Weight(WeightError),
}

impl fmt::Display for WeightFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightFileError::Io(err) => write!(f, "{err}"),
            WeightFileError::Weight(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for WeightFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WeightFileError::Io(err) => Some(err),
            // `WeightError` has no `std::error::Error` impl; it is `no_std`.
            WeightFileError::Weight(_) => None,
        }
    }
}

impl From<io::Error> for WeightFileError {
    fn from(err: io::Error) -> Self {
        WeightFileError::Io(err)
    }
}

impl From<WeightError> for WeightFileError {
    fn from(err: WeightError) -> Self {
        WeightFileError::Weight(err)
    }
}

/// Read-only map of a `.safetensors` file.
///
/// [`Self::parse`] borrows the map. Tensor payloads are not copied into a
/// `Vec<u8>`. The header index inside [`SafeTensorsWeights`] still owns names
/// and shapes; that is separate from the payload bytes.
///
/// Do not truncate or replace the file while this value is alive. The mapping
/// would then be invalid, and any live tensor views would be too.
pub struct SafeTensorsMmap {
    mmap: Mmap,
}

impl fmt::Debug for SafeTensorsMmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SafeTensorsMmap")
            .field("len", &self.mmap.len())
            .finish()
    }
}

impl SafeTensorsMmap {
    /// Map `path` read-only.
    ///
    /// `memmap2` turns an empty file into an empty slice. That still fails in
    /// [`Self::parse`], as does any other buffer that is not SafeTensors.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WeightFileError> {
        let file = File::open(path.as_ref())?;
        // SAFETY: `file` is opened read-only and this map is not `MmapMut`, so
        // this process does not write through the mapping. Callers must not
        // truncate or replace the file while `SafeTensorsMmap` (and any tensor
        // views borrowed from it) is alive; that is undefined for a POSIX map.
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(Self { mmap })
    }

    /// Mapped file bytes, including the SafeTensors header.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Parse the map as SafeTensors. Payload slices point into [`Self::as_bytes`].
    pub fn parse(&self) -> Result<SafeTensorsWeights<'_>, WeightError> {
        SafeTensorsWeights::parse(self.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{load_f32_tensor, DType, WeightMap};
    use safetensors::tensor::TensorView as StView;
    use safetensors::{Dtype as StDtype, SafeTensorError};
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn le_f32(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn write_fixture(
        tensors: &[(&str, StDtype, Vec<usize>, &[u8])],
    ) -> Result<Vec<u8>, SafeTensorError> {
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                StView::new(*dtype, shape.clone(), bytes).map(|view| (*name, view))
            })
            .collect::<Result<Vec<_>, _>>()?;
        safetensors::serialize(views, None)
    }

    fn assert_borrowed_from(file: &[u8], view: &[u8]) {
        assert!(!view.is_empty());
        let file_start = file.as_ptr() as usize;
        let file_end = file_start + file.len();
        let view_start = view.as_ptr() as usize;
        let view_end = view_start + view.len();
        assert!(view_start >= file_start && view_end <= file_end);
    }

    struct TempFile(PathBuf);

    impl TempFile {
        fn write(label: &str, bytes: &[u8]) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "myos-weight-loader-{label}-{}-{nanos}.safetensors",
                std::process::id()
            ));
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn open_mmap_parse_borrows_payloads_from_the_file() {
        let f32_bytes = le_f32(&[1.0, -2.5]);
        let f16_bytes = [0x00, 0x3C, 0x00, 0xC0];
        let file = write_fixture(&[
            ("w", StDtype::F32, vec![2], &f32_bytes),
            ("scale", StDtype::F16, vec![2], &f16_bytes),
        ])
        .unwrap();
        // Declared first so the map drops before the file is unlinked.
        let tmp = TempFile::write("borrow", &file);

        let mapped = SafeTensorsMmap::open(tmp.path()).unwrap();
        assert_eq!(mapped.as_bytes(), file.as_slice());

        let weights = mapped.parse().unwrap();
        assert_eq!(weights.len(), 2);
        assert_eq!(weights.names().collect::<Vec<_>>(), vec!["scale", "w"]);

        let w = weights.tensor_bytes("w").unwrap();
        let scale = weights.tensor_bytes("scale").unwrap();
        assert_eq!(w, f32_bytes.as_slice());
        assert_eq!(scale, f16_bytes.as_slice());
        assert_borrowed_from(mapped.as_bytes(), w);
        assert_borrowed_from(mapped.as_bytes(), scale);
        assert_ne!(w.as_ptr(), scale.as_ptr());
        assert!(w.len() < mapped.as_bytes().len());

        let info = weights.tensor_info("scale").unwrap();
        assert_eq!(info.dtype, DType::F16);
        assert_eq!(info.shape, vec![2]);

        if cfg!(target_endian = "little") {
            let (info, data) = load_f32_tensor(&weights, "w").unwrap();
            assert_eq!(info.shape, vec![2]);
            assert_eq!(data, vec![1.0, -2.5]);
            assert_eq!(
                load_f32_tensor(&weights, "scale"),
                Err(WeightError::DTypeMismatch)
            );
        }
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let path = std::env::temp_dir().join(format!(
            "myos-weight-loader-missing-{}-{}.safetensors",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let err = SafeTensorsMmap::open(&path).unwrap_err();
        match err {
            WeightFileError::Io(err) => assert_eq!(err.kind(), io::ErrorKind::NotFound),
            WeightFileError::Weight(_) => panic!("expected an I/O error"),
        }
    }

    #[test]
    fn empty_file_maps_and_fails_parse() {
        let tmp = TempFile::write("empty", b"");
        let mapped = SafeTensorsMmap::open(tmp.path()).unwrap();
        assert!(mapped.as_bytes().is_empty());
        assert_eq!(mapped.parse().unwrap_err(), WeightError::Parse);
    }

    #[test]
    fn truncated_file_maps_and_fails_parse() {
        let tmp = TempFile::write("truncated", &[0xff, 0, 0, 0, 0, 0, 0, 0]);
        let mapped = SafeTensorsMmap::open(tmp.path()).unwrap();
        assert_eq!(mapped.as_bytes().len(), 8);
        assert_eq!(mapped.parse().unwrap_err(), WeightError::Parse);
    }
}
