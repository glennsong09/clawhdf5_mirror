//! Comparable-primitive baseline integrity backends (S2-D2 task P1.2b).
//!
//! These three backends are *not* part of ClawHDF5's Merkle-tree provenance
//! design — they exist so the Merkle tree's benchmark results (RQ1–RQ3) can
//! be reported relative to measured numbers for the nearest comparable
//! primitives, rather than asserted to be better in the abstract:
//!
//! - [`FlatHashBackend`] — a single SHA-256 over the whole dataset. This is
//!   ClawHDF5's current provenance feature (the status quo).
//! - [`PerChunkMacBackend`] — keyed HMAC-SHA-256 per chunk.
//! - [`PerChunkSigBackend`] — Ed25519 per chunk. The closest public-verifiable
//!   competitor to the Merkle tree (it satisfies R1, R4, R5a, and R6 but not
//!   R7/R10 — see §4.3). An ML-DSA-65 hybrid variant is deferred to Phase 2,
//!   once the P2.1 signing crate lands; it is intentionally not stubbed here.
//!
//! Each backend mirrors the Merkle verification API
//! (`commit`/`verify_chunk`/`verify_dataset`/`append`/`update`) so a single
//! benchmark harness can drive all of them identically. Where a primitive
//! structurally cannot perform an operation, the method returns
//! [`BaselineError::Unsupported`] rather than silently degrading to a more
//! expensive equivalent — that gap is itself part of the measurement.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "ed25519_dalek")]
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
#[cfg(feature = "hmac")]
use hmac::{Hmac, Mac};
#[cfg(feature = "rand_core")]
use rand_core::OsRng;
#[cfg(feature = "sha2")]
use sha2::{Digest, Sha256};

/// Errors that can occur during baseline backend operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineError {
    /// The backend structurally cannot perform the requested operation
    /// (e.g. a flat hash cannot verify a single chunk without rehashing the
    /// whole dataset).
    Unsupported {
        /// Name of the backend that was asked to perform the operation.
        backend: &'static str,
        /// Name of the unsupported operation.
        op: &'static str,
    },
    /// The requested chunk index lies outside the committed range. This is a
    /// distinct condition from a verification mismatch (which is reported as
    /// `Ok(false)`): there is simply no chunk at `chunk_idx` to check against.
    OutOfRange {
        /// Name of the backend that was queried.
        backend: &'static str,
        /// The offending chunk index.
        chunk_idx: usize,
    },
}

impl core::fmt::Display for BaselineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BaselineError::Unsupported { backend, op } => {
                write!(
                    f,
                    "backend '{}' does not support operation '{}'",
                    backend, op
                )
            }
            BaselineError::OutOfRange { backend, chunk_idx } => {
                write!(
                    f,
                    "backend '{}' has no chunk at index {}",
                    backend, chunk_idx
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BaselineError {}

/// Constant-time comparison of two equal-length byte slices.
///
/// Prevents timing attacks by always comparing all bytes regardless of
/// where the first difference occurs.
#[inline]
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

/// Reference backend: a single SHA-256 over the whole dataset.
///
/// This is ClawHDF5's current provenance feature (the status quo). It has
/// constant storage overhead but cannot verify a single chunk, append, or
/// update without rehashing the entire dataset — O(N) for every mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatHashBackend {
    hash: [u8; 32],
}

#[cfg(feature = "sha2")]
impl FlatHashBackend {
    /// Compute a single SHA-256 over all chunks, in order.
    #[must_use]
    pub fn commit(chunks: &[&[u8]]) -> Self {
        let mut hasher = Sha256::new();
        for chunk in chunks {
            hasher.update(chunk);
        }
        Self {
            hash: hasher.finalize().into(),
        }
    }

    /// Always returns `Err(Unsupported)`: a flat hash cannot check one
    /// chunk without rehashing the whole dataset, so this is reported as a
    /// capability gap rather than silently routed through
    /// [`Self::verify_dataset`].
    pub fn verify_chunk(&self, _idx: usize, _data: &[u8]) -> Result<bool, BaselineError> {
        Err(BaselineError::Unsupported {
            backend: "flat_hash",
            op: "verify_chunk",
        })
    }

    /// Rehash all chunks and compare against the committed digest.
    #[must_use]
    pub fn verify_dataset(&self, chunks: &[&[u8]]) -> bool {
        let recomputed = Self::commit(chunks).hash;
        constant_time_eq(&recomputed, &self.hash)
    }

    /// Recompute the digest over the full, updated chunk list.
    ///
    /// This *is* the O(N) cost the benchmark is meant to capture: a flat
    /// hash has no incremental update path.
    pub fn append(&mut self, all_chunks: &[&[u8]]) {
        *self = Self::commit(all_chunks);
    }

    /// Recompute the digest over the full, updated chunk list.
    pub fn update(&mut self, all_chunks: &[&[u8]]) {
        *self = Self::commit(all_chunks);
    }

    /// Bytes of integrity metadata: always 32, independent of chunk count.
    #[must_use]
    pub fn metadata_bytes(&self) -> usize {
        32
    }
}

/// Reference backend: keyed HMAC-SHA-256 per chunk.
///
/// Storage and commit cost scale linearly with chunk count, and append /
/// update are O(1). The capability gap relative to a public-key scheme is
/// that verification requires possession of the shared secret key — no
/// third party without the key can verify, which is recorded as an explicit
/// benchmark capability entry rather than a missing method here.
#[cfg(feature = "hmac")]
pub struct PerChunkMacBackend {
    key: [u8; 32],
    tags: Vec<[u8; 32]>,
}

#[cfg(feature = "hmac")]
type HmacSha256 = Hmac<Sha256>;

#[cfg(feature = "hmac")]
impl PerChunkMacBackend {
    fn tag(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
        let mut mac =
            <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC accepts any key length");
        mac.update(data);
        mac.finalize().into_bytes().into()
    }

    /// Generate a random 32-byte key and compute `HMAC-SHA256(key, chunk_i)`
    /// for every chunk.
    #[cfg(feature = "rand_core")]
    #[must_use]
    pub fn commit(chunks: &[&[u8]]) -> Self {
        use rand_core::RngCore;
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        let tags = chunks.iter().map(|c| Self::tag(&key, c)).collect();
        Self { key, tags }
    }

    /// Recompute the HMAC for chunk `idx` with the held key and compare.
    pub fn verify_chunk(&self, idx: usize, data: &[u8]) -> Result<bool, BaselineError> {
        let Some(expected) = self.tags.get(idx) else {
            return Err(BaselineError::OutOfRange {
                backend: "per_chunk_mac",
                chunk_idx: idx,
            });
        };
        Ok(constant_time_eq(&Self::tag(&self.key, data), expected))
    }

    /// Verify every chunk against its stored tag.
    #[must_use]
    pub fn verify_dataset(&self, chunks: &[&[u8]]) -> bool {
        chunks.len() == self.tags.len()
            && chunks
                .iter()
                .enumerate()
                .all(|(i, c)| matches!(self.verify_chunk(i, c), Ok(true)))
    }

    /// O(1): compute and push the tag for one newly appended chunk.
    pub fn append(&mut self, data: &[u8]) {
        self.tags.push(Self::tag(&self.key, data));
    }

    /// O(1): recompute the tag for an in-place chunk overwrite. Returns
    /// `OutOfRange` rather than panicking if `idx` has not been committed.
    pub fn update(&mut self, idx: usize, data: &[u8]) -> Result<(), BaselineError> {
        if idx >= self.tags.len() {
            return Err(BaselineError::OutOfRange {
                backend: "per_chunk_mac",
                chunk_idx: idx,
            });
        }
        self.tags[idx] = Self::tag(&self.key, data);
        Ok(())
    }

    /// Bytes of integrity metadata: one 32-byte tag per chunk.
    #[must_use]
    pub fn metadata_bytes(&self) -> usize {
        self.tags.len() * 32
    }
}

/// Reference backend: Ed25519 signature per chunk.
///
/// The closest public-verifiable competitor to the Merkle tree: it satisfies
/// R1 (sub-dataset partial verification), R4 (tamper localization), R5a
/// (per-chunk authenticity), and R6 (public verifiability via the public
/// key alone), but differs sharply on R7 (compact subset proof — see
/// [`Self::subset_proof_bytes`]) and R10 (HPC-feasible build cost) per §4.3.
#[cfg(feature = "ed25519_dalek")]
pub struct PerChunkSigBackend {
    #[allow(dead_code)]
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    signatures: Vec<Signature>,
}

#[cfg(feature = "ed25519_dalek")]
impl PerChunkSigBackend {
    /// Generate an Ed25519 keypair and sign every chunk independently.
    #[cfg(feature = "rand_core")]
    #[must_use]
    pub fn commit(chunks: &[&[u8]]) -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let signatures = chunks.iter().map(|c| signing_key.sign(c)).collect();
        Self {
            signing_key,
            verifying_key,
            signatures,
        }
    }

    /// Verify chunk `idx` against its stored signature using the public key
    /// alone — no secret needed.
    pub fn verify_chunk(&self, idx: usize, data: &[u8]) -> Result<bool, BaselineError> {
        let Some(sig) = self.signatures.get(idx) else {
            return Err(BaselineError::OutOfRange {
                backend: "per_chunk_sig",
                chunk_idx: idx,
            });
        };
        Ok(self.verifying_key.verify(data, sig).is_ok())
    }

    /// Verify every chunk against its stored signature.
    #[must_use]
    pub fn verify_dataset(&self, chunks: &[&[u8]]) -> bool {
        chunks.len() == self.signatures.len()
            && chunks
                .iter()
                .enumerate()
                .all(|(i, c)| matches!(self.verify_chunk(i, c), Ok(true)))
    }

    /// O(1): sign one newly appended chunk.
    pub fn append(&mut self, data: &[u8]) {
        self.signatures.push(self.signing_key.sign(data));
    }

    /// O(1): re-sign an in-place chunk overwrite. Returns `OutOfRange` rather
    /// than panicking if `idx` has not been committed.
    pub fn update(&mut self, idx: usize, data: &[u8]) -> Result<(), BaselineError> {
        if idx >= self.signatures.len() {
            return Err(BaselineError::OutOfRange {
                backend: "per_chunk_sig",
                chunk_idx: idx,
            });
        }
        self.signatures[idx] = self.signing_key.sign(data);
        Ok(())
    }

    /// Bytes of integrity metadata: one 64-byte signature per chunk.
    #[must_use]
    pub fn metadata_bytes(&self) -> usize {
        self.signatures.len() * 64
    }

    /// Bytes a recipient must receive to verify a `k`-chunk hyperslab.
    ///
    /// Per-chunk signatures have no shared-node compaction (unlike a Merkle
    /// proof), so a `k`-chunk hyperslab costs exactly `k` signatures —
    /// `O(k)`, not Merkle's `O(k log N)` (R7, §4.3).
    #[must_use]
    pub fn subset_proof_bytes(k: usize) -> usize {
        k * 64
    }
}

#[cfg(all(test, feature = "baselines"))]
mod tests {
    use super::*;

    fn chunks() -> Vec<Vec<u8>> {
        (0u8..8).map(|i| vec![i; 16]).collect()
    }

    fn refs(chunks: &[Vec<u8>]) -> Vec<&[u8]> {
        chunks.iter().map(|c| c.as_slice()).collect()
    }

    #[test]
    fn flat_hash_round_trip() {
        let data = chunks();
        let r = refs(&data);
        let backend = FlatHashBackend::commit(&r);
        assert!(backend.verify_dataset(&r));
    }

    #[test]
    fn flat_hash_detects_tamper() {
        let data = chunks();
        let r = refs(&data);
        let backend = FlatHashBackend::commit(&r);

        let mut tampered = data.clone();
        tampered[3][0] ^= 0xFF;
        let rt = refs(&tampered);
        assert!(!backend.verify_dataset(&rt));
    }

    #[test]
    fn flat_hash_verify_chunk_unsupported() {
        let data = chunks();
        let r = refs(&data);
        let backend = FlatHashBackend::commit(&r);
        assert_eq!(
            backend.verify_chunk(0, &data[0]),
            Err(BaselineError::Unsupported {
                backend: "flat_hash",
                op: "verify_chunk"
            })
        );
    }

    #[test]
    fn flat_hash_metadata_bytes_constant_in_n() {
        let small = chunks();
        let mut large = chunks();
        large.extend(chunks());
        assert_eq!(
            FlatHashBackend::commit(&refs(&small)).metadata_bytes(),
            FlatHashBackend::commit(&refs(&large)).metadata_bytes()
        );
    }

    #[test]
    fn mac_round_trip_and_tamper_detection() {
        let data = chunks();
        let r = refs(&data);
        let backend = PerChunkMacBackend::commit(&r);
        assert!(backend.verify_dataset(&r));
        assert_eq!(backend.verify_chunk(2, &data[2]), Ok(true));
        assert_eq!(backend.verify_chunk(2, b"wrong"), Ok(false));
    }

    #[test]
    fn mac_append_and_update_are_local() {
        let data = chunks();
        let r = refs(&data);
        let mut backend = PerChunkMacBackend::commit(&r);
        let before = backend.metadata_bytes();

        backend.append(b"new chunk");
        assert_eq!(backend.metadata_bytes(), before + 32);
        assert_eq!(backend.verify_chunk(data.len(), b"new chunk"), Ok(true));

        backend.update(0, b"overwritten").unwrap();
        assert_eq!(backend.verify_chunk(0, b"overwritten"), Ok(true));
        assert_eq!(backend.verify_chunk(1, &data[1]), Ok(true));
    }

    #[test]
    fn mac_out_of_range_chunk_is_distinct_from_mismatch() {
        let data = chunks();
        let mut backend = PerChunkMacBackend::commit(&refs(&data));
        let oob = data.len();
        assert_eq!(
            backend.verify_chunk(oob, b"whatever"),
            Err(BaselineError::OutOfRange {
                backend: "per_chunk_mac",
                chunk_idx: oob,
            })
        );
        assert_eq!(
            backend.update(oob, b"whatever"),
            Err(BaselineError::OutOfRange {
                backend: "per_chunk_mac",
                chunk_idx: oob,
            })
        );
        // A committed-but-tampered chunk is a mismatch, not OutOfRange.
        assert_eq!(backend.verify_chunk(0, b"tampered"), Ok(false));
    }

    #[test]
    fn mac_metadata_bytes_scales_linearly() {
        let data = chunks();
        let backend = PerChunkMacBackend::commit(&refs(&data));
        assert_eq!(backend.metadata_bytes(), data.len() * 32);
    }

    #[test]
    fn sig_round_trip_and_tamper_detection() {
        let data = chunks();
        let r = refs(&data);
        let backend = PerChunkSigBackend::commit(&r);
        assert!(backend.verify_dataset(&r));
        assert_eq!(backend.verify_chunk(2, &data[2]), Ok(true));
        assert_eq!(backend.verify_chunk(2, b"wrong"), Ok(false));
    }

    #[test]
    fn sig_append_and_update_are_local() {
        let data = chunks();
        let r = refs(&data);
        let mut backend = PerChunkSigBackend::commit(&r);
        let before = backend.metadata_bytes();

        backend.append(b"new chunk");
        assert_eq!(backend.metadata_bytes(), before + 64);
        assert_eq!(backend.verify_chunk(data.len(), b"new chunk"), Ok(true));

        backend.update(0, b"overwritten").unwrap();
        assert_eq!(backend.verify_chunk(0, b"overwritten"), Ok(true));
        assert_eq!(backend.verify_chunk(1, &data[1]), Ok(true));
    }

    #[test]
    fn sig_out_of_range_chunk_is_distinct_from_mismatch() {
        let data = chunks();
        let mut backend = PerChunkSigBackend::commit(&refs(&data));
        let oob = data.len();
        assert_eq!(
            backend.verify_chunk(oob, b"whatever"),
            Err(BaselineError::OutOfRange {
                backend: "per_chunk_sig",
                chunk_idx: oob,
            })
        );
        assert_eq!(
            backend.update(oob, b"whatever"),
            Err(BaselineError::OutOfRange {
                backend: "per_chunk_sig",
                chunk_idx: oob,
            })
        );
        // A committed-but-tampered chunk is a mismatch, not OutOfRange.
        assert_eq!(backend.verify_chunk(0, b"tampered"), Ok(false));
    }

    #[test]
    fn sig_rejects_wrong_keypair() {
        let data = chunks();
        let r = refs(&data);
        let a = PerChunkSigBackend::commit(&r);
        let b = PerChunkSigBackend::commit(&r);
        // Cross-checking chunk 0's data against the *other* backend's
        // signature must fail: signatures are bound to their own key.
        assert_eq!(
            b.verifying_key.verify(&data[0], &a.signatures[0]).is_ok(),
            false
        );
    }

    #[test]
    fn sig_metadata_bytes_scales_linearly() {
        let data = chunks();
        let backend = PerChunkSigBackend::commit(&refs(&data));
        assert_eq!(backend.metadata_bytes(), data.len() * 64);
    }

    #[test]
    fn sig_subset_proof_bytes_is_linear_in_k_not_log_n() {
        assert_eq!(PerChunkSigBackend::subset_proof_bytes(5), 5 * 64);
        assert_eq!(PerChunkSigBackend::subset_proof_bytes(0), 0);
    }
}
