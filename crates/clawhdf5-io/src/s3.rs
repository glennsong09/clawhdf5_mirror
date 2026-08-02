//! S3 range-GET backed I/O for companion-dataset Merkle proof retrieval (P2.5).
//!
//! S2-D2-Yr2 §7 (P2.5, "Cloud / WAN evaluation") and §"Cloud Object-Store
//! Evaluation": a proof path for one chunk needs only O(log N) 32-byte
//! companion-dataset node reads, retrievable as a single S3 HTTP range-GET
//! without downloading the data chunks or the full companion dataset. This
//! module wires that retrieval path: [`S3Reader`] implements
//! [`AsyncHDF5Read`] over real ranged `GetObject` calls, and
//! [`resolve_companion_layout`] / [`fetch_chunk_proof`] combine it with
//! `clawhdf5-format`'s existing (in-memory) HDF5 metadata parsers to resolve
//! and fetch exactly the proof-path bytes for one chunk.
//!
//! Gated behind the `s3` feature (pulls in `aws-sdk-s3` + `aws-config`).
//! Exercising this against real AWS infrastructure — as opposed to the
//! in-memory unit tests below, which cover the byte-offset arithmetic
//! without any network access — requires AWS credentials and an S3 bucket.
//! See `examples/cloud_eval.rs` for the P2.5 benchmark harness that produces
//! `benches/results/phase2-cloud.csv`.

use std::collections::BTreeMap;
use std::io;

use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;

use clawhdf5_format::data_layout::DataLayout;
use clawhdf5_format::group_v2::resolve_path_any;
use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature::find_signature;
use clawhdf5_format::superblock::Superblock;

use crate::async_read::AsyncHDF5Read;

/// Byte size of one Merkle tree node hash (mirrors
/// `clawhdf5_format::merkle::HASH_SIZE`, which is `pub(crate)` there).
pub const NODE_HASH_SIZE: usize = 32;

/// Identity of a single HDF5 object stored in S3.
#[derive(Debug, Clone)]
pub struct S3Location {
    /// Bucket name.
    pub bucket: String,
    /// Object key.
    pub key: String,
}

/// Async reader over a single S3 object using ranged `GetObject` requests.
///
/// Each [`AsyncHDF5Read::read_at`] call issues exactly one HTTP range-GET
/// (`Range: bytes=start-end`) against the object — no local caching and no
/// speculative prefetch beyond what the caller asks for. That is what makes
/// [`fetch_chunk_proof`]'s "O(log N) node reads as a single range request"
/// property visible in wall-clock and byte-count terms: what is measured is
/// what the S3 SDK actually transferred, not an estimate.
#[derive(Debug, Clone)]
pub struct S3Reader {
    client: Client,
    location: S3Location,
}

impl S3Reader {
    /// Build a reader for `location` using an already-configured S3 client.
    pub fn new(client: Client, location: S3Location) -> Self {
        Self { client, location }
    }

    /// Convenience constructor: loads credentials/region from the standard
    /// AWS environment (env vars, `~/.aws/config`, IMDS, ...) via
    /// `aws-config`'s default credential chain.
    pub async fn connect(bucket: impl Into<String>, key: impl Into<String>) -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = Client::new(&config);
        Self::new(
            client,
            S3Location {
                bucket: bucket.into(),
                key: key.into(),
            },
        )
    }

    /// The bucket/key this reader is bound to.
    pub fn location(&self) -> &S3Location {
        &self.location
    }

    /// The underlying S3 client, e.g. for issuing an upload via [`upload`].
    pub fn client(&self) -> &Client {
        &self.client
    }
}

impl AsyncHDF5Read for S3Reader {
    async fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let end = offset + len as u64 - 1;
        let resp = self
            .client
            .get_object()
            .bucket(&self.location.bucket)
            .key(&self.location.key)
            .range(format!("bytes={offset}-{end}"))
            .send()
            .await
            .map_err(|e| io::Error::other(format!("S3 GetObject failed: {e}")))?;
        let bytes = resp
            .body
            .collect()
            .await
            .map_err(|e| io::Error::other(format!("S3 body read failed: {e}")))?;
        Ok(bytes.into_bytes().to_vec())
    }

    async fn len(&self) -> io::Result<u64> {
        let resp = self
            .client
            .head_object()
            .bucket(&self.location.bucket)
            .key(&self.location.key)
            .send()
            .await
            .map_err(|e| io::Error::other(format!("S3 HeadObject failed: {e}")))?;
        Ok(resp.content_length().unwrap_or(0).max(0) as u64)
    }
}

/// Upload `data` to `location`, replacing any existing object.
///
/// Used by the P2.5 harness to stage test files before measuring
/// proof-retrieval latency; not on the read path itself. `data` up to a few
/// GB is fine as a single `PutObject`; S3's 5 GB single-request limit means
/// genuinely TB-scale surrogates need multipart upload, which is out of
/// scope here (see `examples/cloud_eval.rs` for the "surrogate, not literal
/// 1 TB" sizing this harness actually uses).
pub async fn upload(client: &Client, location: &S3Location, data: Vec<u8>) -> io::Result<()> {
    client
        .put_object()
        .bucket(&location.bucket)
        .key(&location.key)
        .body(ByteStream::from(data))
        .send()
        .await
        .map_err(|e| io::Error::other(format!("S3 PutObject failed: {e}")))?;
    Ok(())
}

/// The on-disk byte range of a contiguously-laid-out HDF5 dataset within its
/// backing object — e.g. a companion node array (`total_nodes * 32` bytes)
/// or a primary data dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousLayout {
    /// Absolute byte offset of the dataset's first byte in the object.
    pub address: u64,
    /// Total size of the dataset in bytes.
    pub size: u64,
}

/// Default size of the metadata prefix fetched by [`resolve_companion_layout`]
/// before the superblock/object-header/data-layout chain is resolved
/// locally. An HDF5 superblock plus a handful of small object headers
/// comfortably fits in this; if resolution fails with an
/// [`io::ErrorKind::InvalidData`] error whose message mentions parsing on an
/// unusually deep group hierarchy, retry with a larger prefix.
pub const DEFAULT_METADATA_PREFETCH: u64 = 256 * 1024;

/// Resolve the byte range of the companion dataset `merkle/{dataset_name}`
/// within `reader`'s backing object, fetching only a small metadata prefix
/// (`prefetch_len` bytes) rather than the whole object — the same technique
/// cloud-native readers (kerchunk, zarr reference filesystems) use to avoid
/// a full download just to find where the real data lives.
///
/// Returns [`io::ErrorKind::InvalidData`] if the companion dataset isn't
/// stored contiguously (e.g. it is still inline in the attribute, below
/// `clawhdf5_format::merkle::INLINE_CHUNK_THRESHOLD` chunks, so there is no
/// separate companion object to range-GET) or if `prefetch_len` didn't reach
/// far enough into the file to resolve the object header chain.
pub async fn resolve_companion_layout<R: AsyncHDF5Read>(
    reader: &R,
    dataset_name: &str,
    prefetch_len: u64,
) -> io::Result<ContiguousLayout> {
    resolve_contiguous_layout(reader, &format!("merkle/{dataset_name}"), prefetch_len).await
}

/// Like [`resolve_companion_layout`], but for an arbitrary dataset path
/// rather than always prefixing `merkle/`. Used by callers (e.g. the P2.5
/// harness) that also need the byte range of the *primary* data dataset —
/// to range-GET a single chunk's raw bytes for verification, or the whole
/// data region for a full-redownload baseline — via the same metadata-prefix
/// technique.
pub async fn resolve_contiguous_layout<R: AsyncHDF5Read>(
    reader: &R,
    path: &str,
    prefetch_len: u64,
) -> io::Result<ContiguousLayout> {
    let prefix = reader.read_at(0, prefetch_len as usize).await?;
    parse_contiguous_layout(&prefix, path)
}

/// Synchronous half of [`resolve_contiguous_layout`]: resolves `path`'s
/// on-disk byte range from an already-fetched metadata `prefix`, without
/// issuing any I/O itself.
///
/// Exposed so a caller that needs the layout of *several* paths within the
/// same object (e.g. the P2.5 harness resolving both `merkle/{name}` and
/// the primary dataset) can fetch the metadata prefix once and resolve both
/// from it, instead of paying for the prefix range-GET twice.
pub fn parse_contiguous_layout(prefix: &[u8], path: &str) -> io::Result<ContiguousLayout> {
    let sig_offset = find_signature(prefix)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let sb = Superblock::parse(prefix, sig_offset)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let addr = resolve_path_any(prefix, &sb, path)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let header = ObjectHeader::parse(prefix, addr as usize, sb.offset_size, sb.length_size)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let layout_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DataLayout)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dataset {path:?} object header has no data layout message"),
            )
        })?;
    let layout = DataLayout::parse(&layout_msg.data, sb.offset_size, sb.length_size)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    match layout {
        DataLayout::Contiguous {
            address: Some(address),
            size,
        } => Ok(ContiguousLayout { address, size }),
        DataLayout::Contiguous { address: None, .. } => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("dataset {path:?} has an undefined address (never written)"),
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "dataset {path:?} is not contiguous (inline or chunked layout is not supported \
                 for range-GET retrieval)"
            ),
        )),
    }
}

/// The sibling-node indices (level-order, root = 0) needed to prove
/// `leaf_idx` up to the root of a tree with `padded_leaf_count` leaves —
/// pure index arithmetic, ported from
/// `clawhdf5_format::subset_proof::extract_subset`'s per-leaf walk so it can
/// run without materializing the full in-memory `MerkleTree`.
fn proof_sibling_indices(leaf_idx: u64, padded_leaf_count: u64) -> Vec<u64> {
    let internal_nodes = padded_leaf_count - 1;
    let mut indices = Vec::new();
    let mut node_idx = internal_nodes + leaf_idx;
    while node_idx > 0 {
        let sibling_idx = if node_idx % 2 == 1 {
            node_idx + 1
        } else {
            node_idx - 1
        };
        indices.push(sibling_idx);
        node_idx = (node_idx - 1) / 2;
    }
    indices
}

/// A fetched Merkle proof path: the target leaf's own node plus its sibling
/// chain to the root, keyed by level-order node index.
#[derive(Debug, Clone)]
pub struct FetchedProof {
    /// Level-order node index of the leaf itself (`internal_nodes + leaf_idx`).
    pub leaf_node_index: u64,
    /// All fetched nodes (leaf + siblings), keyed by level-order index.
    pub nodes: BTreeMap<u64, [u8; NODE_HASH_SIZE]>,
    /// Bytes actually transferred over the wire for this proof (the size of
    /// the single covering range-GET, not `nodes.len() * 32` — the covering
    /// span may include node slots that weren't individually needed).
    pub bytes_transferred: u64,
}

/// Fetch the O(log N) proof path for chunk `leaf_idx` from the companion
/// dataset at `layout`, as **one** range-GET spanning the min..max node
/// index touched (P2.5 step 1: "issue them as one range request covering
/// the relevant slice of the companion dataset").
///
/// `padded_leaf_count` is the tree's padded leaf count (the next power of
/// two at or above the grid's total chunk count — see
/// `clawhdf5_format::subset_proof::ChunkGridParams::total_chunk_count`); the
/// caller supplies it rather than this function deriving it, because
/// deriving a trusted chunk count from an untrusted grid is the verifier's
/// job (see `verify_subset`'s grid-hash check), not the fetcher's.
pub async fn fetch_chunk_proof<R: AsyncHDF5Read>(
    reader: &R,
    layout: ContiguousLayout,
    leaf_idx: u64,
    padded_leaf_count: u64,
) -> io::Result<FetchedProof> {
    let internal_nodes = padded_leaf_count - 1;
    let leaf_node_index = internal_nodes + leaf_idx;

    let mut wanted: Vec<u64> = proof_sibling_indices(leaf_idx, padded_leaf_count);
    wanted.push(leaf_node_index);

    // `wanted` always has at least `leaf_node_index`, just pushed above.
    let lo = *wanted.iter().min().expect("wanted is non-empty");
    let hi = *wanted.iter().max().expect("wanted is non-empty");

    let span_start = layout.address + lo * NODE_HASH_SIZE as u64;
    let span_len = ((hi - lo + 1) * NODE_HASH_SIZE as u64) as usize;
    if span_start + span_len as u64 > layout.address + layout.size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "proof node span extends past the companion dataset's recorded size",
        ));
    }

    let span = reader.read_at(span_start, span_len).await?;
    if span.len() < span_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short read while fetching proof node span",
        ));
    }

    let mut nodes = BTreeMap::new();
    for idx in wanted {
        let rel = ((idx - lo) * NODE_HASH_SIZE as u64) as usize;
        let mut hash = [0u8; NODE_HASH_SIZE];
        hash.copy_from_slice(&span[rel..rel + NODE_HASH_SIZE]);
        nodes.insert(idx, hash);
    }

    Ok(FetchedProof {
        leaf_node_index,
        nodes,
        bytes_transferred: span_len as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_read::AsyncMemoryReader;
    use clawhdf5_format::file_writer::FileWriter as FmtWriter;
    use clawhdf5_format::merkle::{
        HashAlg, MerkleCompanionResult, MerkleTree, write_merkle_companion,
    };

    /// Build an in-memory HDF5 file containing a `merkle/{name}` companion
    /// dataset for `n_chunks` synthetic chunks, mirroring
    /// `clawhdf5_format::merkle::test_merkle_roundtrip_1024_chunks`'s setup
    /// so these tests exercise the *real* on-disk layout `s3.rs` parses —
    /// without any network access (that's what real S3 credentials are for;
    /// see `examples/cloud_eval.rs`).
    fn make_test_file_with_companion(name: &str, n_chunks: usize) -> (Vec<u8>, MerkleTree) {
        let chunks: Vec<Vec<u8>> = (0..n_chunks)
            .map(|i| {
                let mut chunk = vec![0u8; 64];
                for (j, byte) in chunk.iter_mut().enumerate() {
                    *byte = ((i + j) % 256) as u8;
                }
                chunk
            })
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let mut fw = FmtWriter::new();
        let result = write_merkle_companion(&mut fw, name, &tree)
            .expect("write_merkle_companion should succeed");
        assert!(
            matches!(result, MerkleCompanionResult::Dataset { .. }),
            "test requires a Dataset (non-inline) companion; increase n_chunks"
        );

        let ds = fw.create_dataset(name);
        let all_data: Vec<u8> = chunks.iter().flatten().copied().collect();
        ds.with_u8_data(&all_data);

        let file_bytes = fw.finish().expect("file should build");
        (file_bytes, tree)
    }

    #[tokio::test]
    async fn resolve_companion_layout_finds_contiguous_dataset() {
        let (file_bytes, tree) = make_test_file_with_companion("sensor_data", 1024);
        let reader = AsyncMemoryReader::new(file_bytes);

        let layout = resolve_companion_layout(&reader, "sensor_data", DEFAULT_METADATA_PREFETCH)
            .await
            .expect("layout resolution should succeed");

        let expected_size = tree.nodes().len() as u64 * NODE_HASH_SIZE as u64;
        assert_eq!(layout.size, expected_size);
        assert!(layout.address > 0);
    }

    #[tokio::test]
    async fn resolve_companion_layout_missing_dataset_errors() {
        let (file_bytes, _tree) = make_test_file_with_companion("sensor_data", 1024);
        let reader = AsyncMemoryReader::new(file_bytes);

        let err = resolve_companion_layout(&reader, "does_not_exist", DEFAULT_METADATA_PREFETCH)
            .await
            .expect_err("nonexistent companion dataset should error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn fetch_chunk_proof_matches_in_memory_tree() {
        let (file_bytes, tree) = make_test_file_with_companion("sensor_data", 1024);
        let reader = AsyncMemoryReader::new(file_bytes);

        let layout = resolve_companion_layout(&reader, "sensor_data", DEFAULT_METADATA_PREFETCH)
            .await
            .expect("layout resolution should succeed");

        let padded_leaf_count = tree.padded_leaf_count() as u64;
        for leaf_idx in [0u64, 1, 500, 1023] {
            let fetched = fetch_chunk_proof(&reader, layout, leaf_idx, padded_leaf_count)
                .await
                .expect("proof fetch should succeed");

            let expected_leaf_hash = *tree.leaf_hash(leaf_idx as usize).unwrap();
            let internal_nodes = padded_leaf_count - 1;
            assert_eq!(fetched.leaf_node_index, internal_nodes + leaf_idx);
            assert_eq!(fetched.nodes[&fetched.leaf_node_index], expected_leaf_hash);

            // Every fetched sibling must match the in-memory tree's own
            // node array at the same level-order index.
            for (&idx, &hash) in &fetched.nodes {
                assert_eq!(hash, tree.nodes()[idx as usize], "mismatch at node {idx}");
            }

            // O(log N): for a 1024-leaf tree (depth 10), the proof touches
            // at most leaf + 10 siblings.
            assert!(fetched.nodes.len() <= 11);
        }
    }

    #[tokio::test]
    async fn fetch_chunk_proof_rejects_out_of_range_span() {
        let (file_bytes, tree) = make_test_file_with_companion("sensor_data", 1024);
        let reader = AsyncMemoryReader::new(file_bytes);
        let layout = resolve_companion_layout(&reader, "sensor_data", DEFAULT_METADATA_PREFETCH)
            .await
            .unwrap();

        let padded_leaf_count = tree.padded_leaf_count() as u64;
        let err = fetch_chunk_proof(&reader, layout, padded_leaf_count, padded_leaf_count)
            .await
            .expect_err("leaf index beyond padded_leaf_count should error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn proof_sibling_indices_depth_matches_tree_height() {
        // 8-leaf tree: internal_nodes = 7, leaf 0 -> node 7.
        // Siblings walked: 7's sibling 8, then (3-1)/2... verify against
        // the same arithmetic MerkleTree::proof() uses.
        let siblings = proof_sibling_indices(0, 8);
        assert_eq!(siblings.len(), 3); // log2(8) = 3 levels
    }
}
