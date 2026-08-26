//! Two-tier (coarse-then-drill-down) verification for chunked datasets.
//!
//! The contiguous layout gets its leaf granularity from a *verification grid*
//! laid over the byte stream ([`crate::verification_grid`]). The chunked
//! layout has had no analogue: leaf granularity is hard-wired to one native
//! HDF5 chunk. This module supplies the analogue -- an **overlay mesh** that
//! groups `k = 2^level` SFC-adjacent native chunks into a single addressable
//! *coarse leaf* for routine checks, while leaving the full per-chunk tree
//! underneath completely untouched.
//!
//! # Why this needs no new tree
//!
//! For any quadrant-recursive curve every aligned `2^k` range of leaf indices
//! is already an axis-aligned rectangle, so a flat Merkle tree over
//! SFC-ordered leaves *is* a Merkle k-d tree (see
//! `subset_proof::hilbert_index` and the `aligned_hilbert_range_is_a_rectangle`
//! test). "Coarsening" is therefore not a second structure: it is the decision
//! to address the tree at the ancestor row `level` steps above the native leaf
//! row. Concretely, in the level-order layout the tree already uses, the coarse
//! row of a `padded_count`-leaf tree is the row of `padded_count >> level`
//! nodes whose first index is `(padded_count >> level) - 1` -- exactly the
//! shape of a `padded_count >> level`-leaf tree's own leaf row, with identical
//! parent arithmetic. That is what lets [`extract_subset_mesh`] reuse
//! `subset_proof`'s witness machinery verbatim at the coarse row, with no tree
//! walk introduced, no companion-layout change, and no extra stored nodes.
//!
//! # What is and is not delivered
//!
//! The mesh is *metadata only*. The delivered payload is still the native
//! covering chunk set -- the verifier hashes exactly the same bytes it would
//! have hashed at `level = 0`. What shrinks is the wire: `M/k` coarse hashes
//! and a witness set drawn from a tree `level` rows shorter, instead of `M`
//! leaf hashes and a full-height witness set.
//!
//! # Partially covered groups, and why unallocated chunks are free
//!
//! A coarse group straddling the edge of a selection contains members the
//! request does not deliver. Their leaf hashes must reach the verifier, as
//! [`MeshProof::fill_in`] -- authenticated like any other witness, since
//! tampering with one changes the coarse hash and breaks the chain to the
//! signed root.
//!
//! Members that are *unallocated* (sparse chunks) or *padding* (beyond the
//! real chunk count) are the exception: their leaf hash is the null sentinel
//! `H(0x02 || "null")`, a public constant. [`extract_subset_mesh`] omits
//! those and the verifier reconstructs them, so a sparse dataset pays no
//! fill-in at all. Omitting a *non*-null hash is not an attack either --
//! the verifier substitutes the sentinel, computes a different coarse hash,
//! and the root check fails.
//!
//! # Localization
//!
//! Coarsening trades localization granularity for wire size: a rejected
//! coarse leaf indicts `k` chunks, not one. [`localize_in_group`] recovers
//! the exact chunks by descending the group, following every branch whose
//! recomputed value disagrees with the companion -- `2` hashes per visited
//! node, so `O(log k)` for a single corruption and `O(b log k)` for `b` of
//! them, in `level` rounds. Worst case is unchanged at `O(log N)`.

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, vec, vec::Vec};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

use crate::merkle::{HASH_SIZE, HashAlg, MerkleError, MerkleTree, constant_time_eq};
use crate::selection::Selection;
use crate::subset_proof::{
    ChunkData, ChunkGridParams, LeafOrder, ProofConstruction, checked_padded_leaf_count,
    compute_expected_chunk_indices, compute_grid_hash, pruned_sibling_indices,
};

/// Bytes charged per index in the wire accounting (a `u64` leaf/node index).
/// Matches `subset_proof_geometry_bench`'s accounting so mesh and native
/// proof sizes are directly comparable.
const INDEX_BYTES: usize = 8;

/// How many native chunks one coarse verification leaf covers, as a power of
/// two: group size `k = 2^level`. `level = 0` is the native per-chunk tree and
/// reproduces `subset_proof`'s behaviour exactly.
///
/// A power of two is not a simplification but a requirement: only an aligned
/// `2^level` range of leaf indices is a node of the existing tree, and only
/// such a range is an axis-aligned rectangle under a quadrant-recursive curve.
/// A non-power-of-two group would be neither, and would need a second tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MeshLevel(u32);

impl MeshLevel {
    /// The native per-chunk tree: one coarse leaf per HDF5 chunk.
    pub const NATIVE: MeshLevel = MeshLevel(0);

    /// A mesh whose coarse leaves each cover `2^level` native chunks.
    ///
    /// Unchecked against any particular tree; [`extract_subset_mesh`] and
    /// [`verify_subset_mesh`] reject a level taller than the tree they are
    /// given. A level no tree this crate will build could reach is rejected
    /// here.
    ///
    /// # Errors
    ///
    /// [`MerkleError::MeshLevelTooCoarse`] if `level` exceeds 63, which no
    /// tree this crate will build can reach.
    pub fn new(level: u32) -> Result<Self, MerkleError> {
        if level > 63 {
            return Err(MerkleError::MeshLevelTooCoarse { level, height: 63 });
        }
        Ok(Self(level))
    }

    /// The mesh whose groups hold `k` native chunks.
    ///
    /// # Errors
    ///
    /// [`MerkleError::MeshLevelTooCoarse`] if `k` is zero or not a power of
    /// two -- see the type-level note on why that is structural.
    pub fn from_group_size(k: u64) -> Result<Self, MerkleError> {
        if k == 0 || !k.is_power_of_two() {
            return Err(MerkleError::MeshLevelTooCoarse {
                level: u32::MAX,
                height: 0,
            });
        }
        Self::new(k.trailing_zeros())
    }

    /// The coarsening level (`k = 2^level`).
    #[must_use]
    pub const fn level(self) -> u32 {
        self.0
    }

    /// Native chunks per coarse leaf.
    #[must_use]
    pub const fn group_size(self) -> u64 {
        1u64 << self.0
    }

    /// The coarse leaf that native leaf `leaf` belongs to.
    #[must_use]
    pub const fn coarse_index(self, leaf: usize) -> usize {
        leaf >> self.0
    }

    /// The half-open native-leaf range `[lo, hi)` a coarse leaf covers.
    #[must_use]
    pub const fn leaf_range(self, coarse: usize) -> (usize, usize) {
        (coarse << self.0, (coarse + 1) << self.0)
    }

    /// Number of coarse leaves over a `padded_count`-leaf tree.
    ///
    /// # Errors
    ///
    /// [`MerkleError::MeshLevelTooCoarse`] if the level sits above the root.
    pub fn coarse_row_len(self, padded_count: usize) -> Result<usize, MerkleError> {
        let height = padded_count.trailing_zeros();
        if self.0 > height {
            return Err(MerkleError::MeshLevelTooCoarse {
                level: self.0,
                height,
            });
        }
        Ok(padded_count >> self.0)
    }

    /// Level-order index, in the *full* tree, of coarse leaf `coarse`.
    ///
    /// The coarse row of a `padded_count`-leaf tree is indexed exactly as the
    /// leaf row of a `padded_count >> level`-leaf tree, which is what lets the
    /// witness machinery be reused unchanged.
    ///
    /// # Errors
    ///
    /// As [`MeshLevel::coarse_row_len`].
    pub fn coarse_node_index(
        self,
        padded_count: usize,
        coarse: usize,
    ) -> Result<usize, MerkleError> {
        Ok(self.coarse_row_len(padded_count)? - 1 + coarse)
    }
}

/// A subset proof addressed at the overlay mesh's coarse granularity.
///
/// Reduces to a `subset_proof::SubsetProof` field for field at
/// [`MeshLevel::NATIVE`]: same covered set, same witnesses, same certificate
/// preimage modulo the bound level. That equivalence is what makes the sweep's
/// `level = 0` row a true baseline rather than a separate code path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshProof {
    /// Sorted, deduplicated coarse-leaf indices covering the requested region.
    pub coarse_indices: Vec<usize>,
    /// Coarse node hashes, in the same order as `coarse_indices`.
    pub coarse_hashes: Vec<[u8; HASH_SIZE]>,
    /// Witness siblings *above* the coarse row, keyed by full-tree level-order
    /// index (root = 0, children of `i` = `2i+1`, `2i+2`).
    pub proof_nodes: BTreeMap<u64, [u8; HASH_SIZE]>,
    /// Native leaf hashes for members of partially covered coarse groups that
    /// the request does not deliver, keyed by native leaf index. Unallocated
    /// and padding members are omitted: the verifier reconstructs their null
    /// sentinel itself.
    pub fill_in: BTreeMap<u64, [u8; HASH_SIZE]>,
    /// Coverage certificate component (b): the trusted chunk-grid parameters.
    pub grid_params: ChunkGridParams,
    /// The coarsening level this proof is addressed at. Bound into
    /// `coverage_cert`, and never read by the verifier in place of its own
    /// trusted level.
    pub level: MeshLevel,
    /// `H(coarse_indices || level || grid_hash)`.
    pub coverage_cert: [u8; HASH_SIZE],
}

impl MeshProof {
    /// Serialized wire size in bytes, under the same accounting
    /// `subset_proof_geometry_bench` uses for a native `SubsetProof`, plus the
    /// 4-byte bound level and the fill-in map.
    #[must_use]
    pub fn wire_bytes(&self) -> usize {
        let c = self.coarse_indices.len();
        let d = self.grid_params.dims.len();
        c * INDEX_BYTES
            + c * HASH_SIZE
            + self.proof_nodes.len() * (INDEX_BYTES + HASH_SIZE)
            + self.fill_in.len() * (INDEX_BYTES + HASH_SIZE)
            + d * INDEX_BYTES * 2
            + HASH_SIZE // grid_params
            + HASH_SIZE // coverage_cert
            + 4 // level
    }

    /// Number of authenticated units the verifier compares, i.e. the
    /// verifier-side comparison count the coarsening is meant to reduce.
    #[must_use]
    pub fn compare_count(&self) -> usize {
        self.coarse_indices.len()
    }
}

/// Outcome of a coarse verification pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshVerdict {
    /// Every coarse leaf chained to the signed root. The region is intact at
    /// native granularity too: a coarse node commits to all `k` of its leaves.
    Verified,
    /// These coarse leaves did not verify. Each indicts `2^level` native
    /// chunks; [`localize_in_group`] narrows it to the exact ones.
    ///
    /// Under [`ProofConstruction::CanonicalPruned`] the covered subtree is
    /// rebuilt bottom-up and checked against the root *once*, so a failure
    /// carries no per-unit information and every covered coarse leaf is
    /// listed. Pruning and localization genuinely trade off; the caller
    /// chooses which it wants.
    Suspect(Vec<usize>),
}

impl MeshVerdict {
    /// Whether the region verified.
    #[must_use]
    pub fn is_verified(&self) -> bool {
        matches!(self, MeshVerdict::Verified)
    }
}

/// `H(coarse_indices || level || grid_hash)` -- the mesh's coverage
/// certificate. Binding `level` is defence in depth: the verifier already
/// recomputes the expected coarse set from its own trusted level, so a
/// cross-level replay is caught as a selection mismatch first.
fn mesh_coverage_cert(
    sorted_coarse: &[usize],
    level: MeshLevel,
    grid_hash: &[u8; HASH_SIZE],
    alg: HashAlg,
) -> [u8; HASH_SIZE] {
    let mut buf = Vec::with_capacity(sorted_coarse.len() * 8 + 4 + HASH_SIZE);
    for &idx in sorted_coarse {
        buf.extend_from_slice(&(idx as u64).to_le_bytes());
    }
    buf.extend_from_slice(&level.level().to_le_bytes());
    buf.extend_from_slice(grid_hash);
    alg.hash_leaf(&buf)
}

/// The sorted, deduplicated coarse-leaf set covering `fine_indices`.
#[must_use]
pub fn coarse_cover(fine_indices: &[usize], level: MeshLevel) -> Vec<usize> {
    let mut c: Vec<usize> = fine_indices
        .iter()
        .map(|&l| level.coarse_index(l))
        .collect();
    c.sort_unstable();
    c.dedup();
    c
}

/// Extract a coarse-granularity proof for `sel` from `tree`.
///
/// `order` must match the linearization `tree`'s leaves were built with, and
/// `level` is the coarsening the *verifier* will also be told to use.
///
/// # Errors
///
/// - [`MerkleError::MeshLevelTooCoarse`] if `level` sits above `tree`'s root.
/// - [`MerkleError::HyperslabOutOfBounds`] if `sel`'s rank does not match
///   `grid.dims`, or a covered index falls outside the tree.
/// - [`MerkleError::TreeTooDeep`] for a structurally invalid or oversized grid.
pub fn extract_subset_mesh(
    tree: &MerkleTree,
    grid: &ChunkGridParams,
    sel: &Selection,
    order: LeafOrder,
    construction: ProofConstruction,
    level: MeshLevel,
) -> Result<MeshProof, MerkleError> {
    let fine_indices = compute_expected_chunk_indices(grid, sel, order)?;
    let padded_count = tree.padded_leaf_count();
    let coarse_row = level.coarse_row_len(padded_count)?;
    let internal_nodes = padded_count - 1;
    let nodes = tree.nodes();
    let null = tree.algorithm().null_sentinel();

    let coarse_indices = coarse_cover(&fine_indices, level);

    let mut coarse_hashes = Vec::with_capacity(coarse_indices.len());
    for &c in &coarse_indices {
        let node = level.coarse_node_index(padded_count, c)?;
        coarse_hashes.push(
            *nodes
                .get(node)
                .ok_or(MerkleError::HyperslabOutOfBounds { idx: c })?,
        );
    }

    // Fill-in: members of a covered group the request does not deliver.
    // `fine_indices` is sorted, so membership is a binary search.
    let mut fill_in: BTreeMap<u64, [u8; HASH_SIZE]> = BTreeMap::new();
    if level != MeshLevel::NATIVE {
        for &c in &coarse_indices {
            let (lo, hi) = level.leaf_range(c);
            for leaf in lo..hi.min(padded_count) {
                if fine_indices.binary_search(&leaf).is_ok() {
                    continue;
                }
                let h = *nodes
                    .get(internal_nodes + leaf)
                    .ok_or(MerkleError::HyperslabOutOfBounds { idx: leaf })?;
                // Unallocated and padding members hash to a public constant;
                // sending them would be pure waste.
                if constant_time_eq(&h, &null) {
                    continue;
                }
                fill_in.insert(leaf as u64, h);
            }
        }
    }

    // Witnesses above the coarse row. The coarse row is indexed exactly as the
    // leaf row of a `coarse_row`-leaf tree, so both constructions apply as-is.
    let mut proof_nodes: BTreeMap<u64, [u8; HASH_SIZE]> = BTreeMap::new();
    match construction {
        ProofConstruction::NaiveDedup => {
            for &c in &coarse_indices {
                let mut node_idx = level.coarse_node_index(padded_count, c)?;
                while node_idx > 0 {
                    let sibling_idx = if node_idx % 2 == 1 {
                        node_idx + 1
                    } else {
                        node_idx - 1
                    };
                    let sibling_hash = *nodes
                        .get(sibling_idx)
                        .ok_or(MerkleError::HyperslabOutOfBounds { idx: c })?;
                    proof_nodes.entry(sibling_idx as u64).or_insert(sibling_hash);
                    node_idx = (node_idx - 1) / 2;
                }
            }
        }
        ProofConstruction::CanonicalPruned => {
            for sibling_idx in pruned_sibling_indices(coarse_row, &coarse_indices) {
                let sibling_hash = *nodes
                    .get(sibling_idx)
                    .ok_or(MerkleError::HyperslabOutOfBounds { idx: sibling_idx })?;
                proof_nodes.insert(sibling_idx as u64, sibling_hash);
            }
        }
    }

    let cert = mesh_coverage_cert(&coarse_indices, level, &grid.grid_hash, tree.algorithm());

    Ok(MeshProof {
        coarse_indices,
        coarse_hashes,
        proof_nodes,
        fill_in,
        grid_params: grid.clone(),
        level,
        coverage_cert: cert,
    })
}

/// Fold a coarse group's `2^level` native leaf hashes into its coarse node
/// hash, resolving each member from the delivered data, the proof's fill-in,
/// or the null sentinel in that order.
fn fold_group(
    alg: HashAlg,
    level: MeshLevel,
    coarse: usize,
    padded_count: usize,
    resolve: &dyn Fn(usize) -> [u8; HASH_SIZE],
) -> [u8; HASH_SIZE] {
    let (lo, hi) = level.leaf_range(coarse);
    let mut row: Vec<[u8; HASH_SIZE]> = (lo..hi)
        .map(|leaf| {
            if leaf < padded_count {
                resolve(leaf)
            } else {
                alg.null_sentinel()
            }
        })
        .collect();
    while row.len() > 1 {
        row = row.chunks(2).map(|p| alg.hash_pair(&p[0], &p[1])).collect();
    }
    row[0]
}

/// Verify that `chunks` is a complete, correct, untampered subset of the
/// dataset rooted at `root`, checked at the overlay mesh's coarse granularity.
///
/// # Trust model
///
/// Identical to `subset_proof::verify_subset`, with `level` joining `sel`,
/// `order` and `construction` as *caller* parameters: the verifier decides the
/// granularity it will accept, never reading it off the untrusted proof.
/// `chunks` is still the **native** covering chunk set -- coarsening changes
/// the metadata, not the payload -- and is still checked index-for-index
/// against the set recomputed from the verifier's own selection, so an omitted
/// chunk is caught at native granularity before any coarse fold happens.
///
/// # Errors
///
/// - [`MerkleError::GridHashMismatch`] if `expected_grid` does not hash to
///   `trusted_grid_hash`.
/// - [`MerkleError::CompanionTampered`] if `chunks` does not line up with the
///   proof, if the certificate does not recompute, or if a witness the walk
///   needs is absent.
/// - [`MerkleError::SelectionMismatch`] if the proof's coarse set is not the
///   set implied by the verifier's own selection at its own level.
/// - [`MerkleError::MeshLevelTooCoarse`] if `level` sits above the root.
#[allow(clippy::too_many_arguments)]
pub fn verify_subset_mesh(
    root: &[u8; HASH_SIZE],
    alg: HashAlg,
    chunks: &[ChunkData<'_>],
    proof: &MeshProof,
    expected_grid: &ChunkGridParams,
    trusted_grid_hash: &[u8; HASH_SIZE],
    sel: &Selection,
    order: LeafOrder,
    construction: ProofConstruction,
    level: MeshLevel,
) -> Result<MeshVerdict, MerkleError> {
    // Authenticate the grid parameters before trusting anything derived from
    // them -- same gate, same reasoning, as the native verifier.
    let recomputed_grid_hash = compute_grid_hash(
        &expected_grid.dims,
        &expected_grid.chunk_shape,
        expected_grid.elem_size,
        expected_grid.layout_class,
        alg,
    );
    if !constant_time_eq(&recomputed_grid_hash, trusted_grid_hash) {
        return Err(MerkleError::GridHashMismatch);
    }

    let padded_count = checked_padded_leaf_count(expected_grid)?;
    let coarse_row = level.coarse_row_len(padded_count)?;

    // Completeness at native granularity: the payload is unchanged by
    // coarsening, so this check is unchanged by it either.
    let expected_fine = compute_expected_chunk_indices(expected_grid, sel, order)?;
    if chunks.len() != expected_fine.len() {
        return Err(MerkleError::CompanionTampered);
    }
    for (chunk, &expected_idx) in chunks.iter().zip(expected_fine.iter()) {
        if chunk.index != expected_idx {
            return Err(MerkleError::CompanionTampered);
        }
    }

    // Bind the proof to the region *and granularity* the verifier asked for.
    let expected_coarse = coarse_cover(&expected_fine, level);
    if proof.coarse_indices != expected_coarse {
        return Err(MerkleError::SelectionMismatch);
    }
    if proof.coarse_hashes.len() != proof.coarse_indices.len() {
        return Err(MerkleError::CompanionTampered);
    }

    let cert = mesh_coverage_cert(&expected_coarse, level, &proof.grid_params.grid_hash, alg);
    if !constant_time_eq(&cert, &proof.coverage_cert) {
        return Err(MerkleError::CompanionTampered);
    }

    // Recompute every delivered chunk's native leaf hash from its bytes.
    let mut delivered: BTreeMap<usize, [u8; HASH_SIZE]> = BTreeMap::new();
    for chunk in chunks {
        delivered.insert(chunk.index, alg.hash_leaf(chunk.data));
    }
    let null = alg.null_sentinel();
    let resolve = |leaf: usize| -> [u8; HASH_SIZE] {
        delivered
            .get(&leaf)
            .copied()
            .or_else(|| proof.fill_in.get(&(leaf as u64)).copied())
            .unwrap_or(null)
    };

    // Fold each group and compare against the coarse hash the proof carries.
    // The comparison is what localizes; the fold is what authenticates.
    let mut suspect: Vec<usize> = Vec::new();
    let mut computed: Vec<[u8; HASH_SIZE]> = Vec::with_capacity(expected_coarse.len());
    for (i, &c) in expected_coarse.iter().enumerate() {
        let folded = fold_group(alg, level, c, padded_count, &resolve);
        if !constant_time_eq(&folded, &proof.coarse_hashes[i]) {
            suspect.push(c);
        }
        computed.push(folded);
    }

    if construction == ProofConstruction::CanonicalPruned {
        // One bottom-up rebuild of the covered subtree, one root comparison.
        // Cheapest on the wire, and blind about which unit failed.
        let mut known: BTreeMap<usize, [u8; HASH_SIZE]> = expected_coarse
            .iter()
            .zip(computed.iter())
            .map(|(&c, &h)| (coarse_row - 1 + c, h))
            .collect();

        let max_levels = usize::BITS as usize;
        let mut levels = 0usize;
        while !(known.len() == 1 && known.contains_key(&0)) {
            if known.is_empty() || levels >= max_levels {
                return Err(MerkleError::CompanionTampered);
            }
            levels += 1;
            let mut next: BTreeMap<usize, [u8; HASH_SIZE]> = BTreeMap::new();
            for (&node, &hash) in &known {
                if node == 0 {
                    return Err(MerkleError::CompanionTampered);
                }
                let parent = (node - 1) / 2;
                if next.contains_key(&parent) {
                    continue;
                }
                let sibling = if node % 2 == 1 { node + 1 } else { node - 1 };
                let sibling_hash = known
                    .get(&sibling)
                    .copied()
                    .or_else(|| proof.proof_nodes.get(&(sibling as u64)).copied())
                    .ok_or(MerkleError::CompanionTampered)?;
                let combined = if node % 2 == 1 {
                    alg.hash_pair(&hash, &sibling_hash)
                } else {
                    alg.hash_pair(&sibling_hash, &hash)
                };
                next.insert(parent, combined);
            }
            known = next;
        }
        let computed_root = known.get(&0).ok_or(MerkleError::CompanionTampered)?;
        if !constant_time_eq(computed_root, root) {
            return Ok(MeshVerdict::Suspect(expected_coarse));
        }
        return Ok(MeshVerdict::Verified);
    }

    // Naive: walk each coarse leaf to the root independently, which is what
    // buys per-unit localization in a single pass.
    for (i, &c) in expected_coarse.iter().enumerate() {
        let mut node_idx = coarse_row - 1 + c;
        let mut current = computed[i];
        let mut pos = c;
        while node_idx > 0 {
            let sibling_idx = if node_idx % 2 == 1 {
                node_idx + 1
            } else {
                node_idx - 1
            };
            let sibling = proof
                .proof_nodes
                .get(&(sibling_idx as u64))
                .copied()
                .ok_or(MerkleError::CompanionTampered)?;
            current = if pos % 2 == 0 {
                alg.hash_pair(&current, &sibling)
            } else {
                alg.hash_pair(&sibling, &current)
            };
            node_idx = (node_idx - 1) / 2;
            pos /= 2;
        }
        if !constant_time_eq(&current, root) && !suspect.contains(&c) {
            suspect.push(c);
        }
    }

    if suspect.is_empty() {
        Ok(MeshVerdict::Verified)
    } else {
        suspect.sort_unstable();
        suspect.dedup();
        Ok(MeshVerdict::Suspect(suspect))
    }
}

/// What a drill-down into one suspect coarse group cost and found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Localization {
    /// Native leaf indices whose recomputed hash disagrees with the companion.
    pub bad_leaves: Vec<usize>,
    /// Rounds of interaction: one per level descended, branches batched.
    pub rounds: usize,
    /// Node hashes the descent had to fetch -- 2 per node visited.
    pub wire_hashes: usize,
}

impl Localization {
    /// Incremental wire cost of the drill-down, in bytes, under the same
    /// accounting as [`MeshProof::wire_bytes`].
    #[must_use]
    pub fn wire_bytes(&self) -> usize {
        self.wire_hashes * (INDEX_BYTES + HASH_SIZE)
    }
}

/// Descend into one suspect coarse group to name the exact bad native chunks.
///
/// `recomputed` holds the verifier's own leaf hashes, computed from the chunk
/// bytes it holds; any member absent from it is treated as the null sentinel,
/// which is correct for unallocated and padding members. `tree` stands in for
/// the companion the prover serves node hashes from.
///
/// Every branch whose recomputed subtree value disagrees with the companion is
/// followed, so the result is the complete set of bad leaves in the group, at
/// `2` hashes per visited node: `O(log k)` for one corruption, `O(b log k)`
/// for `b`. This is the `O(log N)` worst case of the tamper-localization
/// claim, unchanged -- coarsening moved where the work happens, not how much
/// of it there is.
///
/// # Errors
///
/// [`MerkleError::MeshLevelTooCoarse`] if `level` sits above `tree`'s root;
/// [`MerkleError::HyperslabOutOfBounds`] if `coarse` is outside the coarse row.
pub fn localize_in_group(
    tree: &MerkleTree,
    recomputed: &BTreeMap<usize, [u8; HASH_SIZE]>,
    level: MeshLevel,
    coarse: usize,
) -> Result<Localization, MerkleError> {
    let padded_count = tree.padded_leaf_count();
    let coarse_row = level.coarse_row_len(padded_count)?;
    if coarse >= coarse_row {
        return Err(MerkleError::HyperslabOutOfBounds { idx: coarse });
    }
    let alg = tree.algorithm();
    let null = alg.null_sentinel();
    let internal_nodes = padded_count - 1;
    let nodes = tree.nodes();

    // The verifier's value for the subtree rooted at `node`, `height` rows
    // above the native leaf row.
    fn verifier_value(
        alg: HashAlg,
        node: usize,
        height: u32,
        internal_nodes: usize,
        recomputed: &BTreeMap<usize, [u8; HASH_SIZE]>,
        null: [u8; HASH_SIZE],
    ) -> [u8; HASH_SIZE] {
        if height == 0 {
            let leaf = node - internal_nodes;
            return recomputed.get(&leaf).copied().unwrap_or(null);
        }
        let l = verifier_value(alg, 2 * node + 1, height - 1, internal_nodes, recomputed, null);
        let r = verifier_value(alg, 2 * node + 2, height - 1, internal_nodes, recomputed, null);
        alg.hash_pair(&l, &r)
    }

    let mut out = Localization::default();
    let mut frontier = vec![(level.coarse_node_index(padded_count, coarse)?, level.level())];

    while let Some(&(_, h)) = frontier.first() {
        if h == 0 {
            for (node, _) in frontier.drain(..) {
                out.bad_leaves.push(node - internal_nodes);
            }
            break;
        }
        out.rounds += 1;
        let mut next = Vec::new();
        for (node, height) in frontier.drain(..) {
            for child in [2 * node + 1, 2 * node + 2] {
                out.wire_hashes += 1;
                let stored = *nodes
                    .get(child)
                    .ok_or(MerkleError::HyperslabOutOfBounds { idx: child })?;
                let mine = verifier_value(
                    alg,
                    child,
                    height - 1,
                    internal_nodes,
                    recomputed,
                    null,
                );
                if !constant_time_eq(&mine, &stored) {
                    next.push((child, height - 1));
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    out.bad_leaves.sort_unstable();
    out.bad_leaves.dedup();
    Ok(out)
}

/// Number of companion nodes a single native chunk write must rewrite under
/// `level`.
///
/// The mesh stores nothing of its own, so this is `log2(padded_count) + 1` for
/// every level: the leaf plus its ancestor path, exactly as at
/// [`MeshLevel::NATIVE`]. Coarsening cannot amplify an incremental write
/// because there is no coarse row to maintain separately -- the coarse node
/// *is* an ancestor already on that path.
#[must_use]
pub fn write_path_len(padded_count: usize, _level: MeshLevel) -> usize {
    padded_count.trailing_zeros() as usize + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subset_proof::{
        SubsetProof, extract_subset_with, leaf_index_for_coord, verify_subset_with,
    };
    use crate::verification_grid::LayoutClass;

    const ALG: HashAlg = HashAlg::Blake3;

    /// A `side^rank` grid of single-element chunks under `order`, with the
    /// payload placed at the leaf position `order` maps each coordinate to.
    fn build(side: u64, rank: usize, order: LeafOrder) -> (MerkleTree, ChunkGridParams, Vec<Vec<u8>>) {
        let grid = ChunkGridParams::new(
            vec![side; rank],
            vec![1u64; rank],
            4,
            LayoutClass::Chunked,
            ALG,
        );
        let n_per_dim = grid.n_chunks_per_dim();
        let total = grid.total_chunk_count() as usize;
        let mut payload = vec![Vec::new(); total];
        for f in 0..total {
            let mut rem = f as u64;
            let mut coord = vec![0u64; rank];
            for d in (0..rank).rev() {
                coord[d] = rem % side;
                rem /= side;
            }
            let leaf = leaf_index_for_coord(&coord, &n_per_dim, order);
            payload[leaf as usize] = format!("chunk-{f}").into_bytes();
        }
        let refs: Vec<&[u8]> = payload.iter().map(Vec::as_slice).collect();
        let tree = MerkleTree::from_chunks(&refs, ALG);
        (tree, grid, payload)
    }

    fn deliver<'a>(indices: &[usize], payload: &'a [Vec<u8>]) -> Vec<ChunkData<'a>> {
        indices
            .iter()
            .map(|&i| ChunkData {
                index: i,
                data: &payload[i],
            })
            .collect()
    }

    fn block(rank: usize, origin: u64, m: u64) -> Selection {
        Selection::slice(&(0..rank).map(|_| origin..origin + m).collect::<Vec<_>>())
    }

    fn check(
        tree: &MerkleTree,
        grid: &ChunkGridParams,
        payload: &[Vec<u8>],
        sel: &Selection,
        order: LeafOrder,
        construction: ProofConstruction,
        level: MeshLevel,
    ) -> (MeshProof, MeshVerdict) {
        let proof = extract_subset_mesh(tree, grid, sel, order, construction, level).unwrap();
        let fine = compute_expected_chunk_indices(grid, sel, order).unwrap();
        let chunks = deliver(&fine, payload);
        let verdict = verify_subset_mesh(
            tree.root(),
            ALG,
            &chunks,
            &proof,
            grid,
            &grid.grid_hash,
            sel,
            order,
            construction,
            level,
        )
        .unwrap();
        (proof, verdict)
    }

    /// The whole point: a coarse proof over a real selection still chains to
    /// the same signed root, at every level, ordering and construction.
    #[test]
    fn coarse_proofs_verify_at_every_level() {
        for order in [LeafOrder::RowMajor, LeafOrder::Morton, LeafOrder::Hilbert] {
            let (tree, grid, payload) = build(16, 2, order);
            for construction in [ProofConstruction::NaiveDedup, ProofConstruction::CanonicalPruned]
            {
                for l in 0..=8u32 {
                    let level = MeshLevel::new(l).unwrap();
                    let sel = block(2, 4, 8);
                    let (_, verdict) =
                        check(&tree, &grid, &payload, &sel, order, construction, level);
                    assert!(
                        verdict.is_verified(),
                        "order {order:?} construction {construction:?} level {l} failed"
                    );
                }
            }
        }
    }

    /// `MeshLevel::NATIVE` must be the existing per-chunk proof, not a
    /// parallel implementation of it -- otherwise the sweep's level-0 row is
    /// not a baseline.
    #[test]
    fn native_level_reproduces_the_existing_subset_proof() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let sel = block(2, 5, 6);
        for construction in [ProofConstruction::NaiveDedup, ProofConstruction::CanonicalPruned] {
            let mesh =
                extract_subset_mesh(&tree, &grid, &sel, order, construction, MeshLevel::NATIVE)
                    .unwrap();
            let native: SubsetProof =
                extract_subset_with(&tree, &grid, &sel, order, construction).unwrap();
            assert_eq!(mesh.coarse_indices, native.chunk_indices);
            assert_eq!(mesh.coarse_hashes, native.leaf_hashes);
            assert_eq!(mesh.proof_nodes, native.proof_nodes);
            assert!(mesh.fill_in.is_empty());
            // Same wire content, so the size accounting differs only by the
            // 4 bytes the bound level costs.
            let k = native.chunk_indices.len();
            let d = grid.dims.len();
            let native_bytes = k * 8
                + k * HASH_SIZE
                + native.proof_nodes.len() * (8 + HASH_SIZE)
                + d * 16
                + HASH_SIZE
                + HASH_SIZE;
            assert_eq!(mesh.wire_bytes(), native_bytes + 4);
            let _ = &payload;
        }
    }

    /// A coarse leaf is an ancestor of the native tree, so the coarse hash the
    /// prover sends must equal the node already sitting there.
    #[test]
    fn coarse_hash_is_the_ancestor_node_already_in_the_tree() {
        let order = LeafOrder::Morton;
        let (tree, grid, _payload) = build(16, 2, order);
        let padded = tree.padded_leaf_count();
        for l in 1..=6u32 {
            let level = MeshLevel::new(l).unwrap();
            let sel = block(2, 0, 16);
            let proof =
                extract_subset_mesh(&tree, &grid, &sel, order, ProofConstruction::NaiveDedup, level)
                    .unwrap();
            for (i, &c) in proof.coarse_indices.iter().enumerate() {
                let node = level.coarse_node_index(padded, c).unwrap();
                assert_eq!(&proof.coarse_hashes[i], &tree.nodes()[node]);
            }
        }
    }

    /// Coarsening must shrink the wire on a whole-dataset check, monotonically
    /// in `k`. This is the claimed payoff, stated as a test rather than a hope.
    #[test]
    fn wire_shrinks_monotonically_with_group_size() {
        let order = LeafOrder::Morton;
        let (tree, grid, _payload) = build(32, 2, order);
        let sel = Selection::All;
        let mut prev = usize::MAX;
        for l in 0..=10u32 {
            let level = MeshLevel::new(l).unwrap();
            let proof = extract_subset_mesh(
                &tree,
                &grid,
                &sel,
                order,
                ProofConstruction::CanonicalPruned,
                level,
            )
            .unwrap();
            let bytes = proof.wire_bytes();
            assert!(bytes < prev, "level {l}: {bytes} not below {prev}");
            prev = bytes;
        }
    }

    /// A fully covered group needs no fill-in; a ragged one does, and the cost
    /// is bounded by the group size. This is the boundary tax the design has
    /// to pay for, and the reason alignment matters here as it does in RQ6.
    #[test]
    fn fill_in_appears_only_on_partially_covered_groups() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let level = MeshLevel::new(4).unwrap(); // k = 16

        // A quadrant-aligned 4x4 block is exactly one Morton-aligned range of
        // 16 leaves: one whole group, nothing to fill in.
        let aligned = block(2, 4, 4);
        let (p_aligned, v) = check(
            &tree,
            &grid,
            &payload,
            &aligned,
            order,
            ProofConstruction::NaiveDedup,
            level,
        );
        assert!(v.is_verified());
        assert!(
            p_aligned.fill_in.is_empty(),
            "aligned block should need no fill-in, got {}",
            p_aligned.fill_in.len()
        );

        // Slide it off the quadrant grid and the same 16 chunks now straddle
        // several groups, every one of them partial.
        let offset = block(2, 5, 4);
        let (p_offset, v) = check(
            &tree,
            &grid,
            &payload,
            &offset,
            order,
            ProofConstruction::NaiveDedup,
            level,
        );
        assert!(v.is_verified());
        assert!(!p_offset.fill_in.is_empty());
        let groups = p_offset.coarse_indices.len();
        assert!(
            p_offset.fill_in.len() <= groups * level.group_size() as usize,
            "fill-in must stay bounded by the covered groups"
        );
    }

    /// Unallocated (and padding) members hash to a public constant, so they
    /// must never be put on the wire -- the answer to the sparse-chunk
    /// question the design left open.
    #[test]
    fn unallocated_group_members_cost_no_wire_bytes() {
        // 6 real chunks pad to 8 leaves; two padding slots carry the null
        // sentinel. Two more leaves are explicitly written as unallocated.
        let mut leaves: Vec<[u8; HASH_SIZE]> = (0..6)
            .map(|i| ALG.hash_leaf(format!("c{i}").as_bytes()))
            .collect();
        leaves[2] = ALG.null_sentinel();
        leaves[3] = ALG.null_sentinel();
        let tree = MerkleTree::from_leaf_hashes(&leaves, ALG);
        let grid = ChunkGridParams::new(vec![6], vec![1], 4, LayoutClass::Chunked, ALG);
        let level = MeshLevel::new(3).unwrap(); // one group covers all 8 slots

        let sel = Selection::slice(&[0..2]);
        let proof = extract_subset_mesh(
            &tree,
            &grid,
            &sel,
            LeafOrder::RowMajor,
            ProofConstruction::NaiveDedup,
            level,
        )
        .unwrap();

        // Leaves 4 and 5 are real and undelivered, so they must be filled in;
        // 2, 3 (unallocated) and 6, 7 (padding) must not be.
        assert_eq!(
            proof.fill_in.keys().copied().collect::<Vec<_>>(),
            vec![4u64, 5u64],
            "only allocated, undelivered members belong on the wire"
        );

        let payload: Vec<Vec<u8>> = (0..6).map(|i| format!("c{i}").into_bytes()).collect();
        let chunks = deliver(&[0, 1], &payload);
        let verdict = verify_subset_mesh(
            tree.root(),
            ALG,
            &chunks,
            &proof,
            &grid,
            &grid.grid_hash,
            &sel,
            LeafOrder::RowMajor,
            ProofConstruction::NaiveDedup,
            level,
        )
        .unwrap();
        assert!(verdict.is_verified());
    }

    /// Dropping a *real* fill-in hash is not a hole in the scheme: the
    /// verifier substitutes the sentinel, the fold diverges, the root check
    /// fails.
    #[test]
    fn omitting_a_real_fill_in_hash_is_rejected() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let level = MeshLevel::new(4).unwrap();
        let sel = block(2, 5, 4);
        let mut proof =
            extract_subset_mesh(&tree, &grid, &sel, order, ProofConstruction::NaiveDedup, level)
                .unwrap();
        let victim = *proof.fill_in.keys().next().expect("offset block has fill-in");
        proof.fill_in.remove(&victim);

        let fine = compute_expected_chunk_indices(&grid, &sel, order).unwrap();
        let chunks = deliver(&fine, &payload);
        let verdict = verify_subset_mesh(
            tree.root(),
            ALG,
            &chunks,
            &proof,
            &grid,
            &grid.grid_hash,
            &sel,
            order,
            ProofConstruction::NaiveDedup,
            level,
        )
        .unwrap();
        assert!(!verdict.is_verified());
    }

    /// A tampered chunk must be caught, and under the naive construction the
    /// failure must name the group it is in.
    #[test]
    fn tampered_chunk_is_caught_and_the_group_named() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let level = MeshLevel::new(3).unwrap();
        let sel = block(2, 0, 8);
        let fine = compute_expected_chunk_indices(&grid, &sel, order).unwrap();
        let proof =
            extract_subset_mesh(&tree, &grid, &sel, order, ProofConstruction::NaiveDedup, level)
                .unwrap();

        let victim = fine[fine.len() / 2];
        let mut chunks = deliver(&fine, &payload);
        let evil = b"tampered".to_vec();
        for c in chunks.iter_mut() {
            if c.index == victim {
                c.data = &evil;
            }
        }
        let verdict = verify_subset_mesh(
            tree.root(),
            ALG,
            &chunks,
            &proof,
            &grid,
            &grid.grid_hash,
            &sel,
            order,
            ProofConstruction::NaiveDedup,
            level,
        )
        .unwrap();
        match verdict {
            MeshVerdict::Suspect(groups) => {
                assert_eq!(groups, vec![level.coarse_index(victim)]);
            }
            MeshVerdict::Verified => panic!("tampered chunk verified"),
        }
    }

    /// Cross-level replay: a proof built at one granularity must not verify at
    /// another, since the verifier derives the coarse set from its own level.
    #[test]
    fn cross_level_replay_is_rejected() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let sel = block(2, 0, 8);
        let proof = extract_subset_mesh(
            &tree,
            &grid,
            &sel,
            order,
            ProofConstruction::NaiveDedup,
            MeshLevel::new(2).unwrap(),
        )
        .unwrap();
        let fine = compute_expected_chunk_indices(&grid, &sel, order).unwrap();
        let chunks = deliver(&fine, &payload);
        let err = verify_subset_mesh(
            tree.root(),
            ALG,
            &chunks,
            &proof,
            &grid,
            &grid.grid_hash,
            &sel,
            order,
            ProofConstruction::NaiveDedup,
            MeshLevel::new(3).unwrap(),
        )
        .unwrap_err();
        assert_eq!(err, MerkleError::SelectionMismatch);
    }

    /// Coarsening must not weaken completeness: a silently dropped chunk is
    /// still caught, at native granularity, before any fold happens.
    #[test]
    fn dropped_chunk_is_still_caught_at_native_granularity() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let level = MeshLevel::new(4).unwrap();
        let sel = block(2, 0, 8);
        let fine = compute_expected_chunk_indices(&grid, &sel, order).unwrap();
        let proof =
            extract_subset_mesh(&tree, &grid, &sel, order, ProofConstruction::NaiveDedup, level)
                .unwrap();
        let mut chunks = deliver(&fine, &payload);
        chunks.pop();
        let err = verify_subset_mesh(
            tree.root(),
            ALG,
            &chunks,
            &proof,
            &grid,
            &grid.grid_hash,
            &sel,
            order,
            ProofConstruction::NaiveDedup,
            level,
        )
        .unwrap_err();
        assert_eq!(err, MerkleError::CompanionTampered);
    }

    /// Drill-down: one corruption is localized in `level` rounds at two
    /// hashes per visited node -- the `O(log k)` descent the idea claims.
    #[test]
    fn drill_down_localizes_one_corruption_in_log_k() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let level = MeshLevel::new(6).unwrap(); // k = 64
        let victim = 100usize;
        let group = level.coarse_index(victim);

        let mut recomputed: BTreeMap<usize, [u8; HASH_SIZE]> = BTreeMap::new();
        let (lo, hi) = level.leaf_range(group);
        for leaf in lo..hi {
            let bytes: &[u8] = if leaf == victim {
                b"tampered"
            } else {
                &payload[leaf]
            };
            recomputed.insert(leaf, ALG.hash_leaf(bytes));
        }

        let loc = localize_in_group(&tree, &recomputed, level, group).unwrap();
        assert_eq!(loc.bad_leaves, vec![victim]);
        assert_eq!(loc.rounds, level.level() as usize);
        // Two hashes per level: the descent never widens for a single fault.
        assert_eq!(loc.wire_hashes, 2 * level.level() as usize);
        let _ = &grid;
    }

    /// Two corruptions cost `O(b log k)`, not `O(k)`: the descent follows both
    /// branches only where they actually diverge.
    #[test]
    fn drill_down_cost_scales_with_fault_count_not_group_size() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let level = MeshLevel::new(6).unwrap();
        let group = 1usize;
        let (lo, hi) = level.leaf_range(group);
        let victims = [lo + 3, lo + 40];

        let mut recomputed = BTreeMap::new();
        for leaf in lo..hi {
            let bytes: &[u8] = if victims.contains(&leaf) {
                b"tampered"
            } else {
                &payload[leaf]
            };
            recomputed.insert(leaf, ALG.hash_leaf(bytes));
        }
        let loc = localize_in_group(&tree, &recomputed, level, group).unwrap();
        assert_eq!(loc.bad_leaves, victims.to_vec());
        assert!(
            loc.wire_hashes < level.group_size() as usize,
            "descent cost {} should stay below the group size {}",
            loc.wire_hashes,
            level.group_size()
        );
        let _ = &grid;
    }

    /// The write-amplification question: an incremental chunk write touches
    /// the same `O(log N)` ancestor path at every level, because the coarse
    /// node is already on it. Checked against the tree, not asserted.
    #[test]
    fn incremental_write_touches_one_path_at_every_level() {
        let order = LeafOrder::Morton;
        let (tree, _grid, payload) = build(16, 2, order);
        let padded = tree.padded_leaf_count();
        let before = tree.nodes().to_vec();

        let victim = 77usize;
        let mut updated = tree.clone();
        updated
            .update_leaf(victim, ALG.hash_leaf(b"new contents"))
            .unwrap();

        let changed: Vec<usize> = before
            .iter()
            .zip(updated.nodes().iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(changed.len(), write_path_len(padded, MeshLevel::NATIVE));

        // At every level, exactly one coarse node moved, and it is on that
        // same path -- so there is no coarse row to rewrite separately.
        for l in 0..=padded.trailing_zeros() {
            let level = MeshLevel::new(l).unwrap();
            let node = level
                .coarse_node_index(padded, level.coarse_index(victim))
                .unwrap();
            assert!(changed.contains(&node), "level {l} coarse node not on path");
            assert_eq!(write_path_len(padded, level), changed.len());
        }
        let _ = payload;
    }

    /// A level above the root is caller misuse and must surface as a typed
    /// error, not a panic or a silently degenerate proof.
    #[test]
    fn level_above_the_root_is_rejected() {
        let order = LeafOrder::RowMajor;
        let (tree, grid, _) = build(16, 2, order);
        let padded = tree.padded_leaf_count();
        let too_far = MeshLevel::new(padded.trailing_zeros() + 1).unwrap();
        let err = extract_subset_mesh(
            &tree,
            &grid,
            &Selection::All,
            order,
            ProofConstruction::NaiveDedup,
            too_far,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::MeshLevelTooCoarse { .. }));
        assert!(MeshLevel::from_group_size(3).is_err());
        assert_eq!(MeshLevel::from_group_size(64).unwrap().level(), 6);
    }

    /// The whole tree as one coarse leaf is the degenerate end of the mesh and
    /// must still be a correct (if uninformative) proof.
    #[test]
    fn root_level_mesh_is_a_whole_dataset_hash() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let padded = tree.padded_leaf_count();
        let level = MeshLevel::new(padded.trailing_zeros()).unwrap();
        let (proof, verdict) = check(
            &tree,
            &grid,
            &payload,
            &Selection::All,
            order,
            ProofConstruction::CanonicalPruned,
            level,
        );
        assert!(verdict.is_verified());
        assert_eq!(proof.coarse_indices, vec![0]);
        assert_eq!(&proof.coarse_hashes[0], tree.root());
        assert!(proof.proof_nodes.is_empty());
    }

    /// A mesh proof must not be verifiable by the native verifier, or the
    /// granularity would be negotiable by the prover.
    #[test]
    fn a_native_verifier_cannot_be_fed_a_coarse_proof() {
        let order = LeafOrder::Morton;
        let (tree, grid, payload) = build(16, 2, order);
        let sel = block(2, 0, 8);
        let level = MeshLevel::new(2).unwrap();
        let mesh =
            extract_subset_mesh(&tree, &grid, &sel, order, ProofConstruction::NaiveDedup, level)
                .unwrap();
        // Re-present the coarse witnesses as if they were a native proof.
        let native = SubsetProof {
            chunk_indices: mesh.coarse_indices.clone(),
            leaf_hashes: mesh.coarse_hashes.clone(),
            proof_nodes: mesh.proof_nodes.clone(),
            grid_params: mesh.grid_params.clone(),
            coverage_cert: mesh.coverage_cert,
        };
        let fine = compute_expected_chunk_indices(&grid, &sel, order).unwrap();
        let chunks = deliver(&fine, &payload);
        assert!(
            verify_subset_with(
                tree.root(),
                ALG,
                &chunks,
                &native,
                &grid,
                &grid.grid_hash,
                &sel,
                order,
                ProofConstruction::NaiveDedup,
            )
            .is_err()
        );
    }
}
