//! RQ6: subset-proof size as a two-factor study over *leaf geometry*.
//!
//! The existing `subset_proof_size_bench` sweeps k over a **1-D** dataset,
//! where every linearization coincides (Morton and row-major are the identity
//! on one axis), so it cannot say anything about leaf ordering. This benchmark
//! is the multidimensional case the leaf-geometry argument is actually about:
//!
//!   leaf ordering  (row-major | Morton | Hilbert)
//!     x  proof construction  (naive dedup | canonical pruning)
//!     x  hyperslab shape     (solid block | strided | sparse)
//!     x  rank                (2 | 3)
//!
//! The reported quantity is not the raw byte count but *where each combination
//! falls between the two bounds*: naive witness aggregation over a k-chunk
//! selection costs O(Volume . log N), while an ideal layout costs
//! O(SurfaceArea . log N) for an axis-aligned block. `position_frac` is 0.0 at
//! the surface-area bound and 1.0 at the volume bound.
//!
//! Grids are cubic with power-of-two extents so that all three orderings are
//! bijections onto `0..N`; that is the setting in which the comparison is
//! meaningful, and a non-power-of-two axis would leave Morton and Hilbert
//! indices sparse.
//!
//! Proof sizes are computed, never read, so this benchmark touches no storage
//! and carries no storage class. Timings follow the Statistical Protocol
//! (30 measured trials after 5 discarded warmups, median + 95% bootstrap CI).
//!
//! Output: `benches/results/subset-proof-geometry.csv`.

use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    ChunkData, ChunkGridParams, LeafOrder, ProofConstruction, SubsetProof, extract_subset_with,
    verify_subset_with,
};
use clawhdf5_format::verification_grid::LayoutClass;

const HASH_SIZE: usize = 32;
const WARMUP_TRIALS: usize = 5;
const TRIALS: usize = 30;
const BOOTSTRAP_ITERATIONS: usize = 2000;

struct Xorshift64(u64);

impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn bootstrap_median_ci(samples: &[f64], seed: u64) -> (f64, f64, f64) {
    let point = median(samples);
    if samples.len() < 2 {
        return (point, point, point);
    }
    let mut rng = Xorshift64(seed | 1);
    let mut medians = Vec::with_capacity(BOOTSTRAP_ITERATIONS);
    for _ in 0..BOOTSTRAP_ITERATIONS {
        let resample: Vec<f64> = (0..samples.len())
            .map(|_| samples[(rng.next_u64() % samples.len() as u64) as usize])
            .collect();
        medians.push(median(&resample));
    }
    medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = medians[(BOOTSTRAP_ITERATIONS as f64 * 0.025) as usize];
    let hi = medians[((BOOTSTRAP_ITERATIONS as f64 * 0.975) as usize).min(medians.len() - 1)];
    (point, lo, hi)
}

/// Serialized proof size, using the same accounting as `subset_proof_size_bench`
/// so the two are directly comparable.
fn proof_size_bytes(proof: &SubsetProof) -> usize {
    let k = proof.chunk_indices.len();
    let d = proof.grid_params.dims.len();
    k * 8                                        // chunk_indices
        + k * HASH_SIZE                          // leaf_hashes
        + proof.proof_nodes.len() * (8 + HASH_SIZE)
        + d * 8 + d * 8 + HASH_SIZE              // grid_params
        + HASH_SIZE // coverage_cert
}

/// Build the tree for a cubic `side^rank` grid of single-element chunks under
/// `order`, plus the payload indexed by *leaf position*.
fn build(side: u64, rank: usize, order: LeafOrder) -> (MerkleTree, ChunkGridParams, Vec<Vec<u8>>) {
    let dims = vec![side; rank];
    let grid = ChunkGridParams::new(
        dims,
        vec![1u64; rank],
        4,
        LayoutClass::Chunked,
        HashAlg::Blake3,
    );
    let n_per_dim = grid.n_chunks_per_dim();
    let total = grid.total_chunk_count() as usize;

    let mut payload = vec![Vec::new(); total];
    let mut coord = vec![0u64; rank];
    for f in 0..total {
        let mut rem = f as u64;
        for d in (0..rank).rev() {
            coord[d] = rem % side;
            rem /= side;
        }
        let leaf = clawhdf5_format::subset_proof::leaf_index_for_coord(&coord, &n_per_dim, order);
        payload[leaf as usize] = format!("c{f}").into_bytes();
    }
    let refs: Vec<&[u8]> = payload.iter().map(Vec::as_slice).collect();
    let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);
    (tree, grid, payload)
}

/// The three selection geometries, each returning `(selection, k, block_extent)`.
/// `block_extent` is `Some(m)` only for the solid block, where the
/// surface-area bound is defined.
fn selection(
    shape: &str,
    side: u64,
    rank: usize,
    m: u64,
    origin: u64,
) -> (Selection, usize, Option<u64>) {
    match shape {
        // Solid axis-aligned rectangle of side m: the case the surface-area
        // bound is stated for.
        "block" => {
            let ranges: Vec<std::ops::Range<u64>> =
                (0..rank).map(|_| origin..origin + m).collect();
            (Selection::slice(&ranges), m.pow(rank as u32) as usize, Some(m))
        }
        // Same chunk count, but every other cell along the fastest axis, so
        // the selection spans twice the extent with the same volume.
        "strided" => {
            let mut points = Vec::new();
            let mut coord = vec![0u64; rank];
            let total = m.pow(rank as u32);
            for f in 0..total {
                let mut rem = f;
                for d in (0..rank).rev() {
                    coord[d] = rem % m;
                    rem /= m;
                }
                let mut p: Vec<u64> = (0..rank).map(|d| origin + coord[d]).collect();
                p[rank - 1] = origin + coord[rank - 1] * 2;
                if p[rank - 1] < side {
                    points.push(p);
                }
            }
            let k = points.len();
            (Selection::Points(points), k, None)
        }
        // k cells scattered over the whole grid: no locality to exploit.
        "sparse" => {
            let k = m.pow(rank as u32) as usize;
            let total = side.pow(rank as u32);
            let mut rng = Xorshift64(0x5EED_1234);
            let mut chosen: Vec<u64> = Vec::with_capacity(k);
            while chosen.len() < k {
                let f = rng.next_u64() % total;
                if !chosen.contains(&f) {
                    chosen.push(f);
                }
            }
            chosen.sort_unstable();
            let points: Vec<Vec<u64>> = chosen
                .iter()
                .map(|&f| {
                    let mut rem = f;
                    let mut c = vec![0u64; rank];
                    for d in (0..rank).rev() {
                        c[d] = rem % side;
                        rem /= side;
                    }
                    c
                })
                .collect();
            (Selection::Points(points), k, None)
        }
        other => panic!("unknown shape: {other}"),
    }
}

struct Row {
    rank: usize,
    side: u64,
    alignment: &'static str,
    n_total: usize,
    order: &'static str,
    construction: &'static str,
    shape: &'static str,
    k: usize,
    proof_nodes: usize,
    proof_bytes: usize,
    volume_nodes: usize,
    surface_nodes: Option<usize>,
    position_frac: Option<f64>,
    proof_ms: (f64, f64, f64),
    verify_ms: (f64, f64, f64),
}

#[allow(clippy::too_many_arguments)]
fn measure(
    tree: &MerkleTree,
    grid: &ChunkGridParams,
    payload: &[Vec<u8>],
    sel: &Selection,
    order: LeafOrder,
    construction: ProofConstruction,
    seed: u64,
) -> (SubsetProof, (f64, f64, f64), (f64, f64, f64)) {
    let proof = extract_subset_with(tree, grid, sel, order, construction).unwrap();
    let delivered: Vec<ChunkData<'_>> = proof
        .chunk_indices
        .iter()
        .map(|&idx| ChunkData {
            index: idx,
            data: &payload[idx],
        })
        .collect();

    // Correctness gate: a size measured from a proof that does not verify is
    // not a measurement of anything.
    assert!(
        verify_subset_with(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            grid,
            &grid.grid_hash,
            sel,
            order,
            construction,
        )
        .unwrap(),
        "proof failed to verify before timing"
    );

    let mut proof_times = Vec::with_capacity(TRIALS);
    let mut verify_times = Vec::with_capacity(TRIALS);
    for t in 0..(WARMUP_TRIALS + TRIALS) {
        let t0 = Instant::now();
        let p = extract_subset_with(tree, grid, sel, order, construction).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;

        let d: Vec<ChunkData<'_>> = p
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &payload[idx],
            })
            .collect();
        let t1 = Instant::now();
        let _ = verify_subset_with(
            tree.root(),
            HashAlg::Blake3,
            &d,
            &p,
            grid,
            &grid.grid_hash,
            sel,
            order,
            construction,
        )
        .unwrap();
        let vt = t1.elapsed().as_secs_f64() * 1000.0;

        if t >= WARMUP_TRIALS {
            proof_times.push(dt);
            verify_times.push(vt);
        }
    }

    (
        proof,
        bootstrap_median_ci(&proof_times, seed),
        bootstrap_median_ci(&verify_times, seed ^ 0xA5A5),
    )
}

fn main() {
    let orders: [(&str, LeafOrder); 3] = [
        ("row_major", LeafOrder::RowMajor),
        ("morton", LeafOrder::Morton),
        ("hilbert", LeafOrder::Hilbert),
    ];
    let constructions: [(&str, ProofConstruction); 2] = [
        ("naive_dedup", ProofConstruction::NaiveDedup),
        ("canonical_pruned", ProofConstruction::CanonicalPruned),
    ];
    let shapes = ["block", "strided", "sparse"];

    // (rank, side, block side lengths m)
    let configs: [(usize, u64, &[u64]); 2] = [(2, 256, &[4, 8, 16, 32]), (3, 32, &[2, 4, 8, 16])];

    // Alignment is the variable that decides whether leaf ordering can matter
    // at all. An aligned 2^k block is the *same* aligned range of leaf indices
    // under every quadrant-recursive curve, so Morton and Hilbert are
    // indistinguishable on it by construction and the proof collapses to a
    // single subtree root plus its path. Offsetting the block off the quadrant
    // grid is what separates the curves, and it is also the realistic case:
    // a scientific reader's region of interest is not aligned to a power of
    // two in chunk space. Both are reported.
    let alignments: [(&str, fn(u64, u64) -> u64); 2] = [
        ("aligned", |side, m| (side / 4 / m.max(1)) * m),
        ("offset", |side, m| (side / 4 / m.max(1)) * m + m / 2 + 1),
    ];

    let mut rows: Vec<Row> = Vec::new();
    let mut seed = 0xC0FF_EE01u64;

    for &(rank, side, ms) in &configs {
        let n_total = side.pow(rank as u32) as usize;
        let log_n = (n_total as f64).log2().ceil() as usize;
        for &(order_name, order) in &orders {
            let (tree, grid, payload) = build(side, rank, order);
            eprintln!("rank {rank} side {side} order {order_name}: tree built ({n_total} leaves)");
            for &m in ms {
              for &(align_name, origin_fn) in &alignments {
                let origin = origin_fn(side, m).min(side - m);
                for &shape in &shapes {
                    let (sel, k, block_m) = selection(shape, side, rank, m, origin);
                    for &(cname, construction) in &constructions {
                        let (proof, pt, vt) =
                            measure(&tree, &grid, &payload, &sel, order, construction, seed);
                        seed = seed.wrapping_add(0x9E37_79B9);

                        let volume_nodes = k * log_n;
                        let surface_nodes = block_m.map(|m| {
                            // SA(m,..,m) = rank * m^(rank-1)
                            (rank * (m.pow(rank as u32 - 1) as usize)) * log_n
                        });
                        let position = surface_nodes.map(|s| {
                            if volume_nodes > s {
                                (proof.proof_nodes.len() as f64 - s as f64)
                                    / (volume_nodes as f64 - s as f64)
                            } else {
                                0.0
                            }
                        });

                        rows.push(Row {
                            rank,
                            side,
                            alignment: align_name,
                            n_total,
                            order: order_name,
                            construction: cname,
                            shape,
                            k,
                            proof_nodes: proof.proof_nodes.len(),
                            proof_bytes: proof_size_bytes(&proof),
                            volume_nodes,
                            surface_nodes,
                            position_frac: position,
                            proof_ms: pt,
                            verify_ms: vt,
                        });
                    }
                }
              }
            }
        }
    }

    let dir = "benches/results";
    fs::create_dir_all(dir).expect("create results dir");
    let path = format!("{dir}/subset-proof-geometry.csv");
    let mut f = File::create(&path).expect("create csv");
    writeln!(
        f,
        "rank,grid_side,alignment,n_total,leaf_order,construction,shape,k_chunks,proof_nodes,\
proof_size_bytes,volume_bound_nodes,surface_bound_nodes,position_frac,\
proof_ms,proof_ci95_low,proof_ci95_high,verify_ms,verify_ci95_low,verify_ci95_high,trials"
    )
    .unwrap();
    for r in &rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            r.rank,
            r.side,
            r.alignment,
            r.n_total,
            r.order,
            r.construction,
            r.shape,
            r.k,
            r.proof_nodes,
            r.proof_bytes,
            r.volume_nodes,
            r.surface_nodes.map_or(String::new(), |v| v.to_string()),
            r.position_frac.map_or(String::new(), |v| format!("{v:.4}")),
            r.proof_ms.0,
            r.proof_ms.1,
            r.proof_ms.2,
            r.verify_ms.0,
            r.verify_ms.1,
            r.verify_ms.2,
            TRIALS
        )
        .unwrap();
    }
    eprintln!("wrote {path} ({} rows)", rows.len());
}
