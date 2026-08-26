//! Overlay mesh for chunked datasets: what coarsening actually buys.
//!
//! `research-ideas.md` proposes grouping `k = 2^level` SFC-adjacent native
//! chunks into one coarse verification leaf, so the routine no-corruption
//! check gets a smaller proof and fewer verifier-side comparisons, with a
//! drill-down into the one bad group on the rare failure path. It lists five
//! things to check before any of that goes into a paper. This sweep measures
//! them rather than assuming them:
//!
//! 1. **Which cost this reduces.** Every row carries `bytes_hashed` next to
//!    `wire_bytes`. Recomputing a coarse leaf still hashes every byte
//!    underneath it, so if the hypothesis is right `bytes_hashed` is flat in
//!    `level` while `wire_bytes` falls -- the saving is RQ6 (proof size, WAN
//!    bandwidth, comparison count), not RQ2 (raw I/O). Flat-versus-falling is
//!    a measurement, so it is reported as one.
//! 2. **Sparse/unallocated chunks in a group.** The `sparse_grid` arm runs a
//!    grid whose chunk count is not a power of two, so real groups straddle
//!    the null-sentinel padding. `fill_in_nodes` shows what that costs.
//! 3. **Write amplification.** `write_path_nodes` is recorded per level.
//! 4. **Cheap re-binning of the RQ6 sweep.** `naive_rebin_coarse` is what a
//!    re-bin of the existing level-0 data would have predicted for the coarse
//!    count (`ceil(k_chunks / group)`); `n_coarse` is the truth. The gap is
//!    the answer to whether re-binning would have sufficed.
//! 5. **Tamper localization.** One chunk in the selection is corrupted; the
//!    row records whether the coarse pass localized it, and what the
//!    drill-down descent cost to name the exact chunk.
//!
//! Crossed with the two factors RQ6 already established as the ones that
//! matter -- leaf ordering and proof construction -- plus selection shape and
//! alignment, since an aligned block is the case where a coarse group is
//! covered whole and a ragged one is where the boundary tax lands.
//!
//! Proof sizes and node counts are computed, never read from storage, so this
//! benchmark carries no storage class. Timings follow the Statistical Protocol
//! (30 measured trials after 5 discarded warmups, median + 95% bootstrap CI).
//!
//! Output: `benches/results/overlay-mesh.csv`.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use clawhdf5_format::overlay_mesh::{
    MeshLevel, MeshProof, MeshVerdict, extract_subset_mesh, localize_in_group, verify_subset_mesh,
    write_path_len,
};
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    ChunkData, ChunkGridParams, LeafOrder, ProofConstruction, leaf_index_for_coord,
};
use clawhdf5_format::verification_grid::LayoutClass;

const HASH_SIZE: usize = 32;
const WARMUP_TRIALS: usize = 5;
const TRIALS: usize = 30;
const BOOTSTRAP_ITERATIONS: usize = 2000;
const ALG: HashAlg = HashAlg::Blake3;

/// Bytes per native chunk. Fixed across the sweep so `bytes_hashed` isolates
/// the *number* of chunks the verifier must hash from their size.
const CHUNK_BYTES: usize = 4096;

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

struct Dataset {
    tree: MerkleTree,
    grid: ChunkGridParams,
    payload: Vec<Vec<u8>>,
}

/// A `dims`-shaped grid of one-chunk-per-cell data under `order`. `dims` need
/// not be a power of two on any axis; when it is not, the tree pads with the
/// null sentinel and real groups straddle that padding -- which is exactly the
/// sparse-chunk case the design left open.
fn build(dims: &[u64], order: LeafOrder) -> Dataset {
    let rank = dims.len();
    let grid = ChunkGridParams::new(
        dims.to_vec(),
        vec![1u64; rank],
        4,
        LayoutClass::Chunked,
        ALG,
    );
    let n_per_dim = grid.n_chunks_per_dim();
    let total = grid.total_chunk_count() as usize;

    // Leaf index space is the SFC's image, which for a non-power-of-two extent
    // is sparser than the chunk count; size the payload to hold it.
    let mut max_leaf = 0usize;
    let mut placements: Vec<(usize, usize)> = Vec::with_capacity(total);
    let mut coord = vec![0u64; rank];
    for f in 0..total {
        let mut rem = f as u64;
        for d in (0..rank).rev() {
            coord[d] = rem % dims[d];
            rem /= dims[d];
        }
        let leaf = leaf_index_for_coord(&coord, &n_per_dim, order) as usize;
        max_leaf = max_leaf.max(leaf);
        placements.push((leaf, f));
    }

    // Unplaced slots stay empty and are hashed as the null sentinel, standing
    // in for unallocated chunks.
    let mut leaf_hashes = vec![ALG.null_sentinel(); max_leaf + 1];
    let mut payload = vec![Vec::new(); max_leaf + 1];
    for (leaf, f) in placements {
        let mut data = vec![0u8; CHUNK_BYTES];
        data[..8].copy_from_slice(&(f as u64).to_le_bytes());
        leaf_hashes[leaf] = ALG.hash_leaf(&data);
        payload[leaf] = data;
    }
    let tree = MerkleTree::from_leaf_hashes(&leaf_hashes, ALG);
    Dataset {
        tree,
        grid,
        payload,
    }
}

/// The selection geometries. `origin` positions a block on or off the quadrant
/// grid; alignment is the variable RQ6 found outranks the curve, and it is
/// what decides here whether a coarse group is covered whole.
fn selection(shape: &str, side: u64, rank: usize, m: u64, origin: u64) -> Selection {
    match shape {
        "all" => Selection::All,
        "block" => Selection::slice(
            &(0..rank)
                .map(|_| origin..(origin + m).min(side))
                .collect::<Vec<_>>(),
        ),
        "strided" => {
            let mut points = Vec::new();
            let total = m.pow(rank as u32);
            for f in 0..total {
                let mut rem = f;
                let mut coord = vec![0u64; rank];
                for d in (0..rank).rev() {
                    coord[d] = rem % m;
                    rem /= m;
                }
                let mut p: Vec<u64> = (0..rank).map(|d| origin + coord[d]).collect();
                p[rank - 1] = origin + coord[rank - 1] * 2;
                if p.iter().all(|&c| c < side) {
                    points.push(p);
                }
            }
            Selection::Points(points)
        }
        "sparse" => {
            let k = m.pow(rank as u32);
            let total = side.pow(rank as u32);
            let mut rng = Xorshift64(0x5EED_1234);
            let mut chosen: Vec<u64> = Vec::with_capacity(k as usize);
            while (chosen.len() as u64) < k {
                let f = rng.next_u64() % total;
                if !chosen.contains(&f) {
                    chosen.push(f);
                }
            }
            chosen.sort_unstable();
            let points = chosen
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
            Selection::Points(points)
        }
        other => panic!("unknown shape: {other}"),
    }
}

struct Row {
    grid_label: &'static str,
    rank: usize,
    side: u64,
    n_chunks: usize,
    padded_leaves: usize,
    order: &'static str,
    construction: &'static str,
    shape: &'static str,
    alignment: &'static str,
    level: u32,
    group: u64,
    k_chunks: usize,
    n_coarse: usize,
    naive_rebin_coarse: usize,
    witness_nodes: usize,
    fill_in_nodes: usize,
    wire_bytes: usize,
    compares: usize,
    bytes_hashed: usize,
    write_path_nodes: usize,
    localized: &'static str,
    suspect_groups: usize,
    drill_rounds: usize,
    drill_wire_bytes: usize,
    two_phase_bytes: usize,
    extract_ms: (f64, f64, f64),
    verify_ms: (f64, f64, f64),
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

#[allow(clippy::too_many_arguments)]
fn measure(
    ds: &Dataset,
    sel: &Selection,
    order: LeafOrder,
    construction: ProofConstruction,
    level: MeshLevel,
    fine: &[usize],
    seed: u64,
) -> (MeshProof, (f64, f64, f64), (f64, f64, f64)) {
    let proof = extract_subset_mesh(&ds.tree, &ds.grid, sel, order, construction, level).unwrap();
    let chunks = deliver(fine, &ds.payload);

    // A size measured from a proof that does not verify is not a measurement.
    assert!(
        verify_subset_mesh(
            ds.tree.root(),
            ALG,
            &chunks,
            &proof,
            &ds.grid,
            &ds.grid.grid_hash,
            sel,
            order,
            construction,
            level,
        )
        .unwrap()
        .is_verified(),
        "mesh proof failed to verify before timing"
    );

    let mut extract_times = Vec::with_capacity(TRIALS);
    let mut verify_times = Vec::with_capacity(TRIALS);
    for t in 0..(WARMUP_TRIALS + TRIALS) {
        let t0 = Instant::now();
        let p = extract_subset_mesh(&ds.tree, &ds.grid, sel, order, construction, level).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let _ = verify_subset_mesh(
            ds.tree.root(),
            ALG,
            &chunks,
            &p,
            &ds.grid,
            &ds.grid.grid_hash,
            sel,
            order,
            construction,
            level,
        )
        .unwrap();
        let vt = t1.elapsed().as_secs_f64() * 1000.0;

        if t >= WARMUP_TRIALS {
            extract_times.push(dt);
            verify_times.push(vt);
        }
    }

    (
        proof,
        bootstrap_median_ci(&extract_times, seed),
        bootstrap_median_ci(&verify_times, seed ^ 0xA5A5),
    )
}

/// Corrupt one delivered chunk, run the coarse pass, and -- if it localized --
/// drill into the named group. Returns
/// `(localized, suspect_groups, rounds, wire_bytes)`.
#[allow(clippy::too_many_arguments)]
fn tamper_and_localize(
    ds: &Dataset,
    sel: &Selection,
    order: LeafOrder,
    construction: ProofConstruction,
    level: MeshLevel,
    fine: &[usize],
    proof: &MeshProof,
) -> (&'static str, usize, usize, usize) {
    if fine.is_empty() {
        return ("n/a", 0, 0, 0);
    }
    let victim = fine[fine.len() / 2];
    let evil = b"tampered-by-the-benchmark".to_vec();
    let chunks: Vec<ChunkData<'_>> = fine
        .iter()
        .map(|&i| ChunkData {
            index: i,
            data: if i == victim { &evil } else { &ds.payload[i] },
        })
        .collect();

    let verdict = verify_subset_mesh(
        ds.tree.root(),
        ALG,
        &chunks,
        proof,
        &ds.grid,
        &ds.grid.grid_hash,
        sel,
        order,
        construction,
        level,
    )
    .unwrap();

    let suspects = match verdict {
        MeshVerdict::Verified => return ("MISSED", 0, 0, 0),
        MeshVerdict::Suspect(g) => g,
    };
    let expected_group = level.coarse_index(victim);
    let exact = suspects.len() == 1 && suspects[0] == expected_group;

    // Drill into the group the coarse pass named. The verifier already holds
    // every delivered chunk's bytes; members it does not hold resolve to the
    // null sentinel, which is what they are.
    let mut recomputed: BTreeMap<usize, [u8; HASH_SIZE]> = BTreeMap::new();
    let (lo, hi) = level.leaf_range(expected_group);
    for leaf in lo..hi.min(ds.tree.padded_leaf_count()) {
        let bytes: &[u8] = if leaf == victim {
            &evil
        } else if fine.binary_search(&leaf).is_ok() {
            &ds.payload[leaf]
        } else {
            continue;
        };
        recomputed.insert(leaf, ALG.hash_leaf(bytes));
    }
    // Undelivered members are not corrupt; seed them from the companion so the
    // descent charges only for the branch that actually diverged.
    let internal = ds.tree.padded_leaf_count() - 1;
    for leaf in lo..hi.min(ds.tree.padded_leaf_count()) {
        recomputed
            .entry(leaf)
            .or_insert(ds.tree.nodes()[internal + leaf]);
    }

    let loc = localize_in_group(&ds.tree, &recomputed, level, expected_group).unwrap();
    let found = loc.bad_leaves == vec![victim];
    let label = match (exact, found) {
        (true, true) => "group+chunk",
        (true, false) => "group-only",
        (false, true) => "wide+chunk",
        (false, false) => "wide",
    };
    (label, suspects.len(), loc.rounds, loc.wire_bytes())
}

fn main() {
    let orders: [(&str, LeafOrder); 2] = [
        ("row_major", LeafOrder::RowMajor),
        ("morton", LeafOrder::Morton),
    ];
    let constructions: [(&str, ProofConstruction); 2] = [
        ("naive_dedup", ProofConstruction::NaiveDedup),
        ("canonical_pruned", ProofConstruction::CanonicalPruned),
    ];

    // (label, rank, side, block side m, levels to sweep)
    let configs: [(&str, usize, u64, u64, u32); 3] = [
        // Power-of-two grids: every ordering is a bijection onto 0..N, the
        // setting RQ6 established as the one where the comparison is clean.
        ("pow2", 2, 256, 32, 12),
        ("pow2", 3, 32, 8, 12),
        // Not a power of two on any axis: real chunks straddle null-sentinel
        // padding, which is the sparse/unallocated case.
        ("sparse_grid", 2, 200, 25, 12),
    ];

    let alignments: [(&str, fn(u64, u64) -> u64); 2] = [
        ("aligned", |side, m| (side / 4 / m.max(1)) * m),
        ("offset", |side, m| (side / 4 / m.max(1)) * m + m / 2 + 1),
    ];
    let shapes = ["all", "block", "strided", "sparse"];

    let mut rows: Vec<Row> = Vec::new();
    let mut seed = 0xD15E_A5E1u64;

    for &(grid_label, rank, side, m, max_level) in &configs {
        for &(order_name, order) in &orders {
            let ds = build(&vec![side; rank], order);
            let padded = ds.tree.padded_leaf_count();
            let n_chunks = ds.grid.total_chunk_count() as usize;
            let levels_here = max_level.min(padded.trailing_zeros());
            eprintln!(
                "{grid_label} rank {rank} side {side} order {order_name}: \
                 {n_chunks} chunks, {padded} padded leaves, levels 0..={levels_here}"
            );

            for &shape in &shapes {
                for &(align_name, origin_fn) in &alignments {
                    // "all" has no origin, so run it once.
                    if shape == "all" && align_name == "offset" {
                        continue;
                    }
                    let origin = origin_fn(side, m).min(side.saturating_sub(m));
                    let sel = selection(shape, side, rank, m, origin);
                    let fine = clawhdf5_format::subset_proof::extract_subset_with(
                        &ds.tree,
                        &ds.grid,
                        &sel,
                        order,
                        ProofConstruction::NaiveDedup,
                    )
                    .unwrap()
                    .chunk_indices;
                    let k_chunks = fine.len();
                    // Every level hashes exactly this many bytes: the payload
                    // is the native covering set regardless of coarsening.
                    let bytes_hashed = k_chunks * CHUNK_BYTES;

                    for &(cname, construction) in &constructions {
                        for l in 0..=levels_here {
                            let level = MeshLevel::new(l).unwrap();
                            let (proof, et, vt) =
                                measure(&ds, &sel, order, construction, level, &fine, seed);
                            seed = seed.wrapping_add(0x9E37_79B9);

                            let (localized, suspects, rounds, drill_bytes) = tamper_and_localize(
                                &ds,
                                &sel,
                                order,
                                construction,
                                level,
                                &fine,
                                &proof,
                            );

                            let group = level.group_size();
                            rows.push(Row {
                                grid_label,
                                rank,
                                side,
                                n_chunks,
                                padded_leaves: padded,
                                order: order_name,
                                construction: cname,
                                shape,
                                alignment: if shape == "all" { "n/a" } else { align_name },
                                level: l,
                                group,
                                k_chunks,
                                n_coarse: proof.coarse_indices.len(),
                                naive_rebin_coarse: k_chunks.div_ceil(group as usize),
                                witness_nodes: proof.proof_nodes.len(),
                                fill_in_nodes: proof.fill_in.len(),
                                wire_bytes: proof.wire_bytes(),
                                compares: proof.compare_count(),
                                bytes_hashed,
                                write_path_nodes: write_path_len(padded, level),
                                localized,
                                suspect_groups: suspects,
                                drill_rounds: rounds,
                                drill_wire_bytes: drill_bytes,
                                two_phase_bytes: proof.wire_bytes() + drill_bytes,
                                extract_ms: et,
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
    let path = format!("{dir}/overlay-mesh.csv");
    let mut f = File::create(&path).expect("create csv");
    writeln!(
        f,
        "grid,rank,side,n_chunks,padded_leaves,leaf_order,construction,shape,alignment,\
level,group_k,k_chunks,n_coarse,naive_rebin_coarse,witness_nodes,fill_in_nodes,wire_bytes,\
compares,bytes_hashed,write_path_nodes,localized,suspect_groups,drill_rounds,\
drill_wire_bytes,two_phase_bytes,extract_ms,extract_ci95_low,extract_ci95_high,\
verify_ms,verify_ci95_low,verify_ci95_high,trials"
    )
    .unwrap();
    for r in &rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\
{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            r.grid_label,
            r.rank,
            r.side,
            r.n_chunks,
            r.padded_leaves,
            r.order,
            r.construction,
            r.shape,
            r.alignment,
            r.level,
            r.group,
            r.k_chunks,
            r.n_coarse,
            r.naive_rebin_coarse,
            r.witness_nodes,
            r.fill_in_nodes,
            r.wire_bytes,
            r.compares,
            r.bytes_hashed,
            r.write_path_nodes,
            r.localized,
            r.suspect_groups,
            r.drill_rounds,
            r.drill_wire_bytes,
            r.two_phase_bytes,
            r.extract_ms.0,
            r.extract_ms.1,
            r.extract_ms.2,
            r.verify_ms.0,
            r.verify_ms.1,
            r.verify_ms.2,
            TRIALS
        )
        .unwrap();
    }
    eprintln!("wrote {path} ({} rows)", rows.len());
}
