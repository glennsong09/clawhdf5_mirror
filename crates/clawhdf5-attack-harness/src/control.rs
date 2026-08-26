//! Untampered control arm for the P2.4 detection study (RQ5).
//!
//! The attack matrix answers "does the verifier reject a tampered file?" and
//! nothing else. On its own that is a true-positive rate with no denominator
//! for the other column: a verifier that returned `Err` unconditionally would
//! score 33/33 on it. RQ5 asks for true *and* false positives, so this module
//! supplies the missing arm -- the same fixtures, the same verifiers, no
//! tampering -- and asks whether anything is rejected that should not be.
//!
//! A false positive here is any untampered input for which a verifier returns
//! `Err(_)` or `Ok(false)`. Every verifier in this crate returns
//! `Result<bool, MerkleError>`, so that test is uniform across all of them.
//!
//! Output: `attack-results/control.csv`, schema
//! `control_id, dataset, verifier_fn, trials, false_positives, latency_ms`.
//! `trials` is per-invocation, so the per-chunk control reports one trial per
//! chunk rather than one per dataset -- that is where the denominator comes
//! from.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use clawhdf5_format::merkle::{verify_chunk, verify_dataset, verify_root, verify_signed_root};
use clawhdf5_format::subset_proof::{ChunkData, LeafOrder, extract_subset, verify_subset};
use clawhdf5_format::merkle::{Dataset, HashAlg, MerkleAttr, MerkleTree};

use crate::attacks::{
    build_merkle_state, companion_hash, dataset_view, range_1d, sign_honest_root,
    whole_dataset_grid,
};
use crate::fixture::HarnessDataset;

/// One control observation: `trials` untampered verifications, of which
/// `false_positives` were wrongly rejected.
pub struct ControlResult {
    pub control_id: &'static str,
    pub dataset: &'static str,
    pub verifier_fn: &'static str,
    pub trials: usize,
    pub false_positives: usize,
    pub latency: Duration,
}

impl ControlResult {
    #[must_use]
    pub fn latency_ms(&self) -> f64 {
        self.latency.as_secs_f64() * 1000.0
    }
}

/// `Ok(true)` is the only non-false-positive outcome on untampered input.
fn is_false_positive(r: &Result<bool, clawhdf5_format::merkle::MerkleError>) -> bool {
    !matches!(r, Ok(true))
}

/// C1 -- `verify_root` on an untampered dataset.
pub fn c1_root(ds: &HarnessDataset) -> ControlResult {
    let (_, attr, nodes) = build_merkle_state(ds);
    let view = dataset_view(attr, nodes, ds, &ds.bytes);
    let t = Instant::now();
    let r = verify_root(&view);
    ControlResult {
        control_id: "C1",
        dataset: ds.label,
        verifier_fn: "verify_root",
        trials: 1,
        false_positives: usize::from(is_false_positive(&r)),
        latency: t.elapsed(),
    }
}

/// C2 -- `verify_dataset` on an untampered dataset.
pub fn c2_dataset(ds: &HarnessDataset) -> ControlResult {
    let (_, attr, nodes) = build_merkle_state(ds);
    let view = dataset_view(attr, nodes, ds, &ds.bytes);
    let t = Instant::now();
    let r = verify_dataset(&view);
    ControlResult {
        control_id: "C2",
        dataset: ds.label,
        verifier_fn: "verify_dataset",
        trials: 1,
        false_positives: usize::from(is_false_positive(&r)),
        latency: t.elapsed(),
    }
}

/// C3 -- `verify_chunk` over *every* chunk of an untampered dataset. This is
/// the control with a denominator worth quoting: one trial per chunk.
pub fn c3_every_chunk(ds: &HarnessDataset) -> ControlResult {
    let (_, attr, nodes) = build_merkle_state(ds);
    let view = dataset_view(attr, nodes, ds, &ds.bytes);
    let n = ds.chunk_count();
    let t = Instant::now();
    let fp = (0..n).filter(|&i| is_false_positive(&verify_chunk(&view, i))).count();
    ControlResult {
        control_id: "C3",
        dataset: ds.label,
        verifier_fn: "verify_chunk",
        trials: n,
        false_positives: fp,
        latency: t.elapsed(),
    }
}

/// C4 -- `verify_subset` on an honest whole-dataset proof with every claimed
/// chunk delivered intact.
pub fn c4_subset_whole(ds: &HarnessDataset) -> ControlResult {
    subset_control("C4", ds, 0..ds.chunk_count() as u64)
}

/// C5 -- `verify_subset` on an honest *partial* selection. A verifier that
/// only tolerates whole-dataset proofs would pass C4 and fail here.
pub fn c5_subset_partial(ds: &HarnessDataset) -> ControlResult {
    let half = (ds.chunk_count() as u64).div_ceil(2);
    subset_control("C5", ds, 0..half)
}

fn subset_control(
    id: &'static str,
    ds: &HarnessDataset,
    range: std::ops::Range<u64>,
) -> ControlResult {
    let (tree, _, _) = build_merkle_state(ds);
    let grid = whole_dataset_grid(ds);
    let sel = range_1d(range);
    let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
    let delivered: Vec<ChunkData<'_>> = proof
        .chunk_indices
        .iter()
        .map(|&idx| ChunkData { index: idx, data: ds.chunk(&ds.bytes, idx) })
        .collect();
    let t = Instant::now();
    let r = verify_subset(
        tree.root(),
        HashAlg::Blake3,
        &delivered,
        &proof,
        &grid,
        &grid.grid_hash,
        &sel,
        LeafOrder::RowMajor,
    );
    ControlResult {
        control_id: id,
        dataset: ds.label,
        verifier_fn: "verify_subset",
        trials: 1,
        false_positives: usize::from(is_false_positive(&r)),
        latency: t.elapsed(),
    }
}

/// C6 -- `verify_signed_root` on an honestly signed, untampered state. The
/// signed path is where T1d and T6b are detected, so a false positive here
/// would undercut the paper's "signed counterpart is detected" claim.
pub fn c6_signed_root() -> ControlResult {
    let chunks: Vec<&[u8]> = vec![b"chunk-a", b"chunk-b", b"chunk-c", b"chunk-d"];
    let tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
    let mut nodes = Vec::new();
    for n in tree.nodes() {
        nodes.extend_from_slice(n);
    }
    let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash(&nodes));
    let version = 1u64;
    let timestamp = 1_700_000_000u64;
    let verifier = sign_honest_root(&attr, version, timestamp);
    let view = Dataset::from_owned(attr, nodes, chunks.clone());
    let t = Instant::now();
    let r = verify_signed_root(&view, version, timestamp, &verifier);
    ControlResult {
        control_id: "C6",
        dataset: "n/a",
        verifier_fn: "verify_signed_root",
        trials: 1,
        false_positives: usize::from(is_false_positive(&r)),
        latency: t.elapsed(),
    }
}

/// Run every per-dataset control against one dataset.
pub fn run_dataset_controls(ds: &HarnessDataset) -> Vec<ControlResult> {
    vec![
        c1_root(ds),
        c2_dataset(ds),
        c3_every_chunk(ds),
        c4_subset_whole(ds),
        c5_subset_partial(ds),
    ]
}

#[must_use]
pub fn to_csv(results: &[ControlResult]) -> String {
    let mut out = String::new();
    out.push_str("control_id,dataset,verifier_fn,trials,false_positives,latency_ms\n");
    for r in results {
        let _ = writeln!(
            out,
            "{},{},{},{},{},{:.4}",
            r.control_id,
            r.dataset,
            r.verifier_fn,
            r.trials,
            r.false_positives,
            r.latency_ms()
        );
    }
    out
}

pub fn print_table(results: &[ControlResult]) {
    println!(
        "{:<4} {:<24} {:<20} {:>8} {:>16} {:>12}",
        "C#", "dataset", "verifier_fn", "trials", "false_positives", "latency_ms"
    );
    println!("{}", "-".repeat(90));
    for r in results {
        println!(
            "{:<4} {:<24} {:<20} {:>8} {:>16} {:>12.4}",
            r.control_id,
            r.dataset,
            r.verifier_fn,
            r.trials,
            r.false_positives,
            r.latency_ms()
        );
    }
    println!("{}", "-".repeat(90));
    let trials: usize = results.iter().map(|r| r.trials).sum();
    let fp: usize = results.iter().map(|r| r.false_positives).sum();
    println!("{fp}/{trials} untampered verifications wrongly rejected (false positives)");
}
