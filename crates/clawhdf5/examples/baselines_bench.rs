//! P1.2b comparable-primitive baseline benchmark harness.
//!
//! Drives the three `clawhdf5_format::baselines` backends (flat SHA-256,
//! per-chunk HMAC-SHA-256, per-chunk Ed25519) through the same metrics
//! measured for the Merkle tree (§6.6: storage overhead, commit time,
//! single-chunk/whole-dataset verification latency, append/update cost,
//! and per-chunk-sig subset-proof size), so Phase 1 results can be reported
//! relative to measured baselines rather than asserted in the abstract.
//!
//! Run with:
//!   cargo run -p clawhdf5 --release --example baselines_bench --features parallel
//!     -- [N_CHUNKS] [TRIALS] [NOAA_PATH] [NOAA_DATASET]
//!
//! Defaults: N_CHUNKS=256, TRIALS=5 (the spec's 30 trials is the target for
//! a full run; override via argv for the publishable numbers). Chunk sizes
//! are fixed at 64 KB / 256 KB / 1 MB to match `hash_bench.rs`.
//!
//! Output: crates/clawhdf5-format/benches/results/baselines-$(hostname).csv
//! (gitignored by `benches/results/` — see the run's printed note on
//! committing it deliberately).
//!
//! NOAA_PATH/NOAA_DATASET are optional: if given, the harness additionally
//! extracts real on-disk chunk bytes from that dataset (filtered/compressed
//! chunks are hashed as-is, same as the existing filter-agnostic Merkle
//! code) and reruns the same suite tagged `source=NOAA`. If extraction
//! fails or the chunk-index type is unsupported, that source is skipped
//! with a printed warning rather than aborting the synthetic run. Defaults
//! to the `Rad` dataset (GOES-18 ABI L1b radiance) when present.

use clawhdf5::File;
use clawhdf5_format::baselines::{
    BaselineError, FlatHashBackend, PerChunkMacBackend, PerChunkSigBackend,
};
use clawhdf5_format::chunked_read::collect_chunk_info;
use clawhdf5_format::data_layout::DataLayout;
use clawhdf5_format::extensible_array::{ExtensibleArrayHeader, read_extensible_array_chunks};
use clawhdf5_format::filter_pipeline::FilterPipeline;
use clawhdf5_format::fixed_array::{FixedArrayHeader, read_fixed_array_chunks};
use clawhdf5_format::group_v2;
use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature;
use clawhdf5_format::superblock::Superblock;

use std::fs;
use std::time::Instant;

const CHUNK_SIZES_KB: &[usize] = &[64, 256, 1024];

struct Row {
    backend: &'static str,
    metric: &'static str,
    n_chunks: usize,
    chunk_size_kb: usize,
    value: f64,
    unit: &'static str,
    supported: bool,
    source: &'static str,
    trial: usize,
}

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Deterministic, non-trivially-compressible chunk data — same pattern as
/// `hash_bench.rs::make_chunk`, salted by chunk index so chunks differ.
fn make_chunk(size: usize, idx: usize) -> Vec<u8> {
    (0..size)
        .map(|i| ((i ^ (i >> 8) ^ (i >> 16)).wrapping_add(idx)) as u8)
        .collect()
}

/// Run the full metric battery for one (chunk_size, n_chunks, source) cell
/// against all three backends, appending rows to `rows`.
fn bench_cell(
    rows: &mut Vec<Row>,
    chunks: &[&[u8]],
    chunk_size_kb: usize,
    trials: usize,
    source: &'static str,
) {
    let n_chunks = chunks.len();
    if n_chunks == 0 {
        return;
    }

    // --- flat hash ---
    for trial in 1..=trials {
        let start = Instant::now();
        let backend = FlatHashBackend::commit(chunks);
        let commit_ns = start.elapsed().as_nanos() as f64;
        rows.push(Row {
            backend: "flat",
            metric: "commit_time",
            n_chunks,
            chunk_size_kb,
            value: commit_ns,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let start = Instant::now();
        let ok = backend.verify_dataset(chunks);
        let verify_ns = start.elapsed().as_nanos() as f64;
        assert!(ok, "flat hash failed to verify its own commitment");
        rows.push(Row {
            backend: "flat",
            metric: "verify_dataset_latency",
            n_chunks,
            chunk_size_kb,
            value: verify_ns,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        // append/update are both O(N) re-commits for a flat hash; that cost
        // *is* the capability gap being measured.
        let mut all: Vec<&[u8]> = chunks.to_vec();
        let extra = make_chunk(chunk_size_kb * 1024, n_chunks);
        all.push(&extra);
        let mut mutable_backend = backend;
        let start = Instant::now();
        mutable_backend.append(&all);
        rows.push(Row {
            backend: "flat",
            metric: "append_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let start = Instant::now();
        mutable_backend.update(&all);
        rows.push(Row {
            backend: "flat",
            metric: "update_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
    }
    let flat = FlatHashBackend::commit(chunks);
    rows.push(Row {
        backend: "flat",
        metric: "metadata_bytes",
        n_chunks,
        chunk_size_kb,
        value: flat.metadata_bytes() as f64,
        unit: "bytes",
        supported: true,
        source,
        trial: 1,
    });
    match flat.verify_chunk(0, chunks[0]) {
        Ok(_) => unreachable!("flat hash must not support verify_chunk"),
        Err(BaselineError::Unsupported { .. }) => rows.push(Row {
            backend: "flat",
            metric: "verify_chunk_latency",
            n_chunks,
            chunk_size_kb,
            value: 0.0,
            unit: "ns",
            supported: false,
            source,
            trial: 1,
        }),
        Err(e) => panic!("unexpected flat hash error: {e}"),
    }
    rows.push(Row {
        backend: "flat",
        metric: "public_verification",
        n_chunks,
        chunk_size_kb,
        value: 1.0,
        unit: "bool",
        supported: true,
        source,
        trial: 1,
    });

    // --- per-chunk MAC ---
    for trial in 1..=trials {
        let start = Instant::now();
        let mut backend = PerChunkMacBackend::commit(chunks);
        rows.push(Row {
            backend: "mac",
            metric: "commit_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let start = Instant::now();
        let ok = backend.verify_chunk(0, chunks[0]);
        rows.push(Row {
            backend: "mac",
            metric: "verify_chunk_latency",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
        assert_eq!(ok, Ok(true));

        let start = Instant::now();
        let ok = backend.verify_dataset(chunks);
        rows.push(Row {
            backend: "mac",
            metric: "verify_dataset_latency",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
        assert!(ok);

        let extra = make_chunk(chunk_size_kb * 1024, n_chunks);
        let start = Instant::now();
        backend.append(&extra);
        rows.push(Row {
            backend: "mac",
            metric: "append_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let updated = make_chunk(chunk_size_kb * 1024, 0xDEAD);
        let start = Instant::now();
        backend.update(0, &updated).expect("chunk 0 was committed");
        rows.push(Row {
            backend: "mac",
            metric: "update_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
    }
    let mac = PerChunkMacBackend::commit(chunks);
    rows.push(Row {
        backend: "mac",
        metric: "metadata_bytes",
        n_chunks,
        chunk_size_kb,
        value: mac.metadata_bytes() as f64,
        unit: "bytes",
        supported: true,
        source,
        trial: 1,
    });
    rows.push(Row {
        backend: "mac",
        metric: "public_verification",
        n_chunks,
        chunk_size_kb,
        value: 0.0,
        unit: "bool",
        supported: false,
        source,
        trial: 1,
    });

    // --- per-chunk Ed25519 ---
    for trial in 1..=trials {
        let start = Instant::now();
        let mut backend = PerChunkSigBackend::commit(chunks);
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "commit_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let start = Instant::now();
        let ok = backend.verify_chunk(0, chunks[0]);
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "verify_chunk_latency",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
        assert_eq!(ok, Ok(true));

        let start = Instant::now();
        let ok = backend.verify_dataset(chunks);
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "verify_dataset_latency",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
        assert!(ok);

        let extra = make_chunk(chunk_size_kb * 1024, n_chunks);
        let start = Instant::now();
        backend.append(&extra);
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "append_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let updated = make_chunk(chunk_size_kb * 1024, 0xDEAD);
        let start = Instant::now();
        backend.update(0, &updated).expect("chunk 0 was committed");
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "update_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
    }
    let sig = PerChunkSigBackend::commit(chunks);
    rows.push(Row {
        backend: "sig_ed25519",
        metric: "metadata_bytes",
        n_chunks,
        chunk_size_kb,
        value: sig.metadata_bytes() as f64,
        unit: "bytes",
        supported: true,
        source,
        trial: 1,
    });
    rows.push(Row {
        backend: "sig_ed25519",
        metric: "public_verification",
        n_chunks,
        chunk_size_kb,
        value: 1.0,
        unit: "bool",
        supported: true,
        source,
        trial: 1,
    });
    let k = (n_chunks / 10).max(1);
    rows.push(Row {
        backend: "sig_ed25519",
        metric: "subset_proof_bytes",
        n_chunks,
        chunk_size_kb,
        value: PerChunkSigBackend::subset_proof_bytes(k) as f64,
        unit: "bytes",
        supported: true,
        source,
        trial: 1,
    });
}

/// Best-effort extraction of raw on-disk chunk bytes from a real HDF5/
/// NetCDF-4 file. Returns `None` (with a printed warning) rather than
/// erroring out, since the NOAA leg of this harness is optional.
///
/// The bytes returned are whatever is physically stored per chunk —
/// filtered/compressed if the dataset uses a filter pipeline. The baseline
/// backends (like `merkle.rs`) are filter-agnostic: they hash/sign whatever
/// byte slice they're given, so this matches how the existing Merkle code
/// treats chunks and lets the harness run against real (compressed) NOAA
/// products instead of only the synthetic, uncompressed data.
fn extract_noaa_chunks(
    path: &str,
    dataset_override: Option<&str>,
) -> Option<(Vec<Vec<u8>>, usize)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("warning: failed to open NOAA file {path}: {e}");
            return None;
        }
    };
    let datasets = file.root().datasets().ok()?;
    let test_dataset = dataset_override.map(|s| s.to_string()).or_else(|| {
        if datasets.contains(&"Rad".to_string()) {
            Some("Rad".to_string())
        } else {
            datasets.first().cloned()
        }
    })?;

    let file_data = fs::read(path).ok()?;
    let sig_offset = signature::find_signature(&file_data).ok()?;
    let superblock = Superblock::parse(&file_data, sig_offset).ok()?;
    let addr = group_v2::resolve_path_any(&file_data, &superblock, &test_dataset).ok()?;
    let header = ObjectHeader::parse(
        &file_data,
        addr as usize,
        superblock.offset_size,
        superblock.length_size,
    )
    .ok()?;

    let pipeline = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .and_then(|msg| FilterPipeline::parse(&msg.data).ok());
    if let Some(p) = &pipeline {
        println!(
            "  note: dataset '{test_dataset}' has {} filter(s) applied; hashing on-disk (filtered) chunk bytes",
            p.filters.len()
        );
    }

    let layout_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DataLayout)?;
    let layout = DataLayout::parse(
        &layout_msg.data,
        superblock.offset_size,
        superblock.length_size,
    )
    .ok()?;
    let shape = file.dataset(&test_dataset).ok()?.shape().ok()?;

    let (chunks, chunk_dims, element_size) = match &layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            ..
        } => {
            let addr = (*btree_address)?;
            let rank = shape.len();
            let spatial_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            let elem_size = *chunk_dimensions.last().unwrap_or(&1) as usize;
            let chunks = match (*version, *chunk_index_type) {
                (3, _) => collect_chunk_info(
                    &file_data,
                    addr,
                    rank + 1,
                    superblock.offset_size,
                    superblock.length_size,
                )
                .ok()?,
                (4, Some(3)) => {
                    let fa = FixedArrayHeader::parse(
                        &file_data,
                        addr as usize,
                        superblock.offset_size,
                        superblock.length_size,
                    )
                    .ok()?;
                    read_fixed_array_chunks(
                        &file_data,
                        &fa,
                        &shape,
                        &spatial_dims,
                        elem_size as u32,
                        superblock.offset_size,
                        superblock.length_size,
                    )
                    .ok()?
                }
                (4, Some(4)) => {
                    let ea = ExtensibleArrayHeader::parse(
                        &file_data,
                        addr as usize,
                        superblock.offset_size,
                        superblock.length_size,
                    )
                    .ok()?;
                    read_extensible_array_chunks(
                        &file_data,
                        &ea,
                        &shape,
                        &spatial_dims,
                        elem_size as u32,
                        superblock.offset_size,
                        superblock.length_size,
                    )
                    .ok()?
                }
                _ => {
                    eprintln!(
                        "warning: unsupported chunk index for '{test_dataset}'; skipping NOAA leg"
                    );
                    return None;
                }
            };
            (chunks, spatial_dims, elem_size)
        }
        _ => {
            eprintln!("warning: dataset '{test_dataset}' is not chunked; skipping NOAA leg");
            return None;
        }
    };

    let chunk_elements: usize = chunk_dims.iter().map(|&d| d as usize).product();
    let nominal_size = chunk_elements * element_size;

    let mut out = Vec::with_capacity(chunks.len());
    for c in &chunks {
        let start = c.address as usize;
        let end = start + c.chunk_size as usize;
        if end > file_data.len() {
            eprintln!(
                "warning: chunk at 0x{:x} extends past EOF; skipping NOAA leg",
                c.address
            );
            return None;
        }
        out.push(file_data[start..end].to_vec());
    }
    if out.is_empty() {
        return None;
    }
    Some((out, nominal_size.max(1) / 1024))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let n_chunks: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let trials: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let noaa_path = args.get(3).cloned();
    let noaa_dataset = args.get(4).cloned();

    println!("=== P1.2b baseline benchmark harness ===");
    println!("n_chunks={n_chunks} trials={trials}");

    let host = hostname();
    let cpu = cpu_model();
    let mut rows: Vec<Row> = Vec::new();

    for &kb in CHUNK_SIZES_KB {
        println!("--- synthetic: {kb} KB chunks ---");
        let owned: Vec<Vec<u8>> = (0..n_chunks).map(|i| make_chunk(kb * 1024, i)).collect();
        let refs: Vec<&[u8]> = owned.iter().map(|c| c.as_slice()).collect();
        bench_cell(&mut rows, &refs, kb, trials, "synthetic");
    }

    if let Some(path) = noaa_path {
        println!("--- NOAA: {path} ---");
        if let Some((owned, kb)) = extract_noaa_chunks(&path, noaa_dataset.as_deref()) {
            let refs: Vec<&[u8]> = owned.iter().map(|c| c.as_slice()).collect();
            println!("  extracted {} chunks (~{kb} KB each)", refs.len());
            bench_cell(&mut rows, &refs, kb, trials, "NOAA");
        }
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let results_dir = format!("{manifest_dir}/../clawhdf5-format/benches/results");
    fs::create_dir_all(&results_dir)?;
    let out_path = format!("{results_dir}/baselines-{host}.csv");

    let mut csv = String::from(
        "backend,metric,n_chunks,chunk_size_kb,value,unit,supported,source,trial,hostname,cpu_model\n",
    );
    for r in &rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            csv_field(r.backend),
            csv_field(r.metric),
            r.n_chunks,
            r.chunk_size_kb,
            r.value,
            csv_field(r.unit),
            r.supported,
            csv_field(r.source),
            r.trial,
            csv_field(&host),
            csv_field(&cpu),
        ));
    }
    fs::write(&out_path, csv)?;

    println!();
    println!("Wrote {} rows to {out_path}", rows.len());
    println!(
        "Note: benches/results/ is gitignored — `git add -f` this file if you intend to commit it as a P1.2b artifact."
    );

    Ok(())
}
