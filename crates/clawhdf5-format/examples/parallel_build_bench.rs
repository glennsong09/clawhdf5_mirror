//! Benchmark parallel vs sequential Merkle tree building.
//!
//! Usage: cargo run --example parallel_build_bench --release --features "merkle,parallel"
//!
//! Output CSV columns: mode, n_chunks, wall_time_ms, n_threads, trial, hostname

use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn bench_build(chunks: &[&[u8]], alg: HashAlg, parallel: bool) -> (std::time::Duration, [u8; 32]) {
    let start = Instant::now();
    let tree = if parallel {
        MerkleTree::from_chunks_parallel(chunks, alg)
    } else {
        MerkleTree::from_chunks(chunks, alg)
    };
    let elapsed = start.elapsed();
    let mut root = [0u8; 32];
    root.copy_from_slice(tree.root());
    (elapsed, root)
}

fn main() {
    let hostname = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let output_path = format!(
        "crates/clawhdf5-format/benches/results/parallel-build-{}.csv",
        hostname
    );

    // Get number of threads
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!("=== Parallel vs Sequential Merkle Build Benchmark ===");
    println!("Hostname: {}", hostname);
    println!("Threads: {}", n_threads);
    println!("Output: {}\n", output_path);

    // Chunk sizes to test
    let n_values: Vec<usize> = vec![10_000, 100_000, 1_000_000];
    let chunk_size = 1024; // 1 KB chunks
    let warmup_iters = 3;
    let n_trials = 30;

    // Using BLAKE3 as the representative algorithm
    let alg = HashAlg::Blake3;

    // Collect all trial results
    let mut results: Vec<(String, usize, f64, usize, usize, String)> = Vec::new();

    for &n in &n_values {
        println!("--- N = {} chunks ({}) ---", n, format_bytes(n * chunk_size));

        // Generate chunks
        print!("  Generating chunks... ");
        std::io::stdout().flush().unwrap();
        let chunks: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let mut chunk = vec![0u8; chunk_size];
                for (j, byte) in chunk.iter_mut().enumerate() {
                    *byte = ((i * 31 + j * 17) % 256) as u8;
                }
                chunk
            })
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        println!("done");

        // Warmup
        print!("  Warmup... ");
        std::io::stdout().flush().unwrap();
        for _ in 0..warmup_iters {
            let _ = bench_build(&refs, alg, false);
            let _ = bench_build(&refs, alg, true);
        }
        println!("done");

        // Sequential trials
        print!("  Sequential ({} trials)... ", n_trials);
        std::io::stdout().flush().unwrap();
        let mut seq_times = Vec::with_capacity(n_trials);
        for trial in 1..=n_trials {
            let (elapsed, _) = bench_build(&refs, alg, false);
            let time_ms = elapsed.as_secs_f64() * 1000.0;
            seq_times.push(time_ms);
            results.push((
                "sequential".to_string(),
                n,
                time_ms,
                1, // sequential uses 1 thread
                trial,
                hostname.clone(),
            ));
        }
        seq_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let seq_median = seq_times[n_trials / 2];
        println!("median {:.2}ms", seq_median);

        // Parallel trials
        print!("  Parallel ({} trials)... ", n_trials);
        std::io::stdout().flush().unwrap();
        let mut par_times = Vec::with_capacity(n_trials);
        for trial in 1..=n_trials {
            let (elapsed, _) = bench_build(&refs, alg, true);
            let time_ms = elapsed.as_secs_f64() * 1000.0;
            par_times.push(time_ms);
            results.push((
                "parallel".to_string(),
                n,
                time_ms,
                n_threads,
                trial,
                hostname.clone(),
            ));
        }
        par_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let par_median = par_times[n_trials / 2];
        println!("median {:.2}ms", par_median);

        let speedup = seq_median / par_median;
        println!("  Speedup: {:.2}x\n", speedup);
    }

    // Write CSV
    let mut file = File::create(&output_path).expect("Failed to create output file");
    writeln!(file, "mode,n_chunks,wall_time_ms,n_threads,trial,hostname").unwrap();
    for (mode, n, time_ms, threads, trial, host) in &results {
        writeln!(file, "{},{},{:.3},{},{},{}", mode, n, time_ms, threads, trial, host).unwrap();
    }

    println!("Results saved to: {}", output_path);
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / 1024.0 / 1024.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
