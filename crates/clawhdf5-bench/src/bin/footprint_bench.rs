//! Memory Footprint Benchmark (Track 8.4)
//!
//! Measures HDF5 file size at various record counts:
//!   100, 1K, 10K, 50K, 100K records
//!
//! Reports:
//!   - File size on disk (bytes / KB / MB)
//!   - Bytes per record
//!   - Compression ratio (compressed vs uncompressed)
//!   - Ingestion throughput (records/second)
//!
//! Configuration matrix:
//!   - Text lengths: short (50 chars), medium (200 chars), long (1000 chars)
//!   - Embedding: 384-dim f32 (1536 bytes raw per record)
//!   - WAL: enabled and disabled
//!
//! # Usage
//! ```
//! cargo run --release --bin footprint_bench
//! ```

use std::time::Instant;

use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};
use tempfile::TempDir;

const EMBEDDING_DIM: usize = 384;

// Raw bytes per record: 384 f32 embeddings + median text + overhead
const RAW_BYTES_PER_FLOAT: usize = 4;

// ---------------------------------------------------------------------------
// Deterministic text generator (no randomness)
// ---------------------------------------------------------------------------

const WORD_BANK: &[&str] = &[
    "system",
    "architecture",
    "distributed",
    "memory",
    "vector",
    "embedding",
    "agent",
    "knowledge",
    "search",
    "retrieval",
    "temporal",
    "semantic",
    "episodic",
    "working",
    "consolidation",
    "importance",
    "activation",
    "cosine",
    "similarity",
    "hybrid",
    "keyword",
    "BM25",
    "index",
    "session",
    "context",
    "token",
    "chunk",
    "overlap",
    "inference",
    "pipeline",
    "latency",
    "throughput",
    "benchmark",
    "performance",
    "Rust",
    "async",
    "parallel",
    "concurrent",
    "thread",
    "atomic",
];

fn make_text(record_idx: usize, target_chars: usize) -> String {
    let mut result = String::with_capacity(target_chars + 50);
    let mut word_idx = record_idx % WORD_BANK.len();
    while result.len() < target_chars {
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(WORD_BANK[word_idx]);
        word_idx = (word_idx + 7) % WORD_BANK.len(); // stride 7 for variety
    }
    result.truncate(target_chars);
    result
}

fn make_embedding(record_idx: usize) -> Vec<f32> {
    // Deterministic non-zero embeddings to stress compression
    (0..EMBEDDING_DIM)
        .map(|i| ((record_idx * 31 + i * 17) % 1000) as f32 / 1000.0 - 0.5)
        .collect()
}

fn make_entries(n: usize, text_len: usize) -> Vec<MemoryEntry> {
    (0..n)
        .map(|i| MemoryEntry {
            chunk: make_text(i, text_len),
            embedding: make_embedding(i),
            source_channel: "footprint-bench".to_string(),
            timestamp: 1_000_000.0 + i as f64,
            session_id: format!("sess_{}", i / 50),
            tags: String::new(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Single footprint measurement
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct FootprintResult {
    n: usize,
    text_len: usize,
    wal_enabled: bool,
    compression: bool,
    file_bytes: u64,
    wal_bytes: u64,
    ingest_ms: f64,
    raw_bytes: u64,
}

impl FootprintResult {
    fn bytes_per_record(&self) -> u64 {
        self.file_bytes / self.n.max(1) as u64
    }
    fn compression_ratio(&self) -> f64 {
        self.raw_bytes as f64 / self.file_bytes.max(1) as f64
    }
    fn records_per_sec(&self) -> f64 {
        self.n as f64 / (self.ingest_ms / 1000.0).max(0.001)
    }
}

fn measure_footprint(
    n: usize,
    text_len: usize,
    wal_enabled: bool,
    compression: bool,
) -> FootprintResult {
    let dir = TempDir::new().expect("TempDir failed");
    let h5_path = dir.path().join("footprint.h5");

    let mut config = MemoryConfig::new(h5_path.clone(), "footprint-bench", EMBEDDING_DIM);
    config.wal_enabled = wal_enabled;
    config.compression = compression;
    config.compression_level = if compression { 6 } else { 0 };
    config.compact_threshold = 0.0;

    let mut memory = HDF5Memory::create(config).expect("HDF5Memory::create failed");

    let entries = make_entries(n, text_len);
    let raw_bytes = entries
        .iter()
        .map(|e| e.chunk.len() + e.embedding.len() * RAW_BYTES_PER_FLOAT)
        .sum::<usize>() as u64;

    // Batch ingest, measure time
    let t0 = Instant::now();
    memory.save_batch(entries).expect("save_batch failed");
    let ingest_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let file_bytes = std::fs::metadata(&h5_path).map(|m| m.len()).unwrap_or(0);

    let wal_path = h5_path.with_extension("h5.wal");
    let wal_bytes = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);

    FootprintResult {
        n,
        text_len,
        wal_enabled,
        compression,
        file_bytes,
        wal_bytes,
        ingest_ms,
        raw_bytes,
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn fmt_bytes(b: u64) -> String {
    if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}

fn print_table(results: &[FootprintResult], label: &str) {
    println!("### {label}");
    println!();
    println!(
        "{:>8}  {:>12}  {:>12}  {:>10}  {:>8}  {:>14}",
        "Records", "File Size", "Raw Data", "Bytes/Rec", "Ratio", "Throughput"
    );
    println!("{}", "-".repeat(76));
    for r in results {
        let wal_note = if r.wal_enabled && r.wal_bytes > 0 {
            format!(" (+{} WAL)", fmt_bytes(r.wal_bytes))
        } else {
            String::new()
        };
        println!(
            "{:>8}  {:>12}  {:>12}  {:>10}  {:>7.2}x  {:>11.0} rec/s{}",
            fmt_n(r.n),
            fmt_bytes(r.file_bytes),
            fmt_bytes(r.raw_bytes),
            fmt_bytes(r.bytes_per_record()),
            r.compression_ratio(),
            r.records_per_sec(),
            wal_note
        );
    }
    println!();
}

fn fmt_n(n: usize) -> String {
    match n {
        100 => "100".to_owned(),
        1_000 => "1K".to_owned(),
        10_000 => "10K".to_owned(),
        50_000 => "50K".to_owned(),
        100_000 => "100K".to_owned(),
        _ => n.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=================================================================");
    println!("  ClawhDF5 Memory Footprint Benchmark");
    println!("=================================================================");
    println!();
    println!("Embedding: 384-dim f32 = 1,536 bytes raw per record");
    println!("Text lengths: short=50 chars, medium=200 chars, long=1000 chars");
    println!();

    // Test scales
    let scales = [100usize, 1_000, 10_000, 50_000, 100_000];

    // --- Medium text, no compression, no WAL ---
    let mut results = Vec::new();
    for &n in &scales {
        eprint!("\r  medium text, no compression, no WAL: {n} records...");
        results.push(measure_footprint(n, 200, false, false));
    }
    eprintln!();
    print_table(&results, "Medium Text (200 chars), No Compression, No WAL");

    // --- Medium text, with compression, no WAL ---
    let mut results = Vec::new();
    for &n in &scales {
        eprint!("\r  medium text, compression, no WAL: {n} records...");
        results.push(measure_footprint(n, 200, false, true));
    }
    eprintln!();
    print_table(
        &results,
        "Medium Text (200 chars), Gzip Compression (level 6), No WAL",
    );

    // --- Text length comparison at 10K records ---
    println!("### Text Length Comparison at 10K Records (no compression, no WAL)");
    println!();
    println!(
        "{:>12}  {:>12}  {:>12}  {:>10}  {:>14}",
        "Text Length", "File Size", "Raw Data", "Bytes/Rec", "Throughput"
    );
    println!("{}", "-".repeat(65));
    for &text_len in &[50usize, 200, 1000] {
        let r = measure_footprint(10_000, text_len, false, false);
        let label = match text_len {
            50 => "short (50)",
            200 => "medium (200)",
            _ => "long (1000)",
        };
        println!(
            "{:>12}  {:>12}  {:>12}  {:>10}  {:>11.0} rec/s",
            label,
            fmt_bytes(r.file_bytes),
            fmt_bytes(r.raw_bytes),
            fmt_bytes(r.bytes_per_record()),
            r.records_per_sec()
        );
    }
    println!();

    // --- WAL overhead at 1K records ---
    println!("### WAL Overhead at 1K Records (medium text, no compression)");
    println!();
    let no_wal = measure_footprint(1_000, 200, false, false);
    let with_wal = measure_footprint(1_000, 200, true, false);
    println!(
        "  No WAL:   file={:>10}  ingest={:.1}ms",
        fmt_bytes(no_wal.file_bytes),
        no_wal.ingest_ms
    );
    println!(
        "  With WAL: file={:>10}  WAL={:>8}  ingest={:.1}ms  (+{:.0}% latency)",
        fmt_bytes(with_wal.file_bytes),
        fmt_bytes(with_wal.wal_bytes),
        with_wal.ingest_ms,
        (with_wal.ingest_ms / no_wal.ingest_ms.max(0.001) - 1.0) * 100.0
    );
    println!();

    println!("=================================================================");
    println!("  Summary");
    println!("=================================================================");
    println!();
    println!("At 10K records (typical agent memory), 384-dim embeddings + 200-char text:");
    let r10k = measure_footprint(10_000, 200, false, false);
    let r10k_comp = measure_footprint(10_000, 200, false, true);
    println!("  Uncompressed: {}", fmt_bytes(r10k.file_bytes));
    println!(
        "  Compressed:   {} ({:.1}x ratio)",
        fmt_bytes(r10k_comp.file_bytes),
        r10k_comp.compression_ratio()
    );
    println!(
        "  Per record:   {} (uncompressed)",
        fmt_bytes(r10k.bytes_per_record())
    );
    println!("  Throughput:   {:.0} records/sec", r10k.records_per_sec());
    println!();
    println!("At 100K records:");
    let r100k = measure_footprint(100_000, 200, false, false);
    let r100k_comp = measure_footprint(100_000, 200, false, true);
    println!("  Uncompressed: {}", fmt_bytes(r100k.file_bytes));
    println!(
        "  Compressed:   {} ({:.1}x ratio)",
        fmt_bytes(r100k_comp.file_bytes),
        r100k_comp.compression_ratio()
    );
}

// ---------------------------------------------------------------------------
// Ephemeral tier microbenchmark
// ---------------------------------------------------------------------------

#[cfg(test)]
mod ephemeral_perf {
    // placeholder — actual perf measured in main below
}
