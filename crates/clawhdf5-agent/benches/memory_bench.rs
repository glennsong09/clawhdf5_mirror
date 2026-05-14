use clawhdf5_agent::bm25::BM25Index;
use clawhdf5_agent::consolidation::{
    ConsolidationConfig, ConsolidationEngine, ImportanceScorer, ImportanceWeights, MemorySource,
};
use clawhdf5_agent::hybrid::{hybrid_search, rrf_hybrid_search};
use clawhdf5_agent::knowledge::KnowledgeCache;
use clawhdf5_agent::temporal::TemporalIndex;
use clawhdf5_agent::vector_search;
use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Simple deterministic PRNG (LCG)
// ---------------------------------------------------------------------------

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        self.0 >> 16
    }
    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / 65536.0 - 0.5
    }
    fn next_usize(&mut self, max: usize) -> usize {
        self.next_u32() as usize % max
    }
}

// ---------------------------------------------------------------------------
// Data generation helpers
// ---------------------------------------------------------------------------

const WORDS: &[&str] = &[
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "rust",
    "programming",
    "memory",
    "vector",
    "search",
    "index",
    "data",
    "system",
    "agent",
    "knowledge",
    "graph",
    "neural",
    "network",
    "machine",
    "learning",
    "deep",
    "embedding",
    "cosine",
    "similarity",
    "token",
    "chunk",
    "session",
    "channel",
    "hybrid",
    "temporal",
    "consolidation",
    "episodic",
    "semantic",
];

fn make_vec(rng: &mut Rng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.next_f32()).collect()
}

fn make_vecs(n: usize, dim: usize, seed: u32) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| make_vec(&mut rng, dim)).collect()
}

fn make_text(rng: &mut Rng, word_count: usize) -> String {
    (0..word_count)
        .map(|_| WORDS[rng.next_usize(WORDS.len())])
        .collect::<Vec<_>>()
        .join(" ")
}

fn make_texts(n: usize, word_count: usize, seed: u32) -> Vec<String> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| make_text(&mut rng, word_count)).collect()
}

// ---------------------------------------------------------------------------
// Vector search latency benchmarks
// ---------------------------------------------------------------------------

fn vector_search_latency(c: &mut Criterion) {
    const DIM: usize = 384;
    let query = make_vec(&mut Rng::new(99), DIM);

    let mut group = c.benchmark_group("vector_search_latency");
    group.sample_size(50);

    for (label, n) in [("1k", 1_000usize), ("10k", 10_000), ("100k", 100_000)] {
        let vectors = make_vecs(n, DIM, 42);
        let tombstones = vec![0u8; n];

        group.bench_with_input(
            BenchmarkId::new("bench_cosine_search", label),
            &n,
            |b, _| {
                b.iter(|| vector_search::cosine_similarity_batch(&query, &vectors, &tombstones));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Hybrid search benchmarks
// ---------------------------------------------------------------------------

fn hybrid_search_benches(c: &mut Criterion) {
    const DIM: usize = 384;
    const N: usize = 1_000;

    let vectors = make_vecs(N, DIM, 42);
    let docs = make_texts(N, 50, 77);
    let tombstones = vec![0u8; N];
    let bm25 = BM25Index::build(&docs, &tombstones);
    let query_vec = make_vec(&mut Rng::new(99), DIM);

    let mut group = c.benchmark_group("hybrid_search");
    group.sample_size(50);

    // Weighted score fusion (vector + BM25)
    group.bench_function("bench_hybrid_search_1k", |b| {
        b.iter(|| {
            hybrid_search(
                &query_vec,
                "rust memory search",
                &vectors,
                &docs,
                &tombstones,
                &bm25,
                0.7,
                0.3,
                10,
            )
        });
    });

    // Reciprocal Rank Fusion
    group.bench_function("bench_rrf_search_1k", |b| {
        b.iter(|| {
            rrf_hybrid_search(
                &query_vec,
                "rust memory search",
                &vectors,
                &docs,
                &tombstones,
                &bm25,
                10,
            )
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Knowledge graph benchmarks
// ---------------------------------------------------------------------------

fn build_knowledge_graph(n: usize) -> KnowledgeCache {
    let mut kg = KnowledgeCache::new();
    // Add n entities
    for i in 0..n {
        kg.add_entity(&format!("entity_{i}"), "node", -1);
    }
    // Add edges: each node connects to next 3 nodes (ring-like)
    for i in 0..n {
        let src = i as u64;
        let tgt1 = ((i + 1) % n) as u64;
        let tgt2 = ((i + 2) % n) as u64;
        let tgt3 = ((i + 3) % n) as u64;
        kg.add_relation(src, tgt1, "connects", 1.0);
        kg.add_relation(src, tgt2, "relates", 0.8);
        kg.add_relation(src, tgt3, "associated", 0.6);
    }
    kg
}

fn knowledge_graph_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("knowledge_graph");
    group.sample_size(50);

    // BFS traversal benchmarks
    {
        let kg_100 = build_knowledge_graph(100);
        group.bench_function("bench_bfs_100_entities", |b| {
            b.iter(|| kg_100.bfs_neighbors(0, 3));
        });
    }

    {
        let kg_1000 = build_knowledge_graph(1_000);
        group.bench_function("bench_bfs_1000_entities", |b| {
            b.iter(|| kg_1000.bfs_neighbors(0, 3));
        });
    }

    // Spreading activation benchmark
    {
        let kg_100 = build_knowledge_graph(100);
        let seed_ids: Vec<u64> = vec![0, 1, 2];
        group.bench_function("bench_spreading_activation_100", |b| {
            b.iter(|| kg_100.spreading_activation(&seed_ids, 0.85, 0.01, 5));
        });
    }

    // Fuzzy entity resolution benchmark
    {
        // Build a graph with 100 entities, then benchmark fuzzy matching
        let mut kg_100 = build_knowledge_graph(100);
        // Pre-populate with some named entities
        for i in 0..100usize {
            kg_100.add_alias(&format!("alias_{i}"), i as i64);
        }

        group.bench_function("bench_entity_resolution_100", |b| {
            // Queries with slight typos to trigger fuzzy matching
            let queries = [
                ("entty_42", "node"),
                ("entitty_7", "node"),
                ("entity_99x", "node"),
                ("enttiy_50", "node"),
            ];
            let mut idx = 0usize;
            b.iter(|| {
                let (name, etype) = queries[idx % queries.len()];
                idx += 1;
                // Clone needed since resolve_or_create takes &mut self
                let mut kg = kg_100.clone();
                kg.resolve_or_create(name, etype, -1, 3)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Consolidation benchmarks
// ---------------------------------------------------------------------------

fn consolidation_benches(c: &mut Criterion) {
    const DIM: usize = 384;

    let mut group = c.benchmark_group("consolidation");
    group.sample_size(50);

    // Consolidation cycle benchmarks
    for (label, n) in [("100", 100usize), ("1000", 1_000)] {
        group.bench_with_input(
            BenchmarkId::new("bench_consolidation_cycle", label),
            &n,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut engine = ConsolidationEngine::new(ConsolidationConfig {
                            working_capacity: n + 50,
                            episodic_capacity: n * 20,
                            ..ConsolidationConfig::default()
                        });
                        let mut rng = Rng::new(42);
                        let now = 1_000_000.0f64;
                        for i in 0..n {
                            let embedding = make_vec(&mut rng, DIM);
                            let chunk = format!("memory record {i} with some content");
                            engine.add_memory(chunk, embedding, MemorySource::User, now + i as f64);
                        }
                        engine
                    },
                    |mut engine| {
                        engine.consolidate(2_000_000.0);
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    // Importance scoring benchmark
    {
        let mut rng = Rng::new(55);
        // Build a set of existing records for surprise scoring
        let mut engine = ConsolidationEngine::new(ConsolidationConfig::default());
        let now = 1_000_000.0f64;
        for i in 0..50usize {
            let embedding = make_vec(&mut rng, DIM);
            let chunk = format!("existing record {i}");
            engine.add_memory(chunk, embedding, MemorySource::User, now + i as f64);
        }
        let records = engine.records().to_vec();
        let weights = ImportanceWeights::default();
        let query_embedding = make_vec(&mut rng, DIM);
        let sample_text =
            "This is a substantive memory about system architecture and deployment patterns";

        group.bench_function("bench_importance_scoring", |b| {
            b.iter(|| {
                let surprise = ImportanceScorer::score_surprise(&query_embedding, &records);
                let correction = ImportanceScorer::score_correction(&MemorySource::Correction);
                let length = ImportanceScorer::score_length(sample_text);
                ImportanceScorer::score_combined(surprise, correction, length, &weights)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Temporal index benchmarks
// ---------------------------------------------------------------------------

fn temporal_benches(c: &mut Criterion) {
    const N: usize = 10_000;

    let mut group = c.benchmark_group("temporal");
    group.sample_size(50);

    // Build a pre-populated temporal index for range queries
    let mut index = TemporalIndex::new();
    for i in 0..N {
        index.insert(i as u64, i as f64 * 10.0);
    }
    // Query middle third
    let start_ts = (N as f64 * 10.0) / 3.0;
    let end_ts = (N as f64 * 10.0) * 2.0 / 3.0;

    group.bench_function("bench_temporal_range_query_10k", |b| {
        b.iter(|| index.range_query(start_ts, end_ts));
    });

    // Insert benchmark: measure time to insert 10k timestamps one by one
    group.bench_function("bench_temporal_insert_10k", |b| {
        b.iter_batched(
            || TemporalIndex::new(),
            |mut idx| {
                for i in 0..N {
                    // Shuffle insertion order slightly using a simple offset pattern
                    let ts = ((i * 7) % N) as f64 * 10.0;
                    idx.insert(i as u64, ts);
                }
                idx
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// HDF5 write-scale benchmarks (Track 8.4 — memory footprint)
// ---------------------------------------------------------------------------
//
// Measures batch write throughput at 100, 1K, and 10K records.
// Actual file sizes are reported by the standalone `footprint_bench` binary.
//
// # WASM Note (cfg target_arch = "wasm32")
// These benchmarks require std filesystem access (HDF5 on-disk format).
// For wasm32 targets:
//   - HDF5Memory would need an in-memory or IndexedDB-backed storage layer.
//   - `TempDir` would be replaced by a virtual FS.
//   - `criterion` is not available on wasm32; use `console_error_panic_hook`
//     + manual timing via `web_sys::Performance` instead.
// The #[cfg(target_arch = "wasm32")] guard is not applied here because the
// entire bench harness is excluded from wasm32 builds by the `harness = false`
// Cargo configuration.

fn make_bench_entry(idx: usize, dim: usize) -> MemoryEntry {
    let mut rng = Rng::new(idx as u32 + 7777);
    MemoryEntry {
        chunk: make_text(&mut rng, 30),
        embedding: make_vec(&mut rng, dim),
        source_channel: "footprint-bench".to_string(),
        timestamp: 1_000_000.0 + idx as f64,
        session_id: format!("sess_{}", idx / 50),
        tags: String::new(),
    }
}

fn hdf5_write_scale_benches(c: &mut Criterion) {
    const DIM: usize = 384;

    let mut group = c.benchmark_group("hdf5_write_scale");
    group.sample_size(10); // fewer samples — these involve disk I/O

    for (label, n) in [("100", 100usize), ("1k", 1_000), ("10k", 10_000)] {
        let entries: Vec<MemoryEntry> = (0..n).map(|i| make_bench_entry(i, DIM)).collect();

        group.bench_with_input(BenchmarkId::new("batch_write", label), &n, |b, _| {
            b.iter_batched(
                || {
                    let dir = TempDir::new().expect("TempDir");
                    let mut cfg = MemoryConfig::new(dir.path().join("w.h5"), "bench", DIM);
                    cfg.wal_enabled = false;
                    cfg.compact_threshold = 0.0;
                    let mem = HDF5Memory::create(cfg).expect("HDF5Memory");
                    (dir, mem, entries.clone())
                },
                |(dir, mut mem, e)| {
                    mem.save_batch(e).expect("save_batch");
                    // Keep dir alive so the file isn't deleted during measurement
                    std::hint::black_box(dir);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Large-scale consolidation benchmarks (Track 8.5 extension)
// ---------------------------------------------------------------------------

fn large_consolidation_benches(c: &mut Criterion) {
    const DIM: usize = 384;

    let mut group = c.benchmark_group("consolidation_large");
    group.sample_size(10);

    for (label, n) in [("10k", 10_000usize)] {
        group.bench_with_input(
            BenchmarkId::new("bench_consolidation_cycle", label),
            &n,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut engine = ConsolidationEngine::new(ConsolidationConfig {
                            working_capacity: n / 2,
                            episodic_capacity: n * 20,
                            ..ConsolidationConfig::default()
                        });
                        let mut rng = Rng::new(99);
                        let now = 1_000_000.0f64;
                        for i in 0..n {
                            let embedding = make_vec(&mut rng, DIM);
                            let chunk = format!("memory record {i} with content");
                            engine.add_memory(chunk, embedding, MemorySource::User, now + i as f64);
                        }
                        engine
                    },
                    |mut engine| {
                        engine.consolidate(2_000_000.0);
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups & main
// ---------------------------------------------------------------------------

criterion_group!(
    memory_benches,
    vector_search_latency,
    hybrid_search_benches,
    knowledge_graph_benches,
    consolidation_benches,
    large_consolidation_benches,
    temporal_benches,
    hdf5_write_scale_benches,
);

criterion_main!(memory_benches);
