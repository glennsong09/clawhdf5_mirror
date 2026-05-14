use clawhdf5_agent::bm25::BM25Index;
use clawhdf5_agent::cache::MemoryCache;
use clawhdf5_agent::hybrid::hybrid_search;
use clawhdf5_agent::ivf::{IVFIndex, IVFPQIndex};
use clawhdf5_agent::pq::ProductQuantizer;
use clawhdf5_agent::strategy::{self, HardwareCapabilities, SearchStrategy};
use clawhdf5_agent::vector_search;
use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
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

fn make_entry(rng: &mut Rng, dim: usize) -> MemoryEntry {
    MemoryEntry {
        chunk: make_text(rng, 20),
        embedding: make_vec(rng, dim),
        source_channel: "bench".into(),
        timestamp: 1_000_000.0,
        session_id: "bench-session".into(),
        tags: "bench".into(),
    }
}

fn make_entries(n: usize, dim: usize, seed: u32) -> Vec<MemoryEntry> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| make_entry(&mut rng, dim)).collect()
}

fn make_config(dir: &TempDir, dim: usize) -> MemoryConfig {
    MemoryConfig::new(dir.path().join("bench.h5"), "bench-agent", dim)
}

fn nearest_centroid_bench(
    vector: &[f32],
    centroids: &[f32],
    num_clusters: usize,
    dim: usize,
) -> usize {
    let mut best = 0;
    let mut best_sim = f32::NEG_INFINITY;
    let vnorm = vector_search::compute_norm(vector);
    for c in 0..num_clusters {
        let centroid = &centroids[c * dim..(c + 1) * dim];
        let cnorm = vector_search::compute_norm(centroid);
        let sim = clawhdf5_agent::cosine_similarity_prenorm(vector, vnorm, centroid, cnorm);
        if sim > best_sim {
            best_sim = sim;
            best = c;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Save benchmarks
// ---------------------------------------------------------------------------

fn save_benches(c: &mut Criterion) {
    let dim = 384;

    c.bench_function("save_single", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let config = make_config(&dir, dim);
                let mem = HDF5Memory::create(config).unwrap();
                let mut rng = Rng::new(42);
                let entry = make_entry(&mut rng, dim);
                (dir, mem, entry)
            },
            |(_dir, mut mem, entry)| {
                mem.save(entry).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("save_batch_100", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let config = make_config(&dir, dim);
                let mem = HDF5Memory::create(config).unwrap();
                let entries = make_entries(100, dim, 42);
                (dir, mem, entries)
            },
            |(_dir, mut mem, entries)| {
                mem.save_batch(entries).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("save_batch_1000", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let config = make_config(&dir, dim);
                let mem = HDF5Memory::create(config).unwrap();
                let entries = make_entries(1000, dim, 42);
                (dir, mem, entries)
            },
            |(_dir, mut mem, entries)| {
                mem.save_batch(entries).unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

// ---------------------------------------------------------------------------
// Vector search benchmarks (SIMD vs baseline)
// ---------------------------------------------------------------------------

fn vector_search_benches(c: &mut Criterion) {
    let dim = 384;
    let query = make_vec(&mut Rng::new(99), dim);

    {
        let vectors = make_vecs(1_000, dim, 42);
        let tombstones = vec![0u8; 1_000];
        c.bench_function("vector_search_1k", |b| {
            b.iter(|| vector_search::cosine_similarity_batch(&query, &vectors, &tombstones));
        });
    }

    {
        let vectors = make_vecs(10_000, dim, 42);
        let tombstones = vec![0u8; 10_000];
        c.bench_function("simd_cosine_10k", |b| {
            b.iter(|| vector_search::cosine_similarity_batch(&query, &vectors, &tombstones));
        });
    }

    {
        let vectors = make_vecs(100_000, dim, 42);
        let tombstones = vec![0u8; 100_000];
        c.bench_function("simd_cosine_100k", |b| {
            b.iter(|| vector_search::cosine_similarity_batch(&query, &vectors, &tombstones));
        });
    }
}

// ---------------------------------------------------------------------------
// Pre-computed norms benchmark
// ---------------------------------------------------------------------------

fn prenorm_benches(c: &mut Criterion) {
    let dim = 384;
    let n = 10_000;
    let query = make_vec(&mut Rng::new(99), dim);
    let vectors = make_vecs(n, dim, 42);
    let norms: Vec<f32> = vectors
        .iter()
        .map(|v| vector_search::compute_norm(v))
        .collect();
    let tombstones = vec![0u8; n];

    c.bench_function("prenorm_search_10k", |b| {
        b.iter(|| {
            vector_search::cosine_similarity_batch_prenorm(&query, &vectors, &norms, &tombstones)
        });
    });

    c.bench_function("full_norm_search_10k", |b| {
        b.iter(|| vector_search::cosine_similarity_batch(&query, &vectors, &tombstones));
    });
}

// ---------------------------------------------------------------------------
// PQ search benchmark
// ---------------------------------------------------------------------------

fn pq_benches(c: &mut Criterion) {
    let dim = 384;
    let n = 100_000;
    let vectors = make_vecs(n, dim, 42);
    let query = make_vec(&mut Rng::new(99), dim);
    let tombstones = vec![0u8; n];

    // Train PQ on subset for speed
    let train_set: Vec<Vec<f32>> = vectors[..1000].to_vec();
    let pq = ProductQuantizer::train(&train_set, dim, 48, 256);
    let all_codes = pq.encode_all(&vectors);

    c.bench_function("pq_search_100k", |b| {
        b.iter(|| pq.search(&query, &all_codes, &tombstones, 10));
    });
}

// ---------------------------------------------------------------------------
// IVF-PQ search benchmark
// ---------------------------------------------------------------------------

fn ivf_pq_benches(c: &mut Criterion) {
    let dim = 384;
    let n = 100_000;
    let vectors = make_vecs(n, dim, 42);
    let norms: Vec<f32> = vectors
        .iter()
        .map(|v| vector_search::compute_norm(v))
        .collect();
    let query = make_vec(&mut Rng::new(99), dim);
    let tombstones = vec![0u8; n];

    // Train on subset
    let train_set: Vec<Vec<f32>> = vectors[..2000].to_vec();
    let ivf = IVFIndex::train(&train_set, dim, 100);
    // Re-assign all vectors
    let nc = ivf.num_clusters;
    let mut inverted_lists = vec![Vec::new(); nc];
    for (i, v) in vectors.iter().enumerate() {
        let c = nearest_centroid_bench(v, &ivf.centroids, nc, dim);
        inverted_lists[c].push(i);
    }
    let seil_lists = inverted_lists
        .iter()
        .enumerate()
        .map(|(c, list)| {
            list.iter()
                .map(|&idx| clawhdf5_agent::ivf::SharedEntry {
                    vector_idx: idx,
                    primary_list: c,
                })
                .collect()
        })
        .collect();
    let ivf_full = IVFIndex {
        centroids: ivf.centroids.clone(),
        num_clusters: nc,
        dim,
        inverted_lists,
        redundancy_factor: 1,
        seil_lists,
    };

    c.bench_function("ivf_search_100k_nprobe10", |b| {
        b.iter(|| ivf_full.search(&query, &vectors, &norms, &tombstones, 10, 10));
    });

    // IVF-PQ combined
    let pq = ProductQuantizer::train(&train_set, dim, 48, 256);
    let codes = pq.encode_all(&vectors);
    let ivfpq = IVFPQIndex {
        ivf: ivf_full,
        pq,
        codes,
    };

    c.bench_function("ivf_pq_search_100k", |b| {
        b.iter(|| ivfpq.search(&query, &vectors, &norms, &tombstones, 10, 100, 10));
    });
}

// ---------------------------------------------------------------------------
// RAIRS benchmark
// ---------------------------------------------------------------------------

fn rairs_benches(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let vectors = make_vecs(n, dim, 42);
    let norms: Vec<f32> = vectors
        .iter()
        .map(|v| vector_search::compute_norm(v))
        .collect();
    let query = make_vec(&mut Rng::new(99), dim);
    let tombstones = vec![0u8; n];

    // Train RAIRS index with rf=2
    let ivf_rairs = IVFIndex::train_rairs(&vectors, dim, 100, 2);

    c.bench_function("rairs_search_10k_nprobe10", |b| {
        b.iter(|| ivf_rairs.search_rairs(&query, &vectors, &norms, &tombstones, 10, 10));
    });

    // Standard IVF for comparison
    let ivf_std = IVFIndex::train(&vectors, dim, 100);

    c.bench_function("ivf_search_10k_nprobe10", |b| {
        b.iter(|| ivf_std.search(&query, &vectors, &norms, &tombstones, 10, 10));
    });
}

// ---------------------------------------------------------------------------
// BM25 benchmarks
// ---------------------------------------------------------------------------

fn bm25_benches(c: &mut Criterion) {
    {
        let docs = make_texts(1_000, 50, 42);
        let tombstones = vec![0u8; 1_000];
        let index = BM25Index::build(&docs, &tombstones);
        c.bench_function("bm25_search_1k", |b| {
            b.iter(|| index.search("rust programming memory", 10));
        });
    }

    {
        let docs = make_texts(10_000, 50, 42);
        let tombstones = vec![0u8; 10_000];
        let index = BM25Index::build(&docs, &tombstones);
        c.bench_function("bm25_search_10k", |b| {
            b.iter(|| index.search("rust programming memory", 10));
        });
    }
}

// ---------------------------------------------------------------------------
// Hybrid search benchmark
// ---------------------------------------------------------------------------

fn hybrid_benches(c: &mut Criterion) {
    let dim = 384;
    let n = 10_000;
    let vectors = make_vecs(n, dim, 42);
    let docs = make_texts(n, 50, 77);
    let tombstones = vec![0u8; n];
    let bm25 = BM25Index::build(&docs, &tombstones);
    let query_vec = make_vec(&mut Rng::new(99), dim);

    c.bench_function("hybrid_search_10k", |b| {
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
}

// ---------------------------------------------------------------------------
// Compact benchmark
// ---------------------------------------------------------------------------

fn compact_benches(c: &mut Criterion) {
    let dim = 384;
    let n = 10_000;
    let mut rng = Rng::new(42);

    let mut cache = MemoryCache::new(dim);
    for _ in 0..n {
        cache.push(
            make_text(&mut rng, 20),
            make_vec(&mut rng, dim),
            "bench".into(),
            1_000_000.0,
            "session".into(),
            "tag".into(),
        );
    }
    // Tombstone 10%
    for i in (0..n).step_by(10) {
        cache.mark_deleted(i);
    }

    c.bench_function("compact_10pct_tombstoned", |b| {
        b.iter_batched(
            || cache.clone(),
            |mut c| {
                c.compact();
            },
            BatchSize::LargeInput,
        );
    });
}

// ---------------------------------------------------------------------------
// I/O benchmarks (open, snapshot, mmap)
// ---------------------------------------------------------------------------

fn io_benches(_c: &mut Criterion) {
    // Skipped: pre-existing file reopen issue prevents HDF5Memory::open() in bench
}

// ---------------------------------------------------------------------------
// Rayon parallel search benchmarks
// ---------------------------------------------------------------------------

fn rayon_benches(c: &mut Criterion) {
    let dim = 384;
    let query = make_vec(&mut Rng::new(99), dim);

    {
        let n = 10_000;
        let vectors = make_vecs(n, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| vector_search::compute_norm(v))
            .collect();
        let tombstones = vec![0u8; n];

        c.bench_function("rayon_cosine_10k", |b| {
            b.iter(|| {
                use rayon::prelude::*;
                let query_norm = vector_search::compute_norm(&query);
                let num_cores = rayon::current_num_threads().max(1);
                let chunk_size = (n + num_cores - 1) / num_cores;
                let mut results: Vec<(usize, f32)> = vectors
                    .par_chunks(chunk_size)
                    .enumerate()
                    .flat_map(|(ci, chunk)| {
                        let base = ci * chunk_size;
                        chunk
                            .iter()
                            .enumerate()
                            .filter_map(|(j, v)| {
                                let i = base + j;
                                if tombstones[i] != 0 {
                                    return None;
                                }
                                let score = clawhdf5_agent::cosine_similarity_prenorm(
                                    &query, query_norm, v, norms[i],
                                );
                                Some((i, score))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(10);
                results
            });
        });

        c.bench_function("sequential_cosine_10k", |b| {
            b.iter(|| {
                vector_search::cosine_similarity_batch_prenorm(
                    &query,
                    &vectors,
                    &norms,
                    &tombstones,
                )
            });
        });
    }

    {
        let n = 100_000;
        let vectors = make_vecs(n, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| vector_search::compute_norm(v))
            .collect();
        let tombstones = vec![0u8; n];

        c.bench_function("rayon_cosine_100k", |b| {
            b.iter(|| {
                use rayon::prelude::*;
                let query_norm = vector_search::compute_norm(&query);
                let num_cores = rayon::current_num_threads().max(1);
                let chunk_size = (n + num_cores - 1) / num_cores;
                let mut results: Vec<(usize, f32)> = vectors
                    .par_chunks(chunk_size)
                    .enumerate()
                    .flat_map(|(ci, chunk)| {
                        let base = ci * chunk_size;
                        chunk
                            .iter()
                            .enumerate()
                            .filter_map(|(j, v)| {
                                let i = base + j;
                                if tombstones[i] != 0 {
                                    return None;
                                }
                                let score = clawhdf5_agent::cosine_similarity_prenorm(
                                    &query, query_norm, v, norms[i],
                                );
                                Some((i, score))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(10);
                results
            });
        });
    }
}

// ---------------------------------------------------------------------------
// BLAS search benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "fast-math")]
fn blas_benches(c: &mut Criterion) {
    let dim = 384;
    let query = make_vec(&mut Rng::new(99), dim);

    {
        let n = 10_000;
        let vectors = make_vecs(n, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| vector_search::compute_norm(v))
            .collect();
        let tombstones = vec![0u8; n];

        c.bench_function("blas_cosine_10k", |b| {
            b.iter(|| {
                vector_search::cosine_similarity_batch_blas(
                    &query,
                    &vectors,
                    &norms,
                    &tombstones,
                    10,
                )
            });
        });
    }

    {
        let n = 100_000;
        let vectors = make_vecs(n, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| vector_search::compute_norm(v))
            .collect();
        let tombstones = vec![0u8; n];

        c.bench_function("blas_cosine_100k", |b| {
            b.iter(|| {
                vector_search::cosine_similarity_batch_blas(
                    &query,
                    &vectors,
                    &norms,
                    &tombstones,
                    10,
                )
            });
        });
    }

    // Batch norms benchmark
    {
        let n = 10_000;
        let vectors = make_vecs(n, dim, 42);
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        c.bench_function("blas_batch_norms_10k", |b| {
            b.iter(|| clawhdf5_agent::blas_search::blas_batch_norms(&flat, dim));
        });
    }
}

// ---------------------------------------------------------------------------
// Accelerate (cblas_sgemv / AMX) search benchmarks
// ---------------------------------------------------------------------------

#[cfg(any(feature = "accelerate", feature = "openblas"))]
fn accelerate_benches(c: &mut Criterion) {
    let dim = 384;
    let query = make_vec(&mut Rng::new(99), dim);

    {
        let n = 10_000;
        let vectors = make_vecs(n, dim, 42);
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| vector_search::compute_norm(v))
            .collect();
        let tombstones = vec![0u8; n];

        c.bench_function("accelerate_cosine_10k", |b| {
            b.iter(|| {
                clawhdf5_agent::accelerate_search::accelerate_cosine_batch(
                    &query,
                    &flat,
                    &norms,
                    &tombstones,
                    dim,
                    10,
                )
            });
        });

        // Compare: SIMD prenorm on same data
        c.bench_function("simd_prenorm_cosine_10k", |b| {
            b.iter(|| {
                vector_search::cosine_similarity_batch_prenorm(
                    &query,
                    &vectors,
                    &norms,
                    &tombstones,
                )
            });
        });

        // Compare: Accelerate via Vec<Vec> wrapper
        c.bench_function("accelerate_vecs_cosine_10k", |b| {
            b.iter(|| {
                clawhdf5_agent::accelerate_search::accelerate_cosine_batch_vecs(
                    &query,
                    &vectors,
                    &norms,
                    &tombstones,
                    10,
                )
            });
        });
    }

    {
        let n = 100_000;
        let vectors = make_vecs(n, dim, 42);
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| vector_search::compute_norm(v))
            .collect();
        let tombstones = vec![0u8; n];

        c.bench_function("accelerate_cosine_100k", |b| {
            b.iter(|| {
                clawhdf5_agent::accelerate_search::accelerate_cosine_batch(
                    &query,
                    &flat,
                    &norms,
                    &tombstones,
                    dim,
                    10,
                )
            });
        });
    }

    // Batch norms via cblas_snrm2
    {
        let n = 10_000;
        let vectors = make_vecs(n, dim, 42);
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        c.bench_function("accelerate_batch_norms_10k", |b| {
            b.iter(|| clawhdf5_agent::accelerate_search::accelerate_batch_norms(&flat, dim));
        });
    }

    // vDSP comparison (macOS only)
    #[cfg(target_os = "macos")]
    {
        let n = 10_000;
        let vectors = make_vecs(n, dim, 42);
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| vector_search::compute_norm(v))
            .collect();
        let tombstones = vec![0u8; n];

        c.bench_function("vdsp_cosine_10k", |b| {
            b.iter(|| {
                clawhdf5_agent::accelerate_search::vdsp_cosine_batch(
                    &query,
                    &flat,
                    &norms,
                    &tombstones,
                    dim,
                    10,
                )
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Adaptive strategy benchmarks
// ---------------------------------------------------------------------------

fn adaptive_benches(c: &mut Criterion) {
    let dim = 384;
    let n = 10_000;
    let query = make_vec(&mut Rng::new(99), dim);
    let vectors = make_vecs(n, dim, 42);
    let norms: Vec<f32> = vectors
        .iter()
        .map(|v| vector_search::compute_norm(v))
        .collect();
    let tombstones = vec![0u8; n];

    c.bench_function("adaptive_search_10k", |b| {
        let hw = HardwareCapabilities::detect();
        let strat = strategy::auto_select_strategy(n, &hw);
        b.iter(|| {
            strategy::search_with_metrics(&query, &vectors, &norms, &tombstones, 10, strat, None)
        });
    });

    // Strategy comparison: run all CPU strategies on same dataset
    c.bench_function("strategy_scalar_10k", |b| {
        b.iter(|| {
            strategy::search_with_metrics(
                &query,
                &vectors,
                &norms,
                &tombstones,
                10,
                SearchStrategy::Scalar,
                None,
            )
        });
    });

    c.bench_function("strategy_simd_10k", |b| {
        b.iter(|| {
            strategy::search_with_metrics(
                &query,
                &vectors,
                &norms,
                &tombstones,
                10,
                SearchStrategy::SimdBruteForce,
                None,
            )
        });
    });

    c.bench_function("strategy_rayon_10k", |b| {
        b.iter(|| {
            strategy::search_with_metrics(
                &query,
                &vectors,
                &norms,
                &tombstones,
                10,
                SearchStrategy::RayonParallel,
                None,
            )
        });
    });
}

// ---------------------------------------------------------------------------
// Criterion group & main
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// WAL benchmarks — save latency with/without WAL
// ---------------------------------------------------------------------------

fn wal_benches(c: &mut Criterion) {
    let dim = 384;

    c.bench_function("save_with_wal_single", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let mut config = make_config(&dir, dim);
                config.wal_enabled = true;
                let mem = HDF5Memory::create(config).unwrap();
                let mut rng = Rng::new(42);
                let entry = make_entry(&mut rng, dim);
                (dir, mem, entry)
            },
            |(_dir, mut mem, entry)| {
                mem.save(entry).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("save_without_wal_single", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let mut config = make_config(&dir, dim);
                config.wal_enabled = false;
                let mem = HDF5Memory::create(config).unwrap();
                let mut rng = Rng::new(42);
                let entry = make_entry(&mut rng, dim);
                (dir, mem, entry)
            },
            |(_dir, mut mem, entry)| {
                mem.save(entry).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    // WAL save into a store with 1K existing entries
    c.bench_function("save_wal_1k_existing", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let mut config = make_config(&dir, dim);
                config.wal_enabled = true;
                config.wal_max_entries = 5000;
                let mut mem = HDF5Memory::create(config).unwrap();
                let entries = make_entries(1000, dim, 99);
                mem.save_batch(entries).unwrap();
                let mut rng = Rng::new(42);
                let entry = make_entry(&mut rng, dim);
                (dir, mem, entry)
            },
            |(_dir, mut mem, entry)| {
                mem.save(entry).unwrap();
            },
            BatchSize::LargeInput,
        );
    });

    c.bench_function("wal_flush_100_entries", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let mut config = make_config(&dir, dim);
                config.wal_enabled = true;
                config.wal_max_entries = 5000;
                let mut mem = HDF5Memory::create(config).unwrap();
                let entries = make_entries(100, dim, 77);
                for e in entries {
                    mem.save(e).unwrap();
                }
                (dir, mem)
            },
            |(_dir, mut mem)| {
                mem.flush_wal().unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

// ---------------------------------------------------------------------------
// Decision Gate benchmarks
// ---------------------------------------------------------------------------

fn gate_benches(c: &mut Criterion) {
    use clawhdf5_agent::decision_gate::{DecisionGate, GateConfig};

    let gate = DecisionGate::new(GateConfig::default());

    c.bench_function("gate_trivial_skip", |b| {
        b.iter(|| gate.should_save("ok"));
    });

    c.bench_function("gate_nontrivial_pass", |b| {
        b.iter(|| gate.should_save("Tell me about the deployment architecture for the new system"));
    });

    c.bench_function("gate_short_phrase_skip", |b| {
        b.iter(|| gate.should_save("got it"));
    });

    c.bench_function("gate_ratio_check", |b| {
        b.iter(|| gate.should_save("yes yes definitely sure absolutely right"));
    });
}

// ---------------------------------------------------------------------------
// Hebbian activation benchmarks
// ---------------------------------------------------------------------------

fn hebbian_benches(c: &mut Criterion) {
    let dim = 384;

    c.bench_function("tick_session_1k", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let config = make_config(&dir, dim);
                let mut mem = HDF5Memory::create(config).unwrap();
                let entries = make_entries(1000, dim, 55);
                mem.save_batch(entries).unwrap();
                (dir, mem)
            },
            |(_dir, mut mem)| {
                mem.tick_session().unwrap();
            },
            BatchSize::LargeInput,
        );
    });

    c.bench_function("tick_session_10k", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let config = make_config(&dir, dim);
                let mut mem = HDF5Memory::create(config).unwrap();
                let entries = make_entries(10000, dim, 55);
                mem.save_batch(entries).unwrap();
                (dir, mem)
            },
            |(_dir, mut mem)| {
                mem.tick_session().unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

// ---------------------------------------------------------------------------
// Entity alias resolution benchmarks
// ---------------------------------------------------------------------------

fn alias_benches(c: &mut Criterion) {
    use clawhdf5_agent::knowledge::KnowledgeCache;

    // Build a knowledge cache with 50 entities and 100 aliases
    let mut kg = KnowledgeCache::new();
    let names = [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eve",
        "Frank",
        "Grace",
        "Henry",
        "Irene",
        "Jack",
        "Karen",
        "Leo",
        "Mona",
        "Nick",
        "Olivia",
        "Peter",
        "Quinn",
        "Rachel",
        "Sam",
        "Tina",
        "Uma",
        "Victor",
        "Wendy",
        "Xander",
        "Yara",
        "Zack",
        "Acme Corp",
        "BigCo",
        "CloudNet",
        "DataSys",
        "EdgeAI",
        "FinTech",
        "GlobalX",
        "HyperNet",
        "InfoSec",
        "JetOps",
        "KernelDev",
        "LogicAI",
        "MeshNet",
        "NanoTech",
        "OpenStack",
        "PlatformX",
        "QuantumAI",
        "RoboSys",
        "SkyNet",
        "TechFlow",
        "UniCloud",
        "VirtNet",
        "WaveAI",
        "XenoData",
    ];
    for (i, name) in names.iter().enumerate() {
        let id = kg.add_entity(name, "entity", -1);
        kg.add_alias(&format!("alias_{i}"), id as i64);
        kg.add_alias(&name.to_lowercase(), id as i64);
    }

    c.bench_function("alias_resolve_short_query", |b| {
        b.iter(|| kg.resolve_aliases("what does alias_7 do at alias_30?"));
    });

    c.bench_function("alias_resolve_no_match", |b| {
        b.iter(|| kg.resolve_aliases("tell me about the weather forecast for tomorrow"));
    });

    c.bench_function("alias_resolve_long_query", |b| {
        b.iter(|| kg.resolve_aliases("I need alias_0 to talk to alias_15 about the alias_27 project and also check with alias_42"));
    });
}

// ---------------------------------------------------------------------------
// MemoryStrategy benchmarks
// ---------------------------------------------------------------------------

fn strategy_benches(c: &mut Criterion) {
    use clawhdf5_agent::SearchResult;
    use clawhdf5_agent::memory_strategy::*;

    struct EmptyStore;
    impl MemoryStoreView for EmptyStore {
        fn search(&self, _: &[f32], _: usize) -> Vec<SearchResult> {
            vec![]
        }
        fn memory_count(&self) -> usize {
            0
        }
        fn entity_count(&self) -> usize {
            0
        }
    }

    let save_every = SaveEveryExchange::default();
    let shift = SaveOnSemanticShift {
        gate: clawhdf5_agent::decision_gate::DecisionGate::new(
            clawhdf5_agent::decision_gate::GateConfig::default(),
        ),
        shift_threshold: 0.25,
        lookback_k: 5,
    };

    let exchange = Exchange {
        user_turn: "What is the deployment architecture for the new microservices?".into(),
        agent_turn:
            "The deployment uses Kubernetes with a service mesh for inter-service communication."
                .into(),
        session_id: "bench-session".into(),
        turn_number: 1,
        timestamp: 1_000_000.0,
        user_embedding: Some(vec![0.1; 384]),
        agent_embedding: Some(vec![0.2; 384]),
    };

    let trivial_exchange = Exchange {
        user_turn: "ok".into(),
        agent_turn: "Got it!".into(),
        session_id: "bench-session".into(),
        turn_number: 2,
        timestamp: 1_000_001.0,
        user_embedding: None,
        agent_embedding: None,
    };

    let store = EmptyStore;

    c.bench_function("strategy_save_every_substantive", |b| {
        b.iter(|| save_every.evaluate(&exchange, &store));
    });

    c.bench_function("strategy_save_every_trivial", |b| {
        b.iter(|| save_every.evaluate(&trivial_exchange, &store));
    });

    c.bench_function("strategy_semantic_shift_empty_store", |b| {
        b.iter(|| shift.evaluate(&exchange, &store));
    });
}

// ---------------------------------------------------------------------------
// AGENTS.md generation benchmark
// ---------------------------------------------------------------------------

fn agents_md_benches(c: &mut Criterion) {
    let dim = 384;

    c.bench_function("agents_md_generate_1k", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let config = make_config(&dir, dim);
                let mut mem = HDF5Memory::create(config).unwrap();
                let entries = make_entries(1000, dim, 42);
                mem.save_batch(entries).unwrap();
                mem.add_entity("Alice", "person", -1).unwrap();
                mem.add_entity("ProjectX", "project", -1).unwrap();
                (dir, mem)
            },
            |(_dir, mem)| {
                mem.generate_agents_md();
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    benches,
    save_benches,
    vector_search_benches,
    prenorm_benches,
    pq_benches,
    ivf_pq_benches,
    rairs_benches,
    bm25_benches,
    hybrid_benches,
    compact_benches,
    io_benches,
    rayon_benches,
    adaptive_benches,
    wal_benches,
    gate_benches,
    hebbian_benches,
    alias_benches,
    strategy_benches,
    agents_md_benches,
);

#[cfg(feature = "fast-math")]
criterion_group!(blas_bench_group, blas_benches);

#[cfg(any(feature = "accelerate", feature = "openblas"))]
criterion_group!(accelerate_bench_group, accelerate_benches);

// Criterion main: include all available bench groups
#[cfg(all(
    feature = "fast-math",
    any(feature = "accelerate", feature = "openblas")
))]
criterion_main!(benches, blas_bench_group, accelerate_bench_group);

#[cfg(all(
    feature = "fast-math",
    not(any(feature = "accelerate", feature = "openblas"))
))]
criterion_main!(benches, blas_bench_group);

#[cfg(all(
    not(feature = "fast-math"),
    any(feature = "accelerate", feature = "openblas")
))]
criterion_main!(benches, accelerate_bench_group);

#[cfg(all(
    not(feature = "fast-math"),
    not(any(feature = "accelerate", feature = "openblas"))
))]
criterion_main!(benches);
