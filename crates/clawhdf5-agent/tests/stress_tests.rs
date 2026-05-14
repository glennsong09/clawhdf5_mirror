use clawhdf5_agent::bm25::BM25Index;
use clawhdf5_agent::hybrid::hybrid_search;
use clawhdf5_agent::vector_search;
use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};
use std::path::Path;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
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

fn make_vec(rng: &mut Rng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.next_f32()).collect()
}

fn make_config(dir: &TempDir, dim: usize) -> MemoryConfig {
    MemoryConfig::new(dir.path().join("test.h5"), "stress-agent", dim)
}

fn make_entry_simple(i: usize, dim: usize) -> MemoryEntry {
    MemoryEntry {
        chunk: format!("chunk_{i}"),
        embedding: vec![i as f32 * 0.001; dim],
        source_channel: "stress".into(),
        timestamp: i as f64,
        session_id: format!("sess_{}", i / 100),
        tags: String::new(),
    }
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// 1. 100K entries in batches of 1000
// ---------------------------------------------------------------------------

#[test]
fn test_100k_entries() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let mut mem = HDF5Memory::create(config).unwrap();

    for batch in 0..100 {
        let start = batch * 1000;
        let entries: Vec<MemoryEntry> = (start..start + 1000)
            .map(|i| make_entry_simple(i, 4))
            .collect();
        mem.save_batch(entries).unwrap();
    }

    assert_eq!(mem.count(), 100_000);
    assert_eq!(mem.count_active(), 100_000);
}

// ---------------------------------------------------------------------------
// 2. Heavy tombstoning: save 10K, delete 5K, compact, verify 5K remain
// ---------------------------------------------------------------------------

#[test]
fn test_heavy_tombstoning() {
    let dir = TempDir::new().unwrap();
    let mut config = make_config(&dir, 4);
    config.compact_threshold = 0.0; // disable auto-compact
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    let entries: Vec<MemoryEntry> = (0..10_000).map(|i| make_entry_simple(i, 4)).collect();
    mem.save_batch(entries).unwrap();
    assert_eq!(mem.count(), 10_000);

    // Delete every other entry (5000 deletions)
    let mut rng = Rng::new(42);
    let mut deleted = std::collections::HashSet::new();
    while deleted.len() < 5000 {
        let idx = rng.next_usize(10_000);
        if deleted.insert(idx) {
            mem.delete(idx).unwrap();
        }
    }

    assert_eq!(mem.count_active(), 5000);
    let removed = mem.compact().unwrap();
    assert_eq!(removed, 5000);
    assert_eq!(mem.count(), 5000);
    assert_eq!(mem.count_active(), 5000);

    // Verify persistence
    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.count(), 5000);
}

// ---------------------------------------------------------------------------
// 3. Repeated open/close cycles
// ---------------------------------------------------------------------------

#[test]
fn test_repeated_open_close() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();

    {
        let mut mem = HDF5Memory::create(config).unwrap();
        let entries: Vec<MemoryEntry> = (0..100).map(|i| make_entry_simple(i, 4)).collect();
        mem.save_batch(entries).unwrap();
    }

    {
        let mut mem = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem.count(), 100);
        let entries: Vec<MemoryEntry> = (100..200).map(|i| make_entry_simple(i, 4)).collect();
        mem.save_batch(entries).unwrap();
    }

    let mem = HDF5Memory::open(&path).unwrap();
    assert_eq!(mem.count(), 200);
    assert_eq!(mem.count_active(), 200);
}

// ---------------------------------------------------------------------------
// 4. Large embeddings (1536-dim, ada-002 size) x 10K
// ---------------------------------------------------------------------------

#[test]
fn test_large_embeddings_1536() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 1536);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    let mut rng = Rng::new(42);
    let entries: Vec<MemoryEntry> = (0..10_000)
        .map(|i| MemoryEntry {
            chunk: format!("large_emb_{i}"),
            embedding: make_vec(&mut rng, 1536),
            source_channel: "test".into(),
            timestamp: i as f64,
            session_id: "s1".into(),
            tags: String::new(),
        })
        .collect();
    mem.save_batch(entries).unwrap();
    assert_eq!(mem.count(), 10_000);

    // Verify persistence
    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.count(), 10_000);

    // Verify search works on large dims
    let (_, cache, _, _) = clawhdf5_agent::storage::read_from_disk(&path).unwrap();
    let query = make_vec(&mut Rng::new(99), 1536);
    let results =
        vector_search::cosine_similarity_batch(&query, &cache.embeddings, &cache.tombstones);
    assert_eq!(results.len(), 10_000);
}

// ---------------------------------------------------------------------------
// 5. Concurrent-like pattern: interleaved save, search, delete, compact
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_like_pattern() {
    let dir = TempDir::new().unwrap();
    let mut config = make_config(&dir, 8);
    config.compact_threshold = 0.0;
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();
    let mut rng = Rng::new(42);

    // Round 1: save 100 entries
    for i in 0..100 {
        mem.save(MemoryEntry {
            chunk: format!("round1_{i}"),
            embedding: make_vec(&mut rng, 8),
            source_channel: "test".into(),
            timestamp: i as f64,
            session_id: "s1".into(),
            tags: String::new(),
        })
        .unwrap();
    }
    assert_eq!(mem.count(), 100);

    // Flush WAL so read_from_disk sees the data
    mem.flush_wal().unwrap();

    // Read cache for search
    let (_, cache, _, _) = clawhdf5_agent::storage::read_from_disk(&path).unwrap();
    let query = make_vec(&mut Rng::new(99), 8);
    let results =
        vector_search::cosine_similarity_batch(&query, &cache.embeddings, &cache.tombstones);
    assert_eq!(results.len(), 100);

    // Delete 20 entries
    for i in 0..20 {
        mem.delete(i).unwrap();
    }
    assert_eq!(mem.count_active(), 80);

    // Compact
    let removed = mem.compact().unwrap();
    assert_eq!(removed, 20);

    // Round 2: save 50 more
    for i in 0..50 {
        mem.save(MemoryEntry {
            chunk: format!("round2_{i}"),
            embedding: make_vec(&mut rng, 8),
            source_channel: "test".into(),
            timestamp: 200.0 + i as f64,
            session_id: "s2".into(),
            tags: String::new(),
        })
        .unwrap();
    }
    assert_eq!(mem.count_active(), 130);

    // Flush WAL so read_from_disk sees round 2 entries
    mem.flush_wal().unwrap();

    // Final search
    let (_, cache2, _, _) = clawhdf5_agent::storage::read_from_disk(&path).unwrap();
    let results2 =
        vector_search::cosine_similarity_batch(&query, &cache2.embeddings, &cache2.tombstones);
    assert_eq!(results2.len(), 130);
}

// ---------------------------------------------------------------------------
// 6. File size growth is reasonable
// ---------------------------------------------------------------------------

#[test]
fn test_file_size_growth() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 8);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    let mut sizes: Vec<u64> = Vec::new();
    for batch in 0..10 {
        let start = batch * 500;
        let entries: Vec<MemoryEntry> = (start..start + 500)
            .map(|i| make_entry_simple(i, 8))
            .collect();
        mem.save_batch(entries).unwrap();
        sizes.push(file_size(&path));
    }

    // File size should grow roughly linearly (not exponentially)
    // Check that doubling entries doesn't more than triple file size
    for i in 1..sizes.len() {
        assert!(
            sizes[i] > sizes[i - 1],
            "file should grow: sizes[{i}]={} <= sizes[{}]={}",
            sizes[i],
            i - 1,
            sizes[i - 1]
        );
    }

    // The 10x data file should be less than 15x the 1x data file
    let ratio = sizes[9] as f64 / sizes[0] as f64;
    assert!(
        ratio < 15.0,
        "file growth ratio too high: {ratio:.1}x for 10x data"
    );
}

// ---------------------------------------------------------------------------
// 7. Vector search accuracy with known vectors
// ---------------------------------------------------------------------------

#[test]
fn test_vector_search_accuracy() {
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0],                                   // idx 0: unit x
        vec![0.0, 1.0, 0.0, 0.0],                                   // idx 1: unit y
        vec![0.0, 0.0, 1.0, 0.0],                                   // idx 2: unit z
        vec![1.0 / 2.0_f32.sqrt(), 1.0 / 2.0_f32.sqrt(), 0.0, 0.0], // idx 3: 45 deg
        vec![-1.0, 0.0, 0.0, 0.0],                                  // idx 4: opposite x
    ];
    let tombstones = vec![0u8; 5];
    let query = vec![1.0, 0.0, 0.0, 0.0]; // unit x

    let results = vector_search::cosine_similarity_batch(&query, &vectors, &tombstones);

    // Expected order: idx0 (1.0) > idx3 (~0.707) > idx1 (0.0) = idx2 (0.0) > idx4 (-1.0)
    assert_eq!(results[0].0, 0);
    assert!((results[0].1 - 1.0).abs() < 1e-5);

    assert_eq!(results[1].0, 3);
    assert!((results[1].1 - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);

    // idx4 should be last with cos = -1.0
    assert_eq!(results[4].0, 4);
    assert!((results[4].1 - (-1.0)).abs() < 1e-5);

    // Verify cosine_similarity standalone
    let sim = vector_search::cosine_similarity(&query, &vectors[3]);
    assert!((sim - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// 8. BM25 accuracy with known term frequencies
// ---------------------------------------------------------------------------

#[test]
fn test_bm25_accuracy() {
    let docs = vec![
        "rust rust rust systems programming".to_string(), // idx 0: 3x "rust"
        "rust programming language".to_string(),          // idx 1: 1x "rust"
        "python scripting language".to_string(),          // idx 2: 0x "rust"
        "java enterprise rust system".to_string(),        // idx 3: 1x "rust"
        "javascript web development frontend".to_string(), // idx 4: 0x "rust"
    ];
    let tombstones = vec![0u8; 5];
    let index = BM25Index::build(&docs, &tombstones);

    // Query for "rust"
    let results = index.search("rust", 10);

    // Doc 0 should rank first (highest TF for "rust")
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 0, "doc with 3x 'rust' should rank first");

    // Docs 2 and 4 should not appear (no "rust")
    let result_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();
    assert!(
        !result_ids.contains(&2),
        "doc without 'rust' should not appear"
    );
    assert!(
        !result_ids.contains(&4),
        "doc without 'rust' should not appear"
    );

    // Query for rare term "enterprise"
    let rare_results = index.search("enterprise", 10);
    assert_eq!(rare_results.len(), 1);
    assert_eq!(rare_results[0].0, 3);

    // Multi-term query: "rust programming" should boost doc 0 and 1
    let multi = index.search("rust programming", 10);
    assert!(multi.len() >= 2);
    let top2: Vec<usize> = multi.iter().take(2).map(|(id, _)| *id).collect();
    assert!(
        top2.contains(&0),
        "doc 0 should be in top 2 for 'rust programming'"
    );
    assert!(
        top2.contains(&1),
        "doc 1 should be in top 2 for 'rust programming'"
    );
}

// ---------------------------------------------------------------------------
// 9. Hybrid search correctness: vector and keyword disagree
// ---------------------------------------------------------------------------

#[test]
fn test_hybrid_search_correctness() {
    // Doc 0: great vector match, no keyword match
    // Doc 1: no vector match, great keyword match
    // Doc 2: moderate vector match, moderate keyword match
    // Doc 3: some vector, some keyword
    // Doc 4: filler
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0], // idx 0: identical to query
        vec![0.0, 1.0, 0.0, 0.0], // idx 1: orthogonal
        vec![0.7, 0.7, 0.0, 0.0], // idx 2: partial match
        vec![0.3, 0.3, 0.3, 0.3], // idx 3: mild match
        vec![0.0, 0.0, 0.0, 1.0], // idx 4: orthogonal
    ];
    let chunks = vec![
        "gamma delta epsilon phi".to_string(), // 0: no keyword match
        "alpha alpha alpha beta alpha".to_string(), // 1: heavy keyword
        "alpha gamma delta".to_string(),       // 2: some keyword
        "alpha beta gamma".to_string(),        // 3: some keyword
        "alpha omega sigma".to_string(),       // 4: some keyword
    ];
    let tombstones = vec![0u8; 5];
    let bm25 = BM25Index::build(&chunks, &tombstones);
    let query_emb = vec![1.0, 0.0, 0.0, 0.0];

    // Vector-only: doc 0 should win
    let vec_only = hybrid_search(
        &query_emb,
        "alpha",
        &vectors,
        &chunks,
        &tombstones,
        &bm25,
        1.0,
        0.0,
        5,
    );
    assert_eq!(vec_only[0].0, 0, "vector-only: doc 0 should win");

    // Keyword-only: doc 1 should win (most "alpha" occurrences)
    let kw_only = hybrid_search(
        &query_emb,
        "alpha",
        &vectors,
        &chunks,
        &tombstones,
        &bm25,
        0.0,
        1.0,
        5,
    );
    assert_eq!(kw_only[0].0, 1, "keyword-only: doc 1 should win");

    // Balanced: both doc 0 and doc 1 should appear in top 3
    let balanced = hybrid_search(
        &query_emb,
        "alpha",
        &vectors,
        &chunks,
        &tombstones,
        &bm25,
        0.5,
        0.5,
        5,
    );
    let top3: Vec<usize> = balanced.iter().take(3).map(|(id, _)| *id).collect();
    assert!(top3.contains(&0), "balanced: doc 0 should be in top 3");
    assert!(top3.contains(&1), "balanced: doc 1 should be in top 3");
}

// ---------------------------------------------------------------------------
// 10. Session tracking stress: 1000 sessions
// ---------------------------------------------------------------------------

#[test]
fn test_session_tracking_stress() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();

    {
        let mut mem = HDF5Memory::create(config).unwrap();
        for i in 0..1000 {
            mem.add_session(
                &format!("sess_{i}"),
                i * 10,
                (i + 1) * 10,
                "api",
                &format!("Summary for session {i}"),
            )
            .unwrap();
        }
    }

    // Reopen and verify random sessions
    let mem = HDF5Memory::open(&path).unwrap();
    let mut rng = Rng::new(42);
    for _ in 0..100 {
        let idx = rng.next_usize(1000);
        let summary = mem
            .get_session_summary(&format!("sess_{idx}"))
            .unwrap()
            .unwrap();
        assert_eq!(summary, format!("Summary for session {idx}"));
    }

    // Non-existent session returns None
    assert!(mem.get_session_summary("nonexistent").unwrap().is_none());
}

// ---------------------------------------------------------------------------
// 11. Varying batch sizes
// ---------------------------------------------------------------------------

#[test]
fn test_batch_sizes_vary() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let mut mem = HDF5Memory::create(config).unwrap();

    // Batch of 1
    mem.save_batch(vec![make_entry_simple(0, 4)]).unwrap();
    assert_eq!(mem.count(), 1);

    // Batch of 10
    let batch10: Vec<MemoryEntry> = (1..11).map(|i| make_entry_simple(i, 4)).collect();
    mem.save_batch(batch10).unwrap();
    assert_eq!(mem.count(), 11);

    // Batch of 500
    let batch500: Vec<MemoryEntry> = (11..511).map(|i| make_entry_simple(i, 4)).collect();
    mem.save_batch(batch500).unwrap();
    assert_eq!(mem.count(), 511);

    // Single saves
    for i in 511..521 {
        mem.save(make_entry_simple(i, 4)).unwrap();
    }
    assert_eq!(mem.count(), 521);
    assert_eq!(mem.count_active(), 521);
}

// ---------------------------------------------------------------------------
// 12. Delete all entries
// ---------------------------------------------------------------------------

#[test]
fn test_delete_all_entries() {
    let dir = TempDir::new().unwrap();
    let mut config = make_config(&dir, 4);
    config.compact_threshold = 0.0;
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    let entries: Vec<MemoryEntry> = (0..100).map(|i| make_entry_simple(i, 4)).collect();
    mem.save_batch(entries).unwrap();

    for i in 0..100 {
        mem.delete(i).unwrap();
    }
    assert_eq!(mem.count(), 100);
    assert_eq!(mem.count_active(), 0);

    let removed = mem.compact().unwrap();
    assert_eq!(removed, 100);
    assert_eq!(mem.count(), 0);

    // Verify persistence
    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.count(), 0);
}

// ---------------------------------------------------------------------------
// 13. Compact empty file
// ---------------------------------------------------------------------------

#[test]
fn test_compact_empty() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let mut mem = HDF5Memory::create(config).unwrap();

    let removed = mem.compact().unwrap();
    assert_eq!(removed, 0);
    assert_eq!(mem.count(), 0);
}

// ---------------------------------------------------------------------------
// 14. Search with all entries tombstoned
// ---------------------------------------------------------------------------

#[test]
fn test_search_all_tombstoned() {
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let tombstones = vec![1u8; 3]; // all tombstoned
    let query = vec![1.0, 0.0, 0.0];

    let vec_results = vector_search::cosine_similarity_batch(&query, &vectors, &tombstones);
    assert!(vec_results.is_empty());

    let chunks = vec!["hello".to_string(), "world".to_string(), "foo".to_string()];
    let bm25 = BM25Index::build(&chunks, &tombstones);
    let bm25_results = bm25.search("hello", 10);
    assert!(bm25_results.is_empty());

    let hybrid_results = hybrid_search(
        &query,
        "hello",
        &vectors,
        &chunks,
        &tombstones,
        &bm25,
        0.5,
        0.5,
        10,
    );
    assert!(hybrid_results.is_empty());
}

// ---------------------------------------------------------------------------
// 15. Unicode content handling
// ---------------------------------------------------------------------------

#[test]
fn test_unicode_content() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    let entries = vec![
        MemoryEntry {
            chunk: "Hello world in Japanese: \u{3053}\u{3093}\u{306b}\u{3061}\u{306f}".into(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            source_channel: "test".into(),
            timestamp: 1.0,
            session_id: "s1".into(),
            tags: "\u{00e9}m\u{00f6}ji".into(),
        },
        MemoryEntry {
            chunk: "Chinese: \u{4f60}\u{597d}\u{4e16}\u{754c}".into(),
            embedding: vec![0.0, 1.0, 0.0, 0.0],
            source_channel: "test".into(),
            timestamp: 2.0,
            session_id: "s1".into(),
            tags: String::new(),
        },
        MemoryEntry {
            chunk: "Emoji test: \u{1f600}\u{1f680}\u{2764}".into(),
            embedding: vec![0.0, 0.0, 1.0, 0.0],
            source_channel: "test".into(),
            timestamp: 3.0,
            session_id: "s1".into(),
            tags: String::new(),
        },
    ];
    mem.save_batch(entries).unwrap();

    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.count(), 3);

    let (_, cache, _, _) = clawhdf5_agent::storage::read_from_disk(&path).unwrap();
    assert!(cache.chunks[0].contains("\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}"));
    assert!(cache.chunks[1].contains("\u{4f60}\u{597d}"));
}

// ---------------------------------------------------------------------------
// 16. Rapid save/delete cycles
// ---------------------------------------------------------------------------

#[test]
fn test_rapid_save_delete_cycles() {
    let dir = TempDir::new().unwrap();
    let mut config = make_config(&dir, 4);
    config.compact_threshold = 0.0;
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // 50 cycles of: save 10, delete oldest 5
    let mut next_id = 0usize;
    let mut active_start = 0usize;

    for _ in 0..50 {
        let entries: Vec<MemoryEntry> = (next_id..next_id + 10)
            .map(|i| make_entry_simple(i, 4))
            .collect();
        mem.save_batch(entries).unwrap();
        next_id += 10;

        for i in active_start..active_start + 5 {
            mem.delete(i).unwrap();
        }
        active_start += 5;
    }

    // We saved 500 entries total, deleted 250
    assert_eq!(mem.count(), 500);
    assert_eq!(mem.count_active(), 250);

    // Compact and verify
    let removed = mem.compact().unwrap();
    assert_eq!(removed, 250);
    assert_eq!(mem.count(), 250);

    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.count(), 250);
}
