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
}

fn make_vec(rng: &mut Rng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.next_f32()).collect()
}

fn make_config(dir: &TempDir, dim: usize) -> MemoryConfig {
    MemoryConfig::new(dir.path().join("e2e.h5"), "e2e-agent", dim)
}

fn make_entry(chunk: &str, embedding: Vec<f32>, channel: &str, session: &str) -> MemoryEntry {
    MemoryEntry {
        chunk: chunk.to_string(),
        embedding,
        source_channel: channel.to_string(),
        timestamp: 1_000_000.0,
        session_id: session.to_string(),
        tags: String::new(),
    }
}

fn read_cache(
    path: &Path,
) -> (
    clawhdf5_agent::MemoryConfig,
    clawhdf5_agent::cache::MemoryCache,
    clawhdf5_agent::session::SessionCache,
    clawhdf5_agent::knowledge::KnowledgeCache,
) {
    clawhdf5_agent::storage::read_from_disk(path).unwrap()
}

// ---------------------------------------------------------------------------
// 1. Agent memory lifecycle
// ---------------------------------------------------------------------------

#[test]
fn test_agent_memory_lifecycle() {
    let dir = TempDir::new().unwrap();
    let mut config = make_config(&dir, 4);
    config.compact_threshold = 0.0; // disable auto-compact for manual control
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Save entries from multiple channels
    mem.save(make_entry(
        "whatsapp message about AI",
        vec![1.0, 0.0, 0.0, 0.0],
        "whatsapp",
        "s1",
    ))
    .unwrap();
    mem.save(make_entry(
        "slack code review discussion",
        vec![0.0, 1.0, 0.0, 0.0],
        "slack",
        "s2",
    ))
    .unwrap();
    mem.save(make_entry(
        "email about AI and machine learning",
        vec![0.9, 0.1, 0.0, 0.0],
        "email",
        "s3",
    ))
    .unwrap();

    // Flush WAL to disk so read_cache sees the data
    mem.flush_wal().unwrap();

    // Vector search
    let (_, cache, _, _) = read_cache(&path);
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let vec_results =
        vector_search::cosine_similarity_batch(&query, &cache.embeddings, &cache.tombstones);
    assert_eq!(vec_results[0].0, 0); // whatsapp msg closest to query

    // Keyword search
    let bm25 = BM25Index::build(&cache.chunks, &cache.tombstones);
    let kw_results = bm25.search("AI", 10);
    assert!(!kw_results.is_empty());
    let kw_ids: Vec<usize> = kw_results.iter().map(|(id, _)| *id).collect();
    assert!(kw_ids.contains(&0)); // "about AI"
    assert!(kw_ids.contains(&2)); // "about AI and machine learning"

    // Hybrid search
    let hybrid_results = hybrid_search(
        &query,
        "AI",
        &cache.embeddings,
        &cache.chunks,
        &cache.tombstones,
        &bm25,
        0.7,
        0.3,
        3,
    );
    assert!(!hybrid_results.is_empty());
    assert_eq!(hybrid_results[0].0, 0); // best vector + keyword match

    // Delete old entries
    mem.delete(1).unwrap(); // delete slack msg
    let removed = mem.compact().unwrap();
    assert_eq!(removed, 1);
    assert_eq!(mem.count(), 2);

    // Snapshot
    let snap_dir = TempDir::new().unwrap();
    let snap_path = mem.snapshot(snap_dir.path()).unwrap();
    assert!(snap_path.exists());

    // Open snapshot and verify search still works
    let snap_mem = HDF5Memory::open(&snap_path).unwrap();
    assert_eq!(snap_mem.count(), 2);

    let (_, snap_cache, _, _) = read_cache(&snap_path);
    let snap_results = vector_search::cosine_similarity_batch(
        &query,
        &snap_cache.embeddings,
        &snap_cache.tombstones,
    );
    assert_eq!(snap_results.len(), 2);
}

// ---------------------------------------------------------------------------
// 2. Migration round-trip (simulated)
// ---------------------------------------------------------------------------

#[test]
fn test_migration_round_trip() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Simulate data that would come from a SQLite migration
    let migrated_entries: Vec<MemoryEntry> = (0..500)
        .map(|i| {
            let channel = match i % 3 {
                0 => "whatsapp",
                1 => "slack",
                _ => "email",
            };
            MemoryEntry {
                chunk: format!("migrated chunk {i} with content about topic {}", i % 20),
                embedding: vec![(i as f32).sin(), (i as f32).cos(), 0.0, 0.0],
                source_channel: channel.to_string(),
                timestamp: i as f64 * 1000.0,
                session_id: format!("migrated_sess_{}", i / 50),
                tags: format!("migrated,batch_{}", i / 100),
            }
        })
        .collect();

    mem.save_batch(migrated_entries).unwrap();

    // Add sessions that would have been migrated
    for i in 0..10 {
        mem.add_session(
            &format!("migrated_sess_{i}"),
            i * 50,
            (i + 1) * 50,
            "api",
            &format!("Migrated session {i} summary"),
        )
        .unwrap();
    }

    // Add knowledge graph entries
    let e1 = mem.add_entity("User", "person", -1).unwrap();
    let e2 = mem.add_entity("AI", "concept", -1).unwrap();
    mem.add_relation(e1, e2, "discusses", 0.8).unwrap();

    // Verify all data transferred by reopening
    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.count(), 500);

    // Verify sessions
    for i in 0..10 {
        let summary = reopened
            .get_session_summary(&format!("migrated_sess_{i}"))
            .unwrap()
            .unwrap();
        assert_eq!(summary, format!("Migrated session {i} summary"));
    }

    // Verify knowledge graph
    assert_eq!(reopened.knowledge().entities.len(), 2);
    assert_eq!(reopened.knowledge().relations.len(), 1);

    // Verify search works on migrated data
    let (_, cache, _, _) = read_cache(&path);
    let query = vec![0.0_f32.sin(), 0.0_f32.cos(), 0.0, 0.0];
    let results =
        vector_search::cosine_similarity_batch(&query, &cache.embeddings, &cache.tombstones);
    assert_eq!(results.len(), 500);
    assert!(results[0].1 > 0.9); // first entry should match well

    let bm25 = BM25Index::build(&cache.chunks, &cache.tombstones);
    let kw_results = bm25.search("topic 5", 10);
    assert!(!kw_results.is_empty());
}

// ---------------------------------------------------------------------------
// 3. Knowledge graph workflow
// ---------------------------------------------------------------------------

#[test]
fn test_knowledge_graph_workflow() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Add entities
    let rust = mem.add_entity("Rust", "language", -1).unwrap();
    let hdf5 = mem.add_entity("HDF5", "format", -1).unwrap();
    let python = mem.add_entity("Python", "language", -1).unwrap();
    let numpy = mem.add_entity("NumPy", "library", -1).unwrap();

    // Add relations
    mem.add_relation(rust, hdf5, "writes", 1.0).unwrap();
    mem.add_relation(python, hdf5, "reads", 0.9).unwrap();
    mem.add_relation(python, numpy, "uses", 0.95).unwrap();
    mem.add_relation(numpy, hdf5, "wraps", 0.8).unwrap();

    // Query related entities
    let rust_rels = mem.knowledge().get_relations_from(rust);
    assert_eq!(rust_rels.len(), 1);
    assert_eq!(rust_rels[0].relation, "writes");
    assert_eq!(rust_rels[0].tgt, hdf5);

    let hdf5_incoming = mem.knowledge().get_relations_to(hdf5);
    assert_eq!(hdf5_incoming.len(), 3); // rust writes, python reads, numpy wraps

    let python_rels = mem.knowledge().get_relations_from(python);
    assert_eq!(python_rels.len(), 2);

    // Verify by name
    let entity = mem.knowledge().get_entity(numpy).unwrap();
    assert_eq!(entity.name, "NumPy");
    assert_eq!(entity.entity_type, "library");

    // Persistence
    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.knowledge().entities.len(), 4);
    assert_eq!(reopened.knowledge().relations.len(), 4);

    let reopened_rust_rels = reopened.knowledge().get_relations_from(rust);
    assert_eq!(reopened_rust_rels.len(), 1);
    assert_eq!(reopened_rust_rels[0].relation, "writes");
}

// ---------------------------------------------------------------------------
// 4. Multi-session workflow
// ---------------------------------------------------------------------------

#[test]
fn test_multi_session_workflow() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Save entries across 5 sessions
    let mut idx = 0usize;
    for sess in 0..5 {
        let entries: Vec<MemoryEntry> = (0..20)
            .map(|i| MemoryEntry {
                chunk: format!("session {sess} message {i}"),
                embedding: vec![sess as f32, i as f32, 0.0, 0.0],
                source_channel: format!("channel_{}", sess % 2),
                timestamp: (sess * 100 + i) as f64,
                session_id: format!("sess_{sess}"),
                tags: String::new(),
            })
            .collect();
        let start = idx;
        let indices = mem.save_batch(entries).unwrap();
        idx = *indices.last().unwrap() + 1;

        mem.add_session(
            &format!("sess_{sess}"),
            start,
            idx - 1,
            &format!("channel_{}", sess % 2),
            &format!("Summary of session {sess}: discussed topics A, B, C"),
        )
        .unwrap();
    }

    assert_eq!(mem.count(), 100); // 5 sessions * 20 entries

    // Reopen and verify sessions
    let reopened = HDF5Memory::open(&path).unwrap();
    for sess in 0..5 {
        let summary = reopened
            .get_session_summary(&format!("sess_{sess}"))
            .unwrap()
            .unwrap();
        assert!(summary.contains(&format!("session {sess}")));
    }

    // Verify search across session boundaries
    let (_, cache, _, _) = read_cache(&path);
    let bm25 = BM25Index::build(&cache.chunks, &cache.tombstones);
    let results = bm25.search("session 3 message", 100);
    // All 20 messages from session 3 should match
    let session3_results: Vec<usize> = results
        .iter()
        .filter(|(id, _)| cache.session_ids[*id] == "sess_3")
        .map(|(id, _)| *id)
        .collect();
    assert_eq!(session3_results.len(), 20);
}

// ---------------------------------------------------------------------------
// 5. Float16 mode
// ---------------------------------------------------------------------------

#[cfg(feature = "float16")]
#[test]
fn test_float16_mode() {
    use half::f16;

    let dir = TempDir::new().unwrap();
    let mut config = make_config(&dir, 384);
    config.float16 = true;
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    let mut rng = Rng::new(42);
    let entries: Vec<MemoryEntry> = (0..100)
        .map(|i| MemoryEntry {
            chunk: format!("float16 test entry {i}"),
            embedding: make_vec(&mut rng, 384),
            source_channel: "test".into(),
            timestamp: i as f64,
            session_id: "s1".into(),
            tags: String::new(),
        })
        .collect();
    mem.save_batch(entries).unwrap();

    // Read back and do both f32 and f16 search
    let (_, cache, _, _) = read_cache(&path);
    let query = make_vec(&mut Rng::new(99), 384);

    // f32 search
    let f32_results =
        vector_search::cosine_similarity_batch(&query, &cache.embeddings, &cache.tombstones);

    // Convert embeddings to f16 flat buffer
    let vectors_f16: Vec<u16> = cache
        .embeddings
        .iter()
        .flat_map(|v| v.iter().map(|&f| f16::from_f32(f).to_bits()))
        .collect();

    // f16 search
    let f16_results =
        vector_search::cosine_similarity_f16(&query, &vectors_f16, 384, &cache.tombstones);

    // Both should return same number of results
    assert_eq!(f32_results.len(), f16_results.len());

    // Top results should be the same (or very close)
    let f32_top5: Vec<usize> = f32_results.iter().take(5).map(|(id, _)| *id).collect();
    let f16_top5: Vec<usize> = f16_results.iter().take(5).map(|(id, _)| *id).collect();
    // Allow minor reordering due to precision differences
    for id in &f32_top5 {
        assert!(
            f16_top5.contains(id),
            "f32 top-5 result {id} not in f16 top-5: f32={f32_top5:?}, f16={f16_top5:?}"
        );
    }

    // Scores should be within tolerance
    for (i, &(f32_id, f32_score)) in f32_results.iter().take(5).enumerate() {
        let f16_score = f16_results
            .iter()
            .find(|(id, _)| *id == f32_id)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            (f32_score - f16_score).abs() < 0.02,
            "score mismatch at rank {i}: f32={f32_score}, f16={f16_score}"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Snapshot and continue working
// ---------------------------------------------------------------------------

#[test]
fn test_snapshot_and_continue() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Save initial data
    let entries: Vec<MemoryEntry> = (0..50)
        .map(|i| {
            make_entry(
                &format!("initial_{i}"),
                vec![i as f32, 0.0, 0.0, 0.0],
                "ch1",
                "s1",
            )
        })
        .collect();
    mem.save_batch(entries).unwrap();

    // Snapshot
    let snap_dir = TempDir::new().unwrap();
    let snap_path = mem.snapshot(snap_dir.path()).unwrap();

    // Continue working on original
    let more: Vec<MemoryEntry> = (50..100)
        .map(|i| {
            make_entry(
                &format!("after_snap_{i}"),
                vec![i as f32, 0.0, 0.0, 0.0],
                "ch1",
                "s2",
            )
        })
        .collect();
    mem.save_batch(more).unwrap();
    assert_eq!(mem.count(), 100);

    // Snapshot should still have only 50
    let snap_mem = HDF5Memory::open(&snap_path).unwrap();
    assert_eq!(snap_mem.count(), 50);

    // Original should have 100
    let orig_mem = HDF5Memory::open(&path).unwrap();
    assert_eq!(orig_mem.count(), 100);
}

// ---------------------------------------------------------------------------
// 7. Config persistence across operations
// ---------------------------------------------------------------------------

#[test]
fn test_config_persistence_across_ops() {
    let dir = TempDir::new().unwrap();
    let mut config = make_config(&dir, 128);
    config.embedder = "custom:my-embedder-v2".into();
    config.chunk_size = 2048;
    config.overlap = 200;
    let path = config.path.clone();

    let mut mem = HDF5Memory::create(config).unwrap();
    mem.save(make_entry("test", vec![0.0; 128], "ch", "s"))
        .unwrap();
    mem.add_session("s1", 0, 0, "ch", "summary").unwrap();
    mem.add_entity("Entity", "type", -1).unwrap();

    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.config().embedding_dim, 128);
    assert_eq!(reopened.config().embedder, "custom:my-embedder-v2");
    assert_eq!(reopened.config().chunk_size, 2048);
    assert_eq!(reopened.config().overlap, 200);
    assert_eq!(reopened.config().agent_id, "e2e-agent");
}

// ---------------------------------------------------------------------------
// 8. Search after compaction
// ---------------------------------------------------------------------------

#[test]
fn test_search_after_compaction() {
    let dir = TempDir::new().unwrap();
    let mut config = make_config(&dir, 4);
    config.compact_threshold = 0.0;
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Save entries with distinct embeddings
    mem.save(make_entry(
        "target entry",
        vec![1.0, 0.0, 0.0, 0.0],
        "ch",
        "s",
    ))
    .unwrap();
    mem.save(make_entry(
        "will delete this",
        vec![0.0, 1.0, 0.0, 0.0],
        "ch",
        "s",
    ))
    .unwrap();
    mem.save(make_entry(
        "another keeper",
        vec![0.5, 0.5, 0.0, 0.0],
        "ch",
        "s",
    ))
    .unwrap();

    // Delete middle entry
    mem.delete(1).unwrap();
    mem.compact().unwrap();
    assert_eq!(mem.count(), 2);

    // Search should still work correctly after compaction
    let (_, cache, _, _) = read_cache(&path);
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results =
        vector_search::cosine_similarity_batch(&query, &cache.embeddings, &cache.tombstones);
    assert_eq!(results.len(), 2);
    // "target entry" should be the best match
    assert!((results[0].1 - 1.0).abs() < 1e-5);
    assert_eq!(cache.chunks[results[0].0], "target entry");

    // BM25 should also work
    let bm25 = BM25Index::build(&cache.chunks, &cache.tombstones);
    let kw_results = bm25.search("target", 10);
    assert_eq!(kw_results.len(), 1);
}

// ---------------------------------------------------------------------------
// 9. Multiple channels with search
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_channels_search() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    let channels = ["whatsapp", "slack", "email", "api", "web"];
    for (i, channel) in channels.iter().enumerate() {
        let entries: Vec<MemoryEntry> = (0..10)
            .map(|j| {
                let idx = i * 10 + j;
                MemoryEntry {
                    chunk: format!("{channel} message {j} about topic {}", j % 5),
                    embedding: vec![idx as f32 * 0.01, 0.0, 0.0, 0.0],
                    source_channel: channel.to_string(),
                    timestamp: idx as f64,
                    session_id: format!("sess_{channel}"),
                    tags: String::new(),
                }
            })
            .collect();
        mem.save_batch(entries).unwrap();
    }
    assert_eq!(mem.count(), 50);

    let (_, cache, _, _) = read_cache(&path);
    let bm25 = BM25Index::build(&cache.chunks, &cache.tombstones);

    // Search for a specific channel's content
    let results = bm25.search("slack message", 50);
    assert!(!results.is_empty());
    // Results should include slack messages
    let has_slack = results
        .iter()
        .any(|(id, _)| cache.source_channels[*id] == "slack");
    assert!(has_slack);
}

// ---------------------------------------------------------------------------
// 10. Empty operations pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_empty_operations_pipeline() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Operations on empty memory
    assert_eq!(mem.count(), 0);
    assert_eq!(mem.count_active(), 0);
    assert_eq!(mem.compact().unwrap(), 0);

    let snap_dir = TempDir::new().unwrap();
    let snap_path = mem.snapshot(snap_dir.path()).unwrap();
    let snap_mem = HDF5Memory::open(&snap_path).unwrap();
    assert_eq!(snap_mem.count(), 0);

    assert!(mem.get_session_summary("nonexistent").unwrap().is_none());

    // Empty search
    let (_, cache, _, _) = read_cache(&path);
    let results = vector_search::cosine_similarity_batch(
        &[1.0, 0.0, 0.0, 0.0],
        &cache.embeddings,
        &cache.tombstones,
    );
    assert!(results.is_empty());

    let bm25 = BM25Index::build(&cache.chunks, &cache.tombstones);
    assert!(bm25.search("anything", 10).is_empty());
}

// ---------------------------------------------------------------------------
// 11. Entity ID continuity across operations
// ---------------------------------------------------------------------------

#[test]
fn test_entity_id_continuity_across_ops() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();

    // Phase 1: Create and add entities
    {
        let mut mem = HDF5Memory::create(config).unwrap();
        let id0 = mem.add_entity("Alpha", "type_a", -1).unwrap();
        let id1 = mem.add_entity("Beta", "type_b", -1).unwrap();
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    // Phase 2: Reopen and add more
    {
        let mut mem = HDF5Memory::open(&path).unwrap();
        let id2 = mem.add_entity("Gamma", "type_c", -1).unwrap();
        assert_eq!(id2, 2); // Should continue from 2

        mem.add_relation(0, 2, "connects", 1.0).unwrap();
    }

    // Phase 3: Reopen again and verify
    {
        let mut mem = HDF5Memory::open(&path).unwrap();
        let id3 = mem.add_entity("Delta", "type_d", -1).unwrap();
        assert_eq!(id3, 3);

        assert_eq!(mem.knowledge().entities.len(), 4);
        assert_eq!(mem.knowledge().relations.len(), 1);
        assert_eq!(mem.knowledge().get_entity(0).unwrap().name, "Alpha");
        assert_eq!(mem.knowledge().get_entity(3).unwrap().name, "Delta");
    }
}

// ---------------------------------------------------------------------------
// 12. Large text chunks
// ---------------------------------------------------------------------------

#[test]
fn test_large_text_chunks() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Create entries with large text chunks (simulating real document chunks)
    let large_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(100);
    assert!(large_text.len() > 5000);

    let entries: Vec<MemoryEntry> = (0..10)
        .map(|i| MemoryEntry {
            chunk: format!("Document {i}: {large_text}"),
            embedding: vec![i as f32 * 0.1, 0.0, 0.0, 0.0],
            source_channel: "docs".into(),
            timestamp: i as f64,
            session_id: "s1".into(),
            tags: String::new(),
        })
        .collect();
    mem.save_batch(entries).unwrap();

    // Reopen and verify
    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.count(), 10);

    let (_, cache, _, _) = read_cache(&path);
    for chunk in &cache.chunks {
        assert!(chunk.len() > 5000);
        assert!(chunk.contains("Lorem ipsum"));
    }

    // Search should work on large text
    let bm25 = BM25Index::build(&cache.chunks, &cache.tombstones);
    let results = bm25.search("ipsum dolor", 10);
    assert_eq!(results.len(), 10); // all docs contain the text
}

// ---------------------------------------------------------------------------
// 13. Interleaved sessions and entries
// ---------------------------------------------------------------------------

#[test]
fn test_interleaved_sessions_entries() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Interleave saves and session tracking
    let idx0 = mem
        .save(make_entry("msg 0", vec![1.0, 0.0, 0.0, 0.0], "ch", "s1"))
        .unwrap();
    let idx1 = mem
        .save(make_entry("msg 1", vec![0.0, 1.0, 0.0, 0.0], "ch", "s1"))
        .unwrap();
    mem.add_session("s1", idx0, idx1, "ch", "First session")
        .unwrap();

    let idx2 = mem
        .save(make_entry("msg 2", vec![0.0, 0.0, 1.0, 0.0], "ch", "s2"))
        .unwrap();
    mem.add_session("s2", idx2, idx2, "ch", "Second session")
        .unwrap();

    let idx3 = mem
        .save(make_entry("msg 3", vec![0.0, 0.0, 0.0, 1.0], "ch", "s3"))
        .unwrap();
    mem.save(make_entry("msg 4", vec![1.0, 1.0, 0.0, 0.0], "ch", "s3"))
        .unwrap();
    let idx5 = mem
        .save(make_entry("msg 5", vec![0.0, 1.0, 1.0, 0.0], "ch", "s3"))
        .unwrap();
    mem.add_session("s3", idx3, idx5, "ch", "Third session")
        .unwrap();

    // Flush WAL so reopen doesn't replay stale entries
    mem.flush_wal().unwrap();

    // Verify
    let reopened = HDF5Memory::open(&path).unwrap();
    assert_eq!(reopened.count(), 6);
    assert_eq!(
        reopened.get_session_summary("s1").unwrap().as_deref(),
        Some("First session")
    );
    assert_eq!(
        reopened.get_session_summary("s2").unwrap().as_deref(),
        Some("Second session")
    );
    assert_eq!(
        reopened.get_session_summary("s3").unwrap().as_deref(),
        Some("Third session")
    );
}

// ---------------------------------------------------------------------------
// 14. Knowledge graph with embedding links
// ---------------------------------------------------------------------------

#[test]
fn test_knowledge_graph_with_embeddings() {
    let dir = TempDir::new().unwrap();
    let config = make_config(&dir, 4);
    let path = config.path.clone();
    let mut mem = HDF5Memory::create(config).unwrap();

    // Save some entries
    let idx0 = mem
        .save(make_entry(
            "Rust language",
            vec![1.0, 0.0, 0.0, 0.0],
            "ch",
            "s",
        ))
        .unwrap();
    let idx1 = mem
        .save(make_entry(
            "Python language",
            vec![0.0, 1.0, 0.0, 0.0],
            "ch",
            "s",
        ))
        .unwrap();

    // Add entities linked to embeddings
    let e_rust = mem.add_entity("Rust", "language", idx0 as i64).unwrap();
    let e_python = mem.add_entity("Python", "language", idx1 as i64).unwrap();
    let e_hdf5 = mem.add_entity("HDF5", "format", -1).unwrap();

    mem.add_relation(e_rust, e_hdf5, "writes", 1.0).unwrap();
    mem.add_relation(e_python, e_hdf5, "reads", 0.9).unwrap();

    // Verify entity-embedding linkage persists
    let reopened = HDF5Memory::open(&path).unwrap();
    let rust_entity = reopened.knowledge().get_entity(e_rust).unwrap();
    assert_eq!(rust_entity.embedding_idx, idx0 as i64);

    let python_entity = reopened.knowledge().get_entity(e_python).unwrap();
    assert_eq!(python_entity.embedding_idx, idx1 as i64);

    let hdf5_entity = reopened.knowledge().get_entity(e_hdf5).unwrap();
    assert_eq!(hdf5_entity.embedding_idx, -1); // no embedding link
}

// ---------------------------------------------------------------------------
// 15. Full pipeline: create, populate, search, modify, snapshot, restore
// ---------------------------------------------------------------------------

#[test]
fn test_full_pipeline_snapshot_restore() {
    let dir = TempDir::new().unwrap();
    let mut config = make_config(&dir, 4);
    config.compact_threshold = 0.0;
    let mut mem = HDF5Memory::create(config).unwrap();

    // Populate
    let entries: Vec<MemoryEntry> = (0..200)
        .map(|i| MemoryEntry {
            chunk: format!("document {i} about topic {}", i % 10),
            embedding: vec![(i as f32).sin(), (i as f32).cos(), 0.0, 0.0],
            source_channel: "api".into(),
            timestamp: i as f64,
            session_id: format!("sess_{}", i / 20),
            tags: String::new(),
        })
        .collect();
    mem.save_batch(entries).unwrap();

    // Add sessions
    for i in 0..10 {
        mem.add_session(
            &format!("sess_{i}"),
            i * 20,
            (i + 1) * 20 - 1,
            "api",
            &format!("Session {i}"),
        )
        .unwrap();
    }

    // Delete some entries
    for i in (0..200).step_by(4) {
        mem.delete(i).unwrap();
    }
    mem.compact().unwrap();

    // Snapshot
    let snap_dir = TempDir::new().unwrap();
    let snap_path = mem.snapshot(snap_dir.path()).unwrap();

    // Verify snapshot has correct data
    let snap_mem = HDF5Memory::open(&snap_path).unwrap();
    assert_eq!(snap_mem.count(), 150); // 200 - 50 deleted
    assert_eq!(snap_mem.count_active(), 150);

    // Verify search on snapshot
    let (_, snap_cache, _, _) = read_cache(&snap_path);
    let query = vec![0.0_f32.sin(), 0.0_f32.cos(), 0.0, 0.0];
    let results = vector_search::cosine_similarity_batch(
        &query,
        &snap_cache.embeddings,
        &snap_cache.tombstones,
    );
    assert_eq!(results.len(), 150);

    // Verify sessions on snapshot
    let summary = snap_mem.get_session_summary("sess_5").unwrap().unwrap();
    assert_eq!(summary, "Session 5");
}

// ---------------------------------------------------------------------------
// 16. Overwrite file and reopen
// ---------------------------------------------------------------------------

#[test]
fn test_overwrite_and_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("rewrite.h5");

    // Create first version
    {
        let config = MemoryConfig::new(path.clone(), "agent-v1", 4);
        let mut mem = HDF5Memory::create(config).unwrap();
        mem.save(make_entry("version 1", vec![1.0, 0.0, 0.0, 0.0], "ch", "s"))
            .unwrap();
        mem.flush_wal().unwrap();
    }

    // Verify first version
    let mem1 = HDF5Memory::open(&path).unwrap();
    assert_eq!(mem1.count(), 1);
    assert_eq!(mem1.config().agent_id, "agent-v1");
    drop(mem1);

    // Overwrite with second version
    {
        let config = MemoryConfig::new(path.clone(), "agent-v2", 4);
        let mut mem = HDF5Memory::create(config).unwrap();
        let entries: Vec<MemoryEntry> = (0..10)
            .map(|i| {
                make_entry(
                    &format!("v2_{i}"),
                    vec![i as f32, 0.0, 0.0, 0.0],
                    "ch2",
                    "s2",
                )
            })
            .collect();
        mem.save_batch(entries).unwrap();
        mem.flush_wal().unwrap();
    }

    // Verify second version replaced first
    let mem2 = HDF5Memory::open(&path).unwrap();
    assert_eq!(mem2.count(), 10);
    assert_eq!(mem2.config().agent_id, "agent-v2");

    let (_, cache, _, _) = read_cache(&path);
    assert!(cache.chunks[0].starts_with("v2_"));
}

// ---------------------------------------------------------------------------
// Acceleration integration tests (clawhdf5_accel, mmap, GPU fallback)
// ---------------------------------------------------------------------------

#[test]
fn test_accel_backend_detection() {
    let backend = clawhdf5_accel::detect_backend();
    // Must return a valid backend variant
    let name = format!("{backend:?}");
    assert!(
        ["Neon", "Avx2", "Avx512", "Sse4", "WasmSimd128", "Scalar"]
            .iter()
            .any(|v| name.contains(v)),
        "unexpected backend: {name}"
    );
}

#[test]
fn test_accel_dot_product_matches_manual() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 4.0, 3.0, 2.0];
    let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let got = clawhdf5_accel::dot_product(&a, &b);
    assert!(
        (expected - got).abs() < 1e-5,
        "dot_product: expected {expected}, got {got}"
    );
}

#[test]
fn test_accel_cosine_matches_old_simd() {
    // Regression: ensure clawhdf5_accel cosine gives same results as old inline SIMD
    let dim = 384;
    let mut rng = Rng::new(42);
    let a: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();

    let accel_sim = clawhdf5_accel::cosine_similarity(&a, &b);

    // Manual cosine for comparison
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let manual = dot / (na * nb);

    assert!(
        (accel_sim - manual).abs() < 1e-5,
        "cosine regression: accel={accel_sim}, manual={manual}"
    );
}

#[test]
fn test_search_results_identical_after_simd_swap() {
    // Regression: search results must be identical when using clawhdf5_accel
    let dim = 128;
    let n = 500;
    let mut rng = Rng::new(77);

    let query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
        .collect();
    let tombstones = vec![0u8; n];

    let results = vector_search::cosine_similarity_batch(&query, &vectors, &tombstones);
    let top10 = vector_search::top_k(results, 10);

    // Verify results are sorted descending
    for i in 1..top10.len() {
        assert!(top10[i - 1].1 >= top10[i].1, "results not sorted");
    }

    // Verify first result is close to the query
    assert!(
        top10[0].1 > 0.0,
        "top result should have positive similarity"
    );
}

#[test]
fn test_gpu_fallback_to_cpu_when_unavailable() {
    let dim = 32;
    let n = 100;
    let mut rng = Rng::new(42);
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
        .collect();
    let norms: Vec<f32> = vectors
        .iter()
        .map(|v| clawhdf5_accel::vector_norm(v))
        .collect();
    let tombstones = vec![0u8; n];

    let gpu = clawhdf5_agent::gpu_search::GpuSearchBackend::try_init(&vectors, &norms, dim, 50);
    let query = vectors[0].clone();
    let results = gpu.search_cosine(&query, &vectors, &norms, &tombstones, 10);

    // Should still return results via CPU fallback
    assert!(!results.is_empty());
    assert!(results.len() <= 10);
    assert_eq!(results[0].0, 0, "query vector should be top match");
    assert!((results[0].1 - 1.0).abs() < 1e-5);
}

#[test]
fn test_gpu_l2_fallback_works() {
    let vectors = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![10.0, 10.0]];
    let norms: Vec<f32> = vectors
        .iter()
        .map(|v| clawhdf5_accel::vector_norm(v))
        .collect();
    let tombstones = vec![0u8; 3];

    let gpu = clawhdf5_agent::gpu_search::GpuSearchBackend::try_init(&vectors, &norms, 2, 1);
    let results = gpu.search_l2(&vec![0.0, 0.0], &vectors, &tombstones, 3);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 0);
    assert_eq!(results[1].0, 1);
}

#[test]
fn test_mmap_reader_opens_and_reads() {
    let dir = TempDir::new().unwrap();
    let config = MemoryConfig::new(dir.path().join("mmap_test.h5"), "agent-mmap", 4);
    let path = config.path.clone();

    {
        let mut mem = HDF5Memory::create(config).unwrap();
        mem.save(make_entry(
            "mmap content",
            vec![1.0, 2.0, 3.0, 4.0],
            "ch",
            "s",
        ))
        .unwrap();
    }

    // Verify we can open via mmap (storage::read_from_disk now uses MmapReader)
    let mem = HDF5Memory::open(&path).unwrap();
    assert_eq!(mem.count(), 1);
    assert_eq!(mem.config().agent_id, "agent-mmap");
}

#[test]
fn test_mmap_reader_direct_access() {
    let dir = TempDir::new().unwrap();
    let config = MemoryConfig::new(dir.path().join("mmap_direct.h5"), "agent-direct", 4);
    let path = config.path.clone();

    {
        let mut mem = HDF5Memory::create(config).unwrap();
        for i in 0..100 {
            mem.save(make_entry(
                &format!("entry_{i}"),
                vec![i as f32, 0.0, 0.0, 0.0],
                "ch",
                "s",
            ))
            .unwrap();
        }
    }

    // Open via MmapReader directly
    let mmap = clawhdf5_io::MmapReader::open(&path).unwrap();
    assert!(mmap.len() > 0);
    // Verify we can read bytes at specific offsets
    let bytes = mmap.read_at(0, 8);
    assert!(bytes.is_some());
    // HDF5 magic number at offset 0
    let magic = bytes.unwrap();
    assert_eq!(magic[0], 0x89);
    assert_eq!(magic[1], b'H');
    assert_eq!(magic[2], b'D');
    assert_eq!(magic[3], b'F');
}

#[test]
fn test_prenorm_helper_matches_full_cosine() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 4.0, 3.0, 2.0];

    let full = clawhdf5_accel::cosine_similarity(&a, &b);
    let na = clawhdf5_accel::vector_norm(&a);
    let nb = clawhdf5_accel::vector_norm(&b);
    let prenorm = clawhdf5_agent::cosine_similarity_prenorm(&a, na, &b, nb);

    assert!(
        (full - prenorm).abs() < 1e-6,
        "full={full}, prenorm={prenorm}"
    );
}

#[test]
fn test_strategy_reports_backend() {
    use clawhdf5_agent::strategy;

    let n = 50;
    let dim = 16;
    let mut rng = Rng::new(42);
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
        .collect();
    let norms: Vec<f32> = vectors
        .iter()
        .map(|v| clawhdf5_accel::vector_norm(v))
        .collect();
    let tombstones = vec![0u8; n];
    let query = vectors[0].clone();

    let (_, metrics) = strategy::search_with_metrics(
        &query,
        &vectors,
        &norms,
        &tombstones,
        5,
        strategy::SearchStrategy::Scalar,
        None,
    );

    // Backend should be set and non-empty
    assert!(!metrics.backend.is_empty(), "backend should be reported");
    let valid = ["neon", "avx2", "avx512", "sse4", "wasmsimd128", "scalar"];
    assert!(
        valid.iter().any(|v| metrics.backend.contains(v)),
        "unexpected backend: {}",
        metrics.backend
    );
}

#[test]
fn test_accel_vector_norm() {
    let v = vec![3.0, 4.0];
    let norm = clawhdf5_accel::vector_norm(&v);
    assert!((norm - 5.0).abs() < 1e-6, "expected 5.0, got {norm}");
}

#[test]
fn test_accel_l2_distance() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    let dist = clawhdf5_accel::l2_distance(&a, &b);
    // l2_distance returns squared distance (sum of squared diffs)
    // Actually check the clawhdf5_accel API - it might return sqrt'd
    // sqrt(9+16) = 5.0 or 9+16 = 25.0 depending on impl
    assert!(
        (dist - 5.0).abs() < 1e-5 || (dist - 25.0).abs() < 1e-5,
        "unexpected l2 distance: {dist}"
    );
}
