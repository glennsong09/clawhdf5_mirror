//! AGENTS.md generation from live HDF5Memory state.
//!
//! Produces a structured Markdown self-description of the agent's memory file,
//! suitable for session initialization or introspection.

use crate::MemoryConfig;
use crate::cache::MemoryCache;
use crate::knowledge::KnowledgeCache;
use crate::session::SessionCache;

/// Generate AGENTS.md content from memory state components.
pub fn generate(
    config: &MemoryConfig,
    cache: &MemoryCache,
    sessions: &SessionCache,
    knowledge: &KnowledgeCache,
) -> String {
    let mut md = String::with_capacity(2048);

    // Header
    md.push_str(&format!("# Agent Memory — {}\n\n", config.agent_id));

    // Identity section
    md.push_str("## Identity\n");
    md.push_str(&format!("- Agent ID: {}\n", config.agent_id));
    md.push_str(&format!("- Memory file: {}\n", config.path.display()));
    md.push_str("- Schema version: 1.0\n");
    md.push_str(&format!("- Created: {}\n", config.created_at));
    md.push_str(&format!(
        "- Embedder: {} ({}d)\n",
        config.embedder, config.embedding_dim
    ));
    md.push('\n');

    // Memory Store section
    let active = cache.count_active();
    let total = cache.len();
    let tombstoned = total - active;
    let compression = if config.compression {
        format!("gzip({})", config.compression_level)
    } else {
        "none".to_owned()
    };

    md.push_str("## Memory Store\n");
    md.push_str(&format!("- Active memories: {active}\n"));
    md.push_str(&format!("- Deleted (tombstoned): {tombstoned}\n"));
    md.push_str(&format!("- Total entries: {total}\n"));
    md.push_str(&format!(
        "- Storage: {}, compression={compression}\n",
        if config.float16 { "float16" } else { "float32" }
    ));
    md.push('\n');

    // Sessions section
    let session_count = sessions.len();
    let latest = sessions.latest_session_id().unwrap_or("none").to_owned();

    md.push_str("## Sessions\n");
    md.push_str(&format!("- Total sessions: {session_count}\n"));
    md.push_str(&format!("- Most recent session: {latest}\n"));
    md.push('\n');

    // Knowledge Graph section
    let entity_count = knowledge.entities.len();
    let relation_count = knowledge.relations.len();
    let top_entities = if knowledge.entities.is_empty() {
        "none".to_owned()
    } else {
        knowledge
            .entities
            .iter()
            .take(5)
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };

    md.push_str("## Knowledge Graph\n");
    md.push_str(&format!("- Entities: {entity_count}\n"));
    md.push_str(&format!("- Relations: {relation_count}\n"));
    md.push_str(&format!("- Top entities: {top_entities}\n"));
    md.push('\n');

    // Search Capabilities section
    md.push_str("## Search Capabilities\n");
    md.push_str("- Vector search: cosine similarity with SIMD acceleration\n");
    md.push_str("- Hybrid search: vector + BM25 (RRF)\n");
    md.push_str(&format!(
        "- Hebbian activation: enabled (boost={}, decay={})\n",
        config.hebbian_boost, config.decay_factor
    ));

    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentMemory, HDF5Memory, MemoryEntry};
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn test_config(path: PathBuf) -> MemoryConfig {
        MemoryConfig {
            path,
            agent_id: "test-agent".to_string(),
            embedder: "openai:text-embedding-3-small".to_string(),
            embedding_dim: 384,
            chunk_size: 512,
            overlap: 50,
            float16: false,
            compression: false,
            compression_level: 0,
            compact_threshold: 0.3,
            hebbian_boost: 0.15,
            decay_factor: 0.98,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            wal_enabled: false,
            wal_max_entries: 500,
        }
    }

    fn empty_cache(dim: usize) -> MemoryCache {
        MemoryCache::new(dim)
    }

    fn empty_sessions() -> SessionCache {
        SessionCache::new()
    }

    fn empty_knowledge() -> KnowledgeCache {
        KnowledgeCache::new()
    }

    #[test]
    fn test_generate_contains_agent_id() {
        let config = test_config(PathBuf::from("/tmp/test.h5"));
        let cache = empty_cache(384);
        let sessions = empty_sessions();
        let knowledge = empty_knowledge();

        let md = generate(&config, &cache, &sessions, &knowledge);
        assert!(md.contains("test-agent"), "should contain agent_id");
        assert!(
            md.contains("# Agent Memory — test-agent"),
            "should have header with agent_id"
        );
    }

    #[test]
    fn test_generate_contains_memory_counts() {
        let config = test_config(PathBuf::from("/tmp/test.h5"));
        let mut cache = empty_cache(4);
        let sessions = empty_sessions();
        let knowledge = empty_knowledge();

        // Add 5 entries, tombstone 2
        for i in 0..5 {
            cache.push(
                format!("chunk {i}"),
                vec![i as f32, 0.0, 0.0, 0.0],
                "test".into(),
                1000.0,
                "s1".into(),
                "".into(),
            );
        }
        cache.mark_deleted(1);
        cache.mark_deleted(3);

        let md = generate(&config, &cache, &sessions, &knowledge);
        assert!(md.contains("Active memories: 3"), "md = {md}");
        assert!(md.contains("Deleted (tombstoned): 2"), "md = {md}");
        assert!(md.contains("Total entries: 5"), "md = {md}");
    }

    #[test]
    fn test_generate_contains_embedding_info() {
        let config = test_config(PathBuf::from("/tmp/test.h5"));
        let cache = empty_cache(384);
        let sessions = empty_sessions();
        let knowledge = empty_knowledge();

        let md = generate(&config, &cache, &sessions, &knowledge);
        assert!(
            md.contains("openai:text-embedding-3-small"),
            "should contain embedder name"
        );
        assert!(md.contains("384d"), "should contain dimension");
    }

    #[test]
    fn test_generate_contains_entity_count() {
        let config = test_config(PathBuf::from("/tmp/test.h5"));
        let cache = empty_cache(384);
        let sessions = empty_sessions();
        let mut knowledge = empty_knowledge();

        knowledge.add_entity("Alice", "person", -1);
        knowledge.add_entity("Bob", "person", -1);
        knowledge.add_entity("Rust", "language", -1);

        let md = generate(&config, &cache, &sessions, &knowledge);
        assert!(md.contains("Entities: 3"), "md = {md}");
    }

    #[test]
    fn test_generate_contains_session_info() {
        let config = test_config(PathBuf::from("/tmp/test.h5"));
        let cache = empty_cache(384);
        let mut sessions = empty_sessions();
        let knowledge = empty_knowledge();

        sessions.add("sess-1", 0, 5, "api", "first session");
        sessions.add("sess-2", 6, 10, "slack", "second session");

        let md = generate(&config, &cache, &sessions, &knowledge);
        assert!(md.contains("Total sessions: 2"), "md = {md}");
        assert!(md.contains("Most recent session: sess-2"), "md = {md}");
    }

    #[test]
    fn test_generate_empty_state() {
        let config = test_config(PathBuf::from("/tmp/test.h5"));
        let cache = empty_cache(384);
        let sessions = empty_sessions();
        let knowledge = empty_knowledge();

        let md = generate(&config, &cache, &sessions, &knowledge);
        assert!(md.contains("Active memories: 0"), "md = {md}");
        assert!(md.contains("Deleted (tombstoned): 0"), "md = {md}");
        assert!(md.contains("Total entries: 0"), "md = {md}");
        assert!(md.contains("Total sessions: 0"), "md = {md}");
        assert!(md.contains("Entities: 0"), "md = {md}");
        assert!(md.contains("Relations: 0"), "md = {md}");
        assert!(md.contains("Most recent session: none"), "md = {md}");
        assert!(md.contains("Top entities: none"), "md = {md}");
        // Valid markdown: has headers
        assert!(md.contains("# Agent Memory"));
        assert!(md.contains("## Identity"));
        assert!(md.contains("## Memory Store"));
        assert!(md.contains("## Sessions"));
        assert!(md.contains("## Knowledge Graph"));
        assert!(md.contains("## Search Capabilities"));
    }

    #[test]
    fn test_generate_top_entities() {
        let config = test_config(PathBuf::from("/tmp/test.h5"));
        let cache = empty_cache(384);
        let sessions = empty_sessions();
        let mut knowledge = empty_knowledge();

        // Add 7 entities
        for name in &["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta"] {
            knowledge.add_entity(name, "test", -1);
        }

        let md = generate(&config, &cache, &sessions, &knowledge);
        // Should list first 5 only
        assert!(
            md.contains("Top entities: Alpha, Beta, Gamma, Delta, Epsilon"),
            "md = {md}"
        );
        // Should NOT contain Zeta or Eta in the top entities line
        let top_line = md
            .lines()
            .find(|l| l.starts_with("- Top entities:"))
            .unwrap();
        assert!(!top_line.contains("Zeta"), "top_line = {top_line}");
        assert!(!top_line.contains("Eta"), "top_line = {top_line}");
    }

    #[test]
    fn test_write_and_read_agents_md() {
        let dir = TempDir::new().unwrap();
        let config = MemoryConfig::new(dir.path().join("test.h5"), "write-read-agent", 4);
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(MemoryEntry {
            chunk: "hello".into(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            source_channel: "api".into(),
            timestamp: 1000.0,
            session_id: "s1".into(),
            tags: "".into(),
        })
        .unwrap();

        mem.write_agents_md().unwrap();

        let read_back = HDF5Memory::read_agents_md(&dir.path().join("test.h5")).unwrap();
        let generated = mem.generate_agents_md();
        assert_eq!(read_back, generated);
        assert!(read_back.contains("write-read-agent"));
    }

    #[test]
    fn test_generate_hebbian_config() {
        let mut config = test_config(PathBuf::from("/tmp/test.h5"));
        config.hebbian_boost = 0.25;
        config.decay_factor = 0.95;
        let cache = empty_cache(384);
        let sessions = empty_sessions();
        let knowledge = empty_knowledge();

        let md = generate(&config, &cache, &sessions, &knowledge);
        assert!(md.contains("boost=0.25"), "md = {md}");
        assert!(md.contains("decay=0.95"), "md = {md}");
    }
}
