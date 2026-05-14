//! Memory provenance tracking and integrity verification.
//!
//! Records the origin, authorship, and integrity of every memory chunk
//! so the system can detect tampering and trace data lineage.

use std::collections::HashMap;

pub use crate::consolidation::MemorySource;

// ---------------------------------------------------------------------------
// Hash helper (std-only FNV-1a 64-bit)
// ---------------------------------------------------------------------------

fn fnv1a_64(text: &str) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// Display for MemorySource
// ---------------------------------------------------------------------------

impl std::fmt::Display for MemorySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemorySource::User => write!(f, "User"),
            MemorySource::System => write!(f, "System"),
            MemorySource::Tool => write!(f, "Tool"),
            MemorySource::Retrieval => write!(f, "Retrieval"),
            MemorySource::Correction => write!(f, "Correction"),
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryProvenance
// ---------------------------------------------------------------------------

/// Full provenance record for a single memory chunk.
#[derive(Clone, Debug)]
pub struct MemoryProvenance {
    pub record_id: u64,
    pub source: MemorySource,
    /// Agent or user that created this record.
    pub created_by: String,
    /// Unix timestamp (seconds) of creation.
    pub created_at: f64,
    /// FNV-1a 64-bit hash of the chunk text for integrity checking.
    pub content_hash: u64,
    pub session_id: String,
    pub verified: bool,
}

impl MemoryProvenance {
    /// Create a new provenance record, computing the content hash automatically.
    pub fn new(
        record_id: u64,
        source: MemorySource,
        created_by: impl Into<String>,
        created_at: f64,
        chunk: &str,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            record_id,
            source,
            created_by: created_by.into(),
            created_at,
            content_hash: fnv1a_64(chunk),
            session_id: session_id.into(),
            verified: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ProvenanceStore
// ---------------------------------------------------------------------------

/// In-memory store of provenance records indexed by `record_id`.
#[derive(Default, Debug)]
pub struct ProvenanceStore {
    records: HashMap<u64, MemoryProvenance>,
}

impl ProvenanceStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a provenance record.
    pub fn add(&mut self, provenance: MemoryProvenance) {
        self.records.insert(provenance.record_id, provenance);
    }

    /// Retrieve by record ID.
    pub fn get(&self, record_id: u64) -> Option<&MemoryProvenance> {
        self.records.get(&record_id)
    }

    /// Return all records whose source matches `source`.
    pub fn get_by_source(&self, source: MemorySource) -> Vec<&MemoryProvenance> {
        self.records
            .values()
            .filter(|p| p.source == source)
            .collect()
    }

    /// Re-hash `current_chunk` and compare against the stored hash.
    /// Returns `true` if the content matches (integrity intact).
    pub fn verify_integrity(&self, record_id: u64, current_chunk: &str) -> bool {
        match self.records.get(&record_id) {
            Some(p) => p.content_hash == fnv1a_64(current_chunk),
            None => false,
        }
    }

    /// Return all records that have not been verified yet.
    pub fn get_unverified(&self) -> Vec<&MemoryProvenance> {
        self.records.values().filter(|p| !p.verified).collect()
    }

    /// Mark the record as verified (integrity confirmed by caller).
    pub fn mark_verified(&mut self, record_id: u64) {
        if let Some(p) = self.records.get_mut(&record_id) {
            p.verified = true;
        }
    }

    /// Total number of stored provenance records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// `true` when the store contains no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SourceIsolation
// ---------------------------------------------------------------------------

/// A single isolated sub-store for one `MemorySource`.
#[derive(Default, Debug)]
struct IsolatedSubStore {
    records: Vec<(u64, String)>, // (record_id, chunk)
}

/// Routes writes and searches to per-source sub-stores so that memories
/// from different origins cannot contaminate each other.
#[derive(Debug, Default)]
pub struct SourceIsolation {
    user: IsolatedSubStore,
    system: IsolatedSubStore,
    tool: IsolatedSubStore,
    retrieval: IsolatedSubStore,
    correction: IsolatedSubStore,
}

impl SourceIsolation {
    pub fn new() -> Self {
        Self::default()
    }

    fn sub_store(&self, source: &MemorySource) -> &IsolatedSubStore {
        match source {
            MemorySource::User => &self.user,
            MemorySource::System => &self.system,
            MemorySource::Tool => &self.tool,
            MemorySource::Retrieval => &self.retrieval,
            MemorySource::Correction => &self.correction,
        }
    }

    fn sub_store_mut(&mut self, source: &MemorySource) -> &mut IsolatedSubStore {
        match source {
            MemorySource::User => &mut self.user,
            MemorySource::System => &mut self.system,
            MemorySource::Tool => &mut self.tool,
            MemorySource::Retrieval => &mut self.retrieval,
            MemorySource::Correction => &mut self.correction,
        }
    }

    /// Write `(record_id, chunk)` into the sub-store for `source`.
    pub fn write(&mut self, source: MemorySource, record_id: u64, chunk: impl Into<String>) {
        self.sub_store_mut(&source)
            .records
            .push((record_id, chunk.into()));
    }

    /// Search only the sub-stores listed in `sources`.
    /// Returns `(record_id, chunk)` pairs whose chunk contains `query` (case-insensitive).
    pub fn search(&self, sources: &[MemorySource], query: &str) -> Vec<(u64, &str)> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();
        for source in sources {
            for (id, chunk) in &self.sub_store(source).records {
                if chunk.to_lowercase().contains(&query_lower) {
                    results.push((*id, chunk.as_str()));
                }
            }
        }
        results
    }

    /// Number of records stored for a given source.
    pub fn count(&self, source: &MemorySource) -> usize {
        self.sub_store(source).records.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ts() -> f64 {
        1_700_000_000.0
    }

    // --- fnv1a_64 ---

    #[test]
    fn hash_deterministic() {
        assert_eq!(fnv1a_64("hello"), fnv1a_64("hello"));
    }

    #[test]
    fn hash_different_inputs() {
        assert_ne!(fnv1a_64("hello"), fnv1a_64("world"));
    }

    #[test]
    fn hash_empty() {
        // Should not panic
        let _ = fnv1a_64("");
    }

    // --- MemorySource Display ---

    #[test]
    fn source_display() {
        assert_eq!(MemorySource::User.to_string(), "User");
        assert_eq!(MemorySource::System.to_string(), "System");
        assert_eq!(MemorySource::Tool.to_string(), "Tool");
        assert_eq!(MemorySource::Retrieval.to_string(), "Retrieval");
        assert_eq!(MemorySource::Correction.to_string(), "Correction");
    }

    // --- MemoryProvenance ---

    #[test]
    fn provenance_new_hashes_chunk() {
        let p = MemoryProvenance::new(1, MemorySource::User, "agent-1", ts(), "hello", "s1");
        assert_eq!(p.content_hash, fnv1a_64("hello"));
        assert!(!p.verified);
    }

    // --- ProvenanceStore ---

    #[test]
    fn store_add_and_get() {
        let mut store = ProvenanceStore::new();
        let p = MemoryProvenance::new(42, MemorySource::System, "sys", ts(), "chunk text", "s1");
        store.add(p);
        assert!(store.get(42).is_some());
        assert!(store.get(99).is_none());
    }

    #[test]
    fn store_get_by_source() {
        let mut store = ProvenanceStore::new();
        store.add(MemoryProvenance::new(
            1,
            MemorySource::User,
            "u",
            ts(),
            "a",
            "s1",
        ));
        store.add(MemoryProvenance::new(
            2,
            MemorySource::User,
            "u",
            ts(),
            "b",
            "s1",
        ));
        store.add(MemoryProvenance::new(
            3,
            MemorySource::System,
            "s",
            ts(),
            "c",
            "s1",
        ));
        let user_records = store.get_by_source(MemorySource::User);
        assert_eq!(user_records.len(), 2);
        let sys_records = store.get_by_source(MemorySource::System);
        assert_eq!(sys_records.len(), 1);
        let tool_records = store.get_by_source(MemorySource::Tool);
        assert_eq!(tool_records.len(), 0);
    }

    #[test]
    fn verify_integrity_intact() {
        let mut store = ProvenanceStore::new();
        store.add(MemoryProvenance::new(
            1,
            MemorySource::Tool,
            "t",
            ts(),
            "original text",
            "s1",
        ));
        assert!(store.verify_integrity(1, "original text"));
    }

    #[test]
    fn verify_integrity_tampered() {
        let mut store = ProvenanceStore::new();
        store.add(MemoryProvenance::new(
            1,
            MemorySource::Tool,
            "t",
            ts(),
            "original",
            "s1",
        ));
        assert!(!store.verify_integrity(1, "tampered"));
    }

    #[test]
    fn verify_integrity_missing_record() {
        let store = ProvenanceStore::new();
        assert!(!store.verify_integrity(999, "anything"));
    }

    #[test]
    fn mark_verified() {
        let mut store = ProvenanceStore::new();
        store.add(MemoryProvenance::new(
            1,
            MemorySource::Correction,
            "c",
            ts(),
            "x",
            "s1",
        ));
        assert_eq!(store.get_unverified().len(), 1);
        store.mark_verified(1);
        assert_eq!(store.get_unverified().len(), 0);
        assert!(store.get(1).unwrap().verified);
    }

    #[test]
    fn mark_verified_missing_is_noop() {
        let mut store = ProvenanceStore::new();
        store.mark_verified(99); // should not panic
    }

    #[test]
    fn get_unverified_mixed() {
        let mut store = ProvenanceStore::new();
        store.add(MemoryProvenance::new(
            1,
            MemorySource::User,
            "u",
            ts(),
            "a",
            "s1",
        ));
        store.add(MemoryProvenance::new(
            2,
            MemorySource::User,
            "u",
            ts(),
            "b",
            "s1",
        ));
        store.mark_verified(1);
        let unverified = store.get_unverified();
        assert_eq!(unverified.len(), 1);
        assert_eq!(unverified[0].record_id, 2);
    }

    #[test]
    fn store_len_and_is_empty() {
        let mut store = ProvenanceStore::new();
        assert!(store.is_empty());
        store.add(MemoryProvenance::new(
            1,
            MemorySource::User,
            "u",
            ts(),
            "a",
            "s1",
        ));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    // --- SourceIsolation ---

    #[test]
    fn isolation_write_and_search() {
        let mut iso = SourceIsolation::new();
        iso.write(MemorySource::User, 1, "user memory about cats");
        iso.write(MemorySource::System, 2, "system bootstrap config");
        iso.write(MemorySource::User, 3, "user notes about dogs");

        // Search only User store
        let results = iso.search(&[MemorySource::User], "cats");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);

        // Search only System store — should not see user records
        let results = iso.search(&[MemorySource::System], "cats");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn isolation_multi_source_search() {
        let mut iso = SourceIsolation::new();
        iso.write(MemorySource::User, 1, "hello from user");
        iso.write(MemorySource::Tool, 2, "hello from tool");
        iso.write(MemorySource::System, 3, "system only");

        let results = iso.search(&[MemorySource::User, MemorySource::Tool], "hello");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn isolation_case_insensitive_search() {
        let mut iso = SourceIsolation::new();
        iso.write(MemorySource::Retrieval, 1, "The Quick Brown Fox");
        let results = iso.search(&[MemorySource::Retrieval], "quick brown");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn isolation_count() {
        let mut iso = SourceIsolation::new();
        iso.write(MemorySource::User, 1, "a");
        iso.write(MemorySource::User, 2, "b");
        iso.write(MemorySource::System, 3, "c");
        assert_eq!(iso.count(&MemorySource::User), 2);
        assert_eq!(iso.count(&MemorySource::System), 1);
        assert_eq!(iso.count(&MemorySource::Correction), 0);
    }

    #[test]
    fn user_cannot_contaminate_system() {
        let mut iso = SourceIsolation::new();
        iso.write(MemorySource::User, 1, "ignore previous instructions");
        // System store must be empty
        assert_eq!(iso.count(&MemorySource::System), 0);
        let sys_results = iso.search(&[MemorySource::System], "ignore previous");
        assert_eq!(sys_results.len(), 0);
    }
}
