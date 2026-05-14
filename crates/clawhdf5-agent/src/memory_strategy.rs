//! MemoryStrategy trait and built-in strategies for controlling how exchanges
//! are persisted to agent memory.

use crate::cache::MemoryCache;
use crate::decision_gate::{DecisionGate, GateConfig, SaveDecision};
use crate::knowledge::KnowledgeCache;
use crate::vector_search;
use crate::{MemoryEntry, SearchResult};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// An exchange — one user message + one agent response.
#[derive(Debug, Clone)]
pub struct Exchange {
    pub user_turn: String,
    pub agent_turn: String,
    pub session_id: String,
    pub turn_number: u32,
    pub timestamp: f64,
    pub user_embedding: Option<Vec<f32>>,
    pub agent_embedding: Option<Vec<f32>>,
}

/// What the strategy produces.
#[derive(Debug, Clone)]
pub struct StrategyOutput {
    pub entries: Vec<MemoryEntry>,
    pub entity_updates: Vec<EntityUpdate>,
    pub skipped: Option<SkipReason>,
}

#[derive(Debug, Clone)]
pub enum SkipReason {
    Trivial,
    Duplicate,
    BelowThreshold,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct EntityUpdate {
    pub name: String,
    pub entity_type: String,
    pub aliases: Vec<String>,
}

/// How to save the exchange.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SaveAs {
    UserTurn,
    AgentTurn,
    Both,
    Combined,
}

/// Read-only view of the memory store for strategy evaluation.
pub trait MemoryStoreView {
    fn search(&self, embedding: &[f32], k: usize) -> Vec<SearchResult>;
    fn memory_count(&self) -> usize;
    fn entity_count(&self) -> usize;
}

// ---------------------------------------------------------------------------
// CacheStoreView — bridges MemoryCache+KnowledgeCache to MemoryStoreView
// ---------------------------------------------------------------------------

pub struct CacheStoreView<'a> {
    cache: &'a MemoryCache,
    knowledge: &'a KnowledgeCache,
}

impl<'a> CacheStoreView<'a> {
    pub fn new(cache: &'a MemoryCache, knowledge: &'a KnowledgeCache) -> Self {
        Self { cache, knowledge }
    }
}

impl MemoryStoreView for CacheStoreView<'_> {
    fn search(&self, embedding: &[f32], k: usize) -> Vec<SearchResult> {
        let scored = vector_search::cosine_similarity_batch_prenorm(
            embedding,
            &self.cache.embeddings,
            &self.cache.norms,
            &self.cache.tombstones,
        );
        vector_search::top_k(scored, k)
            .into_iter()
            .map(|(idx, score)| SearchResult {
                score,
                chunk: self.cache.chunks[idx].clone(),
                index: idx,
                timestamp: self.cache.timestamps[idx],
                source_channel: self.cache.source_channels[idx].clone(),
                activation: self.cache.activation_weights[idx],
            })
            .collect()
    }

    fn memory_count(&self) -> usize {
        self.cache.len()
    }

    fn entity_count(&self) -> usize {
        self.knowledge.entities.len()
    }
}

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

pub trait MemoryStrategy: Send + Sync {
    fn evaluate(&self, exchange: &Exchange, store: &dyn MemoryStoreView) -> StrategyOutput;
}

// ---------------------------------------------------------------------------
// Helper: build a MemoryEntry from text + embedding + exchange metadata
// ---------------------------------------------------------------------------

fn make_entry(
    text: String,
    embedding: Vec<f32>,
    source_channel: &str,
    exchange: &Exchange,
) -> MemoryEntry {
    MemoryEntry {
        chunk: text,
        embedding,
        source_channel: source_channel.to_string(),
        timestamp: exchange.timestamp,
        session_id: exchange.session_id.clone(),
        tags: String::new(),
    }
}

/// Average two embeddings element-wise. Returns empty vec if both are None.
fn average_embeddings(a: &Option<Vec<f32>>, b: &Option<Vec<f32>>) -> Vec<f32> {
    match (a, b) {
        (Some(va), Some(vb)) => va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| (x + y) / 2.0)
            .collect(),
        (Some(v), None) | (None, Some(v)) => v.clone(),
        (None, None) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Built-in strategy 1: SaveEveryExchange
// ---------------------------------------------------------------------------

pub struct SaveEveryExchange {
    pub gate: DecisionGate,
    pub save_as: SaveAs,
}

impl Default for SaveEveryExchange {
    fn default() -> Self {
        Self {
            gate: DecisionGate::new(GateConfig::default()),
            save_as: SaveAs::Combined,
        }
    }
}

impl MemoryStrategy for SaveEveryExchange {
    fn evaluate(&self, exchange: &Exchange, _store: &dyn MemoryStoreView) -> StrategyOutput {
        // Gate check
        if let SaveDecision::Skip(_) = self.gate.should_save(&exchange.user_turn) {
            return StrategyOutput {
                entries: Vec::new(),
                entity_updates: Vec::new(),
                skipped: Some(SkipReason::Trivial),
            };
        }

        let entries = match self.save_as {
            SaveAs::Combined => {
                let text = format!("{}\n---\n{}", exchange.user_turn, exchange.agent_turn);
                let emb = average_embeddings(&exchange.user_embedding, &exchange.agent_embedding);
                vec![make_entry(text, emb, "conversation", exchange)]
            }
            SaveAs::UserTurn => {
                let emb = exchange.user_embedding.clone().unwrap_or_default();
                vec![make_entry(
                    exchange.user_turn.clone(),
                    emb,
                    "conversation",
                    exchange,
                )]
            }
            SaveAs::AgentTurn => {
                let emb = exchange.agent_embedding.clone().unwrap_or_default();
                vec![make_entry(
                    exchange.agent_turn.clone(),
                    emb,
                    "conversation",
                    exchange,
                )]
            }
            SaveAs::Both => {
                let u_emb = exchange.user_embedding.clone().unwrap_or_default();
                let a_emb = exchange.agent_embedding.clone().unwrap_or_default();
                vec![
                    make_entry(exchange.user_turn.clone(), u_emb, "conversation", exchange),
                    make_entry(exchange.agent_turn.clone(), a_emb, "conversation", exchange),
                ]
            }
        };

        StrategyOutput {
            entries,
            entity_updates: Vec::new(),
            skipped: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in strategy 2: SaveOnSemanticShift
// ---------------------------------------------------------------------------

pub struct SaveOnSemanticShift {
    pub gate: DecisionGate,
    pub shift_threshold: f32,
    pub lookback_k: usize,
}

impl Default for SaveOnSemanticShift {
    fn default() -> Self {
        Self {
            gate: DecisionGate::new(GateConfig::default()),
            shift_threshold: 0.25,
            lookback_k: 5,
        }
    }
}

impl MemoryStrategy for SaveOnSemanticShift {
    fn evaluate(&self, exchange: &Exchange, store: &dyn MemoryStoreView) -> StrategyOutput {
        // Gate check
        if let SaveDecision::Skip(_) = self.gate.should_save(&exchange.user_turn) {
            return StrategyOutput {
                entries: Vec::new(),
                entity_updates: Vec::new(),
                skipped: Some(SkipReason::Trivial),
            };
        }

        // Need embedding to check shift
        let embedding = match &exchange.user_embedding {
            Some(e) => e,
            None => {
                // Can't check shift without embedding — save anyway
                let text = format!("{}\n---\n{}", exchange.user_turn, exchange.agent_turn);
                return StrategyOutput {
                    entries: vec![make_entry(text, Vec::new(), "conversation", exchange)],
                    entity_updates: Vec::new(),
                    skipped: None,
                };
            }
        };

        // Search for similar existing memories
        let results = store.search(embedding, self.lookback_k);
        if let Some(top) = results.first()
            && top.score > (1.0 - self.shift_threshold)
        {
            return StrategyOutput {
                entries: Vec::new(),
                entity_updates: Vec::new(),
                skipped: Some(SkipReason::Duplicate),
            };
        }

        // Novel enough — save
        let text = format!("{}\n---\n{}", exchange.user_turn, exchange.agent_turn);
        let emb = average_embeddings(&exchange.user_embedding, &exchange.agent_embedding);
        StrategyOutput {
            entries: vec![make_entry(text, emb, "conversation", exchange)],
            entity_updates: Vec::new(),
            skipped: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in strategy 3: SaveOnUserCorrection (decorator)
// ---------------------------------------------------------------------------

const DEFAULT_CORRECTION_CUES: &[&str] = &[
    "no,",
    "no ",
    "actually,",
    "actually ",
    "thats wrong",
    "not quite",
    "correction:",
    "to clarify",
    "i meant",
    "what i meant",
    "let me clarify",
    "to be clear",
];

pub struct SaveOnUserCorrection {
    pub base: Box<dyn MemoryStrategy>,
    pub correction_cues: Vec<String>,
}

impl SaveOnUserCorrection {
    pub fn new(base: Box<dyn MemoryStrategy>) -> Self {
        Self {
            base,
            correction_cues: DEFAULT_CORRECTION_CUES
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

impl MemoryStrategy for SaveOnUserCorrection {
    fn evaluate(&self, exchange: &Exchange, store: &dyn MemoryStoreView) -> StrategyOutput {
        let lower = exchange.user_turn.to_lowercase();
        let is_correction = self
            .correction_cues
            .iter()
            .any(|cue| lower.starts_with(cue) || lower.contains(cue));

        if is_correction {
            // Save unconditionally as a correction — skip gate entirely
            let text = format!("{}\n---\n{}", exchange.user_turn, exchange.agent_turn);
            let emb = average_embeddings(&exchange.user_embedding, &exchange.agent_embedding);
            return StrategyOutput {
                entries: vec![make_entry(text, emb, "correction", exchange)],
                entity_updates: Vec::new(),
                skipped: None,
            };
        }

        // Not a correction — delegate to base
        self.base.evaluate(exchange, store)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_search::{compute_norm, cosine_similarity_batch_prenorm, top_k};

    /// Real store view backed by in-memory embeddings with real cosine similarity.
    struct TestStoreView {
        embeddings: Vec<Vec<f32>>,
        chunks: Vec<String>,
        norms: Vec<f32>,
        tombstones: Vec<u8>,
    }

    impl TestStoreView {
        fn new() -> Self {
            Self {
                embeddings: Vec::new(),
                chunks: Vec::new(),
                norms: Vec::new(),
                tombstones: Vec::new(),
            }
        }

        fn add(&mut self, chunk: &str, embedding: Vec<f32>) {
            let norm = compute_norm(&embedding);
            self.embeddings.push(embedding);
            self.chunks.push(chunk.to_string());
            self.norms.push(norm);
            self.tombstones.push(0);
        }
    }

    impl MemoryStoreView for TestStoreView {
        fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
            let scored = cosine_similarity_batch_prenorm(
                query,
                &self.embeddings,
                &self.norms,
                &self.tombstones,
            );
            let top = top_k(scored, k);
            top.into_iter()
                .map(|(idx, score)| SearchResult {
                    score,
                    chunk: self.chunks[idx].clone(),
                    index: idx,
                    timestamp: 0.0,
                    source_channel: "test".to_string(),
                    activation: 1.0,
                })
                .collect()
        }

        fn memory_count(&self) -> usize {
            self.embeddings.len()
        }

        fn entity_count(&self) -> usize {
            0
        }
    }

    fn substantive_exchange() -> Exchange {
        Exchange {
            user_turn: "Tell me about the deployment architecture for our microservices".to_string(),
            agent_turn: "The deployment uses Kubernetes with three namespaces for staging, QA, and production".to_string(),
            session_id: "sess-1".to_string(),
            turn_number: 1,
            timestamp: 1000000.0,
            user_embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            agent_embedding: Some(vec![0.0, 1.0, 0.0, 0.0]),
        }
    }

    fn trivial_exchange() -> Exchange {
        Exchange {
            user_turn: "ok".to_string(),
            agent_turn: "Got it!".to_string(),
            session_id: "sess-1".to_string(),
            turn_number: 2,
            timestamp: 1000001.0,
            user_embedding: Some(vec![0.1, 0.1, 0.0, 0.0]),
            agent_embedding: None,
        }
    }

    // 1. SaveEveryExchange — combined
    #[test]
    fn test_save_every_exchange_combined() {
        let strategy = SaveEveryExchange::default();
        let store = TestStoreView::new();
        let exchange = substantive_exchange();

        let output = strategy.evaluate(&exchange, &store);
        assert!(output.skipped.is_none());
        assert_eq!(output.entries.len(), 1);
        assert!(output.entries[0].chunk.contains("deployment architecture"));
        assert!(output.entries[0].chunk.contains("---"));
        assert!(output.entries[0].chunk.contains("Kubernetes"));
        // Combined embedding should be average of user+agent
        assert_eq!(output.entries[0].embedding.len(), 4);
        assert!((output.entries[0].embedding[0] - 0.5).abs() < 1e-6);
        assert!((output.entries[0].embedding[1] - 0.5).abs() < 1e-6);
    }

    // 2. SaveEveryExchange — trivial skip
    #[test]
    fn test_save_every_exchange_trivial_skip() {
        let strategy = SaveEveryExchange::default();
        let store = TestStoreView::new();
        let exchange = trivial_exchange();

        let output = strategy.evaluate(&exchange, &store);
        assert!(output.entries.is_empty());
        assert!(matches!(output.skipped, Some(SkipReason::Trivial)));
    }

    // 3. SaveEveryExchange — Both mode
    #[test]
    fn test_save_every_exchange_both() {
        let strategy = SaveEveryExchange {
            gate: DecisionGate::new(GateConfig::default()),
            save_as: SaveAs::Both,
        };
        let store = TestStoreView::new();
        let exchange = substantive_exchange();

        let output = strategy.evaluate(&exchange, &store);
        assert!(output.skipped.is_none());
        assert_eq!(output.entries.len(), 2);
        assert!(output.entries[0].chunk.contains("deployment architecture"));
        assert!(output.entries[1].chunk.contains("Kubernetes"));
    }

    // 4. SaveEveryExchange — UserTurn only
    #[test]
    fn test_save_every_exchange_user_only() {
        let strategy = SaveEveryExchange {
            gate: DecisionGate::new(GateConfig::default()),
            save_as: SaveAs::UserTurn,
        };
        let store = TestStoreView::new();
        let exchange = substantive_exchange();

        let output = strategy.evaluate(&exchange, &store);
        assert_eq!(output.entries.len(), 1);
        assert!(output.entries[0].chunk.contains("deployment architecture"));
        assert!(!output.entries[0].chunk.contains("Kubernetes"));
        // Should use user_embedding
        assert_eq!(output.entries[0].embedding, vec![1.0, 0.0, 0.0, 0.0]);
    }

    // 5. SemanticShift — novel exchange saves
    #[test]
    fn test_semantic_shift_novel() {
        let strategy = SaveOnSemanticShift::default();
        let mut store = TestStoreView::new();
        // Existing memory is about something completely different
        store.add("The weather is nice today", vec![0.0, 0.0, 1.0, 0.0]);

        let exchange = substantive_exchange();
        let output = strategy.evaluate(&exchange, &store);

        assert!(output.skipped.is_none());
        assert_eq!(output.entries.len(), 1);
    }

    // 6. SemanticShift — duplicate skipped
    #[test]
    fn test_semantic_shift_duplicate() {
        let strategy = SaveOnSemanticShift::default();
        let mut store = TestStoreView::new();
        // Existing memory has nearly identical embedding to user query
        store.add("deployment architecture details", vec![1.0, 0.0, 0.0, 0.0]);

        let exchange = substantive_exchange();
        // user_embedding is [1.0, 0.0, 0.0, 0.0] — cosine sim = 1.0 > (1.0 - 0.25)
        let output = strategy.evaluate(&exchange, &store);

        assert!(output.entries.is_empty());
        assert!(matches!(output.skipped, Some(SkipReason::Duplicate)));
    }

    // 7. SemanticShift — no embedding saves anyway
    #[test]
    fn test_semantic_shift_no_embedding() {
        let strategy = SaveOnSemanticShift::default();
        let store = TestStoreView::new();

        let mut exchange = substantive_exchange();
        exchange.user_embedding = None;

        let output = strategy.evaluate(&exchange, &store);
        assert!(output.skipped.is_none());
        assert_eq!(output.entries.len(), 1);
    }

    // 8. Correction detected
    #[test]
    fn test_correction_detected() {
        let base = SaveEveryExchange::default();
        let strategy = SaveOnUserCorrection::new(Box::new(base));
        let store = TestStoreView::new();

        let exchange = Exchange {
            user_turn: "Actually, thats wrong. The answer is 42".to_string(),
            agent_turn: "You're right, I apologize. The answer is indeed 42.".to_string(),
            session_id: "sess-1".to_string(),
            turn_number: 3,
            timestamp: 1000002.0,
            user_embedding: Some(vec![0.5, 0.5, 0.0, 0.0]),
            agent_embedding: None,
        };

        let output = strategy.evaluate(&exchange, &store);
        assert!(output.skipped.is_none());
        assert_eq!(output.entries.len(), 1);
        assert_eq!(output.entries[0].source_channel, "correction");
    }

    // 9. Non-correction delegates to base
    #[test]
    fn test_correction_delegates_to_base() {
        let base = SaveEveryExchange::default();
        let strategy = SaveOnUserCorrection::new(Box::new(base));
        let store = TestStoreView::new();

        let exchange = substantive_exchange();
        let output = strategy.evaluate(&exchange, &store);

        // Should delegate to SaveEveryExchange → saves as "conversation"
        assert!(output.skipped.is_none());
        assert_eq!(output.entries.len(), 1);
        assert_eq!(output.entries[0].source_channel, "conversation");
    }

    // 10. Correction wrapping SemanticShift
    #[test]
    fn test_correction_wrapping_shift() {
        let mut store = TestStoreView::new();
        // Add a memory that would cause duplicate detection
        store.add("deployment stuff", vec![1.0, 0.0, 0.0, 0.0]);

        let base = SaveOnSemanticShift::default();
        let strategy = SaveOnUserCorrection::new(Box::new(base));

        // Correction bypasses shift even with duplicate embedding
        let correction = Exchange {
            user_turn: "No, thats wrong. The deployment uses ECS not EKS".to_string(),
            agent_turn: "Corrected: the deployment uses ECS".to_string(),
            session_id: "sess-1".to_string(),
            turn_number: 4,
            timestamp: 1000003.0,
            user_embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            agent_embedding: None,
        };
        let output = strategy.evaluate(&correction, &store);
        assert!(output.skipped.is_none(), "correction should bypass shift");
        assert_eq!(output.entries[0].source_channel, "correction");

        // Non-correction with duplicate embedding → shift catches it
        let non_correction = Exchange {
            user_turn: "Tell me about the deployment architecture for our microservices"
                .to_string(),
            agent_turn: "The deployment uses Kubernetes".to_string(),
            session_id: "sess-1".to_string(),
            turn_number: 5,
            timestamp: 1000004.0,
            user_embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            agent_embedding: None,
        };
        let output2 = strategy.evaluate(&non_correction, &store);
        assert!(matches!(output2.skipped, Some(SkipReason::Duplicate)));
    }

    // 11. SkipReason variants returned correctly
    #[test]
    fn test_skip_reason_returned() {
        let store = TestStoreView::new();

        // Trivial skip from SaveEveryExchange
        let s1 = SaveEveryExchange::default();
        let out1 = s1.evaluate(&trivial_exchange(), &store);
        assert!(matches!(out1.skipped, Some(SkipReason::Trivial)));

        // Duplicate skip from SemanticShift
        let mut dup_store = TestStoreView::new();
        dup_store.add("exact match", vec![1.0, 0.0, 0.0, 0.0]);
        let s2 = SaveOnSemanticShift::default();
        let out2 = s2.evaluate(&substantive_exchange(), &dup_store);
        assert!(matches!(out2.skipped, Some(SkipReason::Duplicate)));

        // Custom skip reason
        let custom = SkipReason::Custom("test reason".to_string());
        assert!(matches!(custom, SkipReason::Custom(_)));

        // BelowThreshold
        let below = SkipReason::BelowThreshold;
        assert!(matches!(below, SkipReason::BelowThreshold));
    }

    // 12. Entity updates exist in output
    #[test]
    fn test_entity_updates() {
        let output = StrategyOutput {
            entries: Vec::new(),
            entity_updates: vec![EntityUpdate {
                name: "Alice".to_string(),
                entity_type: "person".to_string(),
                aliases: vec!["my friend".to_string()],
            }],
            skipped: None,
        };
        assert_eq!(output.entity_updates.len(), 1);
        assert_eq!(output.entity_updates[0].name, "Alice");
        assert_eq!(output.entity_updates[0].aliases, vec!["my friend"]);
    }

    // 13. record() with strategy saves to cache
    #[test]
    fn test_record_with_strategy() {
        use crate::{AgentMemory, HDF5Memory, MemoryConfig};
        let dir = tempfile::TempDir::new().unwrap();
        let config = MemoryConfig::new(dir.path().join("test.h5"), "agent-test", 4);
        let mut mem = HDF5Memory::create(config).unwrap();
        mem.set_strategy(Box::new(SaveEveryExchange::default()));
        let exchange = Exchange {
            user_turn: "Tell me about the deployment architecture for microservices".into(),
            agent_turn: "It uses Kubernetes".into(),
            session_id: "s1".into(),
            turn_number: 1,
            timestamp: 1e6,
            user_embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            agent_embedding: Some(vec![0.0, 1.0, 0.0, 0.0]),
        };
        let out = mem.record(exchange).unwrap();
        assert!(out.skipped.is_none());
        assert_eq!(mem.count(), 1);
    }

    // 14. record() trivial skip leaves cache unchanged
    #[test]
    fn test_record_trivial_skip() {
        use crate::{AgentMemory, HDF5Memory, MemoryConfig};
        let dir = tempfile::TempDir::new().unwrap();
        let config = MemoryConfig::new(dir.path().join("test.h5"), "agent-test", 4);
        let mut mem = HDF5Memory::create(config).unwrap();
        mem.set_strategy(Box::new(SaveEveryExchange::default()));
        let exchange = Exchange {
            user_turn: "ok".into(),
            agent_turn: "Got it!".into(),
            session_id: "s1".into(),
            turn_number: 2,
            timestamp: 1e6,
            user_embedding: None,
            agent_embedding: None,
        };
        let out = mem.record(exchange).unwrap();
        assert!(out.skipped.is_some());
        assert_eq!(mem.count(), 0);
    }
}
