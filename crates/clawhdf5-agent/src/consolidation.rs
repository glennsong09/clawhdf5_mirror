//! Hippocampal-inspired memory consolidation system.
//!
//! Models Working → Episodic → Semantic memory tiers with importance scoring,
//! exponential decay, and capacity-based eviction.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum MemorySource {
    User,
    System,
    Tool,
    Retrieval,
    Correction,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MemoryTier {
    Working,
    Episodic,
    Semantic,
}

#[derive(Clone, Debug)]
pub struct MemoryRecord {
    pub id: u64,
    pub chunk: String,
    pub embedding: Vec<f32>,
    pub tier: MemoryTier,
    pub importance: f32,
    pub access_count: u32,
    pub last_accessed: f64,
    pub created_at: f64,
    pub source: MemorySource,
}

#[derive(Clone, Debug, Copy)]
pub struct ImportanceWeights {
    pub surprise: f32,
    pub correction: f32,
    pub length: f32,
}

impl Default for ImportanceWeights {
    fn default() -> Self {
        Self {
            surprise: 0.5,
            correction: 0.3,
            length: 0.2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConsolidationConfig {
    pub working_capacity: usize,
    pub episodic_capacity: usize,
    pub episodic_lambda: f64,
    pub semantic_lambda: f64,
    pub working_to_episodic_threshold: f32,
    pub episodic_to_semantic_threshold: u32,
    pub importance_weights: ImportanceWeights,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            working_capacity: 100,
            episodic_capacity: 10_000,
            // ln(2) / 604800  → half-life 7 days in seconds
            episodic_lambda: std::f64::consts::LN_2 / 604_800.0,
            // ln(2) / 2592000 → half-life 30 days in seconds
            semantic_lambda: std::f64::consts::LN_2 / 2_592_000.0,
            working_to_episodic_threshold: 0.6,
            episodic_to_semantic_threshold: 10,
            importance_weights: ImportanceWeights::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConsolidationStats {
    pub working_count: usize,
    pub episodic_count: usize,
    pub semantic_count: usize,
    pub total_evictions: u64,
    pub total_promotions: u64,
}

// ---------------------------------------------------------------------------
// ImportanceScorer
// ---------------------------------------------------------------------------

pub struct ImportanceScorer;

impl ImportanceScorer {
    /// Cosine similarity between two embedding slices.
    /// Returns 0.0 if either norm is zero.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 0.0;
        }
        let dot: f32 = a[..len]
            .iter()
            .zip(b[..len].iter())
            .map(|(x, y)| x * y)
            .sum();
        let norm_a: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Novelty score: 1.0 − max cosine similarity against all existing records.
    /// Returns 1.0 when there are no existing memories.
    pub fn score_surprise(embedding: &[f32], existing_memories: &[MemoryRecord]) -> f32 {
        if existing_memories.is_empty() {
            return 1.0;
        }
        let max_sim = existing_memories
            .iter()
            .map(|r| Self::cosine_similarity(embedding, &r.embedding))
            .fold(f32::NEG_INFINITY, f32::max);
        (1.0 - max_sim).clamp(0.0, 1.0)
    }

    /// Returns 1.0 for Correction source, 0.0 otherwise.
    pub fn score_correction(source: &MemorySource) -> f32 {
        if *source == MemorySource::Correction {
            1.0
        } else {
            0.0
        }
    }

    /// Normalised word-count score, clamped at 1.0 (ceiling = 100 words).
    pub fn score_length(text: &str) -> f32 {
        let word_count = text.split_whitespace().count();
        (word_count as f32 / 100.0).min(1.0)
    }

    /// Weighted combination of sub-scores, normalised by weight total.
    pub fn score_combined(
        surprise: f32,
        correction: f32,
        length: f32,
        weights: &ImportanceWeights,
    ) -> f32 {
        let weight_total = weights.surprise + weights.correction + weights.length;
        if weight_total == 0.0 {
            return 0.0;
        }
        let weighted_sum =
            surprise * weights.surprise + correction * weights.correction + length * weights.length;
        (weighted_sum / weight_total).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// DecayCalculator
// ---------------------------------------------------------------------------

pub struct DecayCalculator;

impl DecayCalculator {
    /// Exponential decay score for a record.
    ///
    /// decay = importance × (access_count + 1) × e^(−λ × elapsed)
    pub fn compute_decay(record: &MemoryRecord, now: f64, lambda: f64) -> f32 {
        let elapsed = (now - record.last_accessed).max(0.0);
        let decay_factor = f64::exp(-lambda * elapsed);
        record.importance * (record.access_count + 1) as f32 * decay_factor as f32
    }
}

// ---------------------------------------------------------------------------
// ConsolidationEngine
// ---------------------------------------------------------------------------

pub struct ConsolidationEngine {
    pub config: ConsolidationConfig,
    pub records: Vec<MemoryRecord>,
    pub next_id: u64,
    pub stats: ConsolidationStats,
}

impl ConsolidationEngine {
    pub fn new(config: ConsolidationConfig) -> Self {
        Self {
            config,
            records: Vec::new(),
            next_id: 0,
            stats: ConsolidationStats::default(),
        }
    }

    /// Add a new memory to the Working tier.
    ///
    /// Importance is scored against existing Working-tier records only.
    pub fn add_memory(
        &mut self,
        chunk: String,
        embedding: Vec<f32>,
        source: MemorySource,
        now: f64,
    ) -> u64 {
        let working: Vec<MemoryRecord> = self
            .records
            .iter()
            .filter(|r| r.tier == MemoryTier::Working)
            .cloned()
            .collect();

        let surprise = ImportanceScorer::score_surprise(&embedding, &working);
        let correction = ImportanceScorer::score_correction(&source);
        let length = ImportanceScorer::score_length(&chunk);
        let importance = ImportanceScorer::score_combined(
            surprise,
            correction,
            length,
            &self.config.importance_weights,
        );

        let id = self.next_id;
        self.next_id += 1;

        self.records.push(MemoryRecord {
            id,
            chunk,
            embedding,
            tier: MemoryTier::Working,
            importance,
            access_count: 0,
            last_accessed: now,
            created_at: now,
            source,
        });

        id
    }

    /// Increment access count and update last-accessed timestamp for a record.
    pub fn access_memory(&mut self, id: u64, now: f64) {
        if let Some(rec) = self.records.iter_mut().find(|r| r.id == id) {
            rec.access_count += 1;
            rec.last_accessed = now;
        }
    }

    /// Run one full consolidation cycle.
    pub fn consolidate(&mut self, now: f64) {
        // ------------------------------------------------------------------
        // Step 1 — Compute decay scores for Working records; sort ascending.
        // ------------------------------------------------------------------
        let working_lambda = self.config.episodic_lambda; // reuse episodic lambda for working
        let mut working_indices: Vec<usize> = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.tier == MemoryTier::Working)
            .map(|(i, _)| i)
            .collect();

        working_indices.sort_by(|&a, &b| {
            let da = DecayCalculator::compute_decay(&self.records[a], now, working_lambda);
            let db = DecayCalculator::compute_decay(&self.records[b], now, working_lambda);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        // ------------------------------------------------------------------
        // Step 2 — Evict lowest-decay Working records until count ≤ capacity.
        // ------------------------------------------------------------------
        let working_count = working_indices.len();
        let capacity = self.config.working_capacity;

        if working_count > capacity {
            let evict_n = working_count - capacity;
            // Collect the ids of the records to evict (lowest decay = first in sorted list).
            let evict_ids: Vec<u64> = working_indices[..evict_n]
                .iter()
                .map(|&i| self.records[i].id)
                .collect();

            self.records.retain(|r| !evict_ids.contains(&r.id));
            self.stats.total_evictions += evict_n as u64;
        }

        // ------------------------------------------------------------------
        // Step 3 — Promote high-importance Working records → Episodic.
        // ------------------------------------------------------------------
        let threshold = self.config.working_to_episodic_threshold;
        let mut promotions: u64 = 0;

        for rec in self.records.iter_mut() {
            if rec.tier == MemoryTier::Working && rec.importance > threshold {
                rec.tier = MemoryTier::Episodic;
                promotions += 1;
            }
        }
        self.stats.total_promotions += promotions;

        // ------------------------------------------------------------------
        // Step 4 — Promote high-access Episodic records → Semantic.
        // ------------------------------------------------------------------
        let semantic_threshold = self.config.episodic_to_semantic_threshold;
        let mut sem_promotions: u64 = 0;

        for rec in self.records.iter_mut() {
            if rec.tier == MemoryTier::Episodic && rec.access_count > semantic_threshold {
                rec.tier = MemoryTier::Semantic;
                sem_promotions += 1;
            }
        }
        self.stats.total_promotions += sem_promotions;

        // ------------------------------------------------------------------
        // Step 5 — Evict lowest-decay Episodic records when over capacity.
        // ------------------------------------------------------------------
        let episodic_lambda = self.config.episodic_lambda;
        let episodic_capacity = self.config.episodic_capacity;

        let mut episodic_indices: Vec<usize> = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.tier == MemoryTier::Episodic)
            .map(|(i, _)| i)
            .collect();

        let episodic_count = episodic_indices.len();

        if episodic_count > episodic_capacity {
            episodic_indices.sort_by(|&a, &b| {
                let da = DecayCalculator::compute_decay(&self.records[a], now, episodic_lambda);
                let db = DecayCalculator::compute_decay(&self.records[b], now, episodic_lambda);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

            let evict_n = episodic_count - episodic_capacity;
            let evict_ids: Vec<u64> = episodic_indices[..evict_n]
                .iter()
                .map(|&i| self.records[i].id)
                .collect();

            self.records.retain(|r| !evict_ids.contains(&r.id));
            self.stats.total_evictions += evict_n as u64;
        }
    }

    /// Return live per-tier counts merged with running totals.
    pub fn get_stats(&self) -> ConsolidationStats {
        let mut stats = self.stats.clone();
        stats.working_count = self
            .records
            .iter()
            .filter(|r| r.tier == MemoryTier::Working)
            .count();
        stats.episodic_count = self
            .records
            .iter()
            .filter(|r| r.tier == MemoryTier::Episodic)
            .count();
        stats.semantic_count = self
            .records
            .iter()
            .filter(|r| r.tier == MemoryTier::Semantic)
            .count();
        stats
    }

    /// Slice over all records.
    pub fn records(&self) -> &[MemoryRecord] {
        &self.records
    }

    /// Look up a record by id.
    pub fn get_by_id(&self, id: u64) -> Option<&MemoryRecord> {
        self.records.iter().find(|r| r.id == id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple normalised embedding of given dimension.
    fn unit_vec(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[hot % dim] = 1.0;
        v
    }

    // ---------------------------------------------------------------------------
    // 1. Default config values
    // ---------------------------------------------------------------------------
    #[test]
    fn test_memory_tiers_default_config() {
        let cfg = ConsolidationConfig::default();
        assert_eq!(cfg.working_capacity, 100);
        assert_eq!(cfg.episodic_capacity, 10_000);
        assert!((cfg.working_to_episodic_threshold - 0.6_f32).abs() < f32::EPSILON);
        assert_eq!(cfg.episodic_to_semantic_threshold, 10);
        // Verify half-lives roughly: λ = ln2/T → T = ln2/λ
        let working_half_life = std::f64::consts::LN_2 / cfg.episodic_lambda;
        let semantic_half_life = std::f64::consts::LN_2 / cfg.semantic_lambda;
        assert!((working_half_life - 604_800.0).abs() < 1.0);
        assert!((semantic_half_life - 2_592_000.0).abs() < 1.0);
    }

    // ---------------------------------------------------------------------------
    // 2. Add memory — basic
    // ---------------------------------------------------------------------------
    #[test]
    fn test_add_memory_basic() {
        let mut engine = ConsolidationEngine::new(ConsolidationConfig::default());
        let id = engine.add_memory(
            "Hello world".to_string(),
            unit_vec(4, 0),
            MemorySource::User,
            1_000_000.0,
        );
        assert_eq!(id, 0);
        assert_eq!(engine.records().len(), 1);
        let rec = engine.get_by_id(0).unwrap();
        assert_eq!(rec.tier, MemoryTier::Working);
        assert_eq!(rec.access_count, 0);
        assert!((rec.last_accessed - 1_000_000.0_f64).abs() < f64::EPSILON);
        assert!((rec.created_at - 1_000_000.0_f64).abs() < f64::EPSILON);
        assert_eq!(rec.source, MemorySource::User);
    }

    // ---------------------------------------------------------------------------
    // 3. Surprise score — no existing memories
    // ---------------------------------------------------------------------------
    #[test]
    fn test_importance_scorer_surprise_no_memories() {
        let score = ImportanceScorer::score_surprise(&unit_vec(4, 0), &[]);
        assert!((score - 1.0_f32).abs() < f32::EPSILON);
    }

    // ---------------------------------------------------------------------------
    // 4. Surprise score — identical embedding
    // ---------------------------------------------------------------------------
    #[test]
    fn test_importance_scorer_surprise_identical() {
        let emb = unit_vec(4, 0);
        let existing = vec![MemoryRecord {
            id: 0,
            chunk: "existing".to_string(),
            embedding: emb.clone(),
            tier: MemoryTier::Working,
            importance: 0.5,
            access_count: 0,
            last_accessed: 0.0,
            created_at: 0.0,
            source: MemorySource::User,
        }];
        let score = ImportanceScorer::score_surprise(&emb, &existing);
        assert!(score < 0.01, "expected ~0.0, got {score}");
    }

    // ---------------------------------------------------------------------------
    // 5. Correction score
    // ---------------------------------------------------------------------------
    #[test]
    fn test_importance_scorer_correction() {
        assert!(
            (ImportanceScorer::score_correction(&MemorySource::Correction) - 1.0_f32).abs()
                < f32::EPSILON
        );
        assert!((ImportanceScorer::score_correction(&MemorySource::User)).abs() < f32::EPSILON);
        assert!((ImportanceScorer::score_correction(&MemorySource::System)).abs() < f32::EPSILON);
        assert!((ImportanceScorer::score_correction(&MemorySource::Tool)).abs() < f32::EPSILON);
        assert!(
            (ImportanceScorer::score_correction(&MemorySource::Retrieval)).abs() < f32::EPSILON
        );
    }

    // ---------------------------------------------------------------------------
    // 6. Length score
    // ---------------------------------------------------------------------------
    #[test]
    fn test_importance_scorer_length() {
        assert!((ImportanceScorer::score_length("")).abs() < f32::EPSILON);
        // 50 words → 0.5
        let fifty_words = std::iter::repeat("word")
            .take(50)
            .collect::<Vec<_>>()
            .join(" ");
        let s50 = ImportanceScorer::score_length(&fifty_words);
        assert!((s50 - 0.5).abs() < 1e-5, "expected 0.5, got {s50}");

        // 100 words → 1.0
        let hundred_words = std::iter::repeat("word")
            .take(100)
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(ImportanceScorer::score_length(&hundred_words), 1.0);

        // 200 words → still 1.0 (clamped)
        let two_hundred = std::iter::repeat("word")
            .take(200)
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(ImportanceScorer::score_length(&two_hundred), 1.0);
    }

    // ---------------------------------------------------------------------------
    // 7. Combined scorer
    // ---------------------------------------------------------------------------
    #[test]
    fn test_importance_scorer_combined() {
        let weights = ImportanceWeights {
            surprise: 0.5,
            correction: 0.3,
            length: 0.2,
        };
        // All 1.0 → should return 1.0
        assert!(
            (ImportanceScorer::score_combined(1.0, 1.0, 1.0, &weights) - 1.0_f32).abs()
                < f32::EPSILON
        );
        assert!((ImportanceScorer::score_combined(0.0, 0.0, 0.0, &weights)).abs() < f32::EPSILON);

        // Weighted: 0.5*0.5 + 0.0*0.3 + 1.0*0.2 = 0.25 + 0.0 + 0.20 = 0.45, total=1.0 → 0.45
        let v = ImportanceScorer::score_combined(0.5, 0.0, 1.0, &weights);
        assert!((v - 0.45).abs() < 1e-5, "expected 0.45, got {v}");
    }

    // ---------------------------------------------------------------------------
    // 8. Decay calculator
    // ---------------------------------------------------------------------------
    #[test]
    fn test_decay_calculator() {
        let rec = MemoryRecord {
            id: 0,
            chunk: "test".to_string(),
            embedding: vec![1.0],
            tier: MemoryTier::Episodic,
            importance: 1.0,
            access_count: 0,
            last_accessed: 0.0,
            created_at: 0.0,
            source: MemorySource::User,
        };
        // At t=0 → decay = 1.0 * 1 * exp(0) = 1.0
        let lambda = 0.001_f64;
        let d0 = DecayCalculator::compute_decay(&rec, 0.0, lambda);
        assert!((d0 - 1.0).abs() < 1e-5, "expected 1.0 at t=0, got {d0}");

        // At t=1000 → decay = 1.0 * 1 * exp(-1.0) ≈ 0.3679
        let d1 = DecayCalculator::compute_decay(&rec, 1000.0, lambda);
        let expected = f64::exp(-1.0) as f32;
        assert!(
            (d1 - expected).abs() < 1e-4,
            "expected {expected}, got {d1}"
        );

        // Higher access_count boosts the score
        let rec2 = MemoryRecord {
            access_count: 9,
            ..rec.clone()
        };
        let d2 = DecayCalculator::compute_decay(&rec2, 0.0, lambda);
        assert!(
            (d2 - 10.0).abs() < 1e-4,
            "expected 10.0 with access_count=9, got {d2}"
        );
    }

    // ---------------------------------------------------------------------------
    // 9. Consolidate — eviction from Working
    // ---------------------------------------------------------------------------
    #[test]
    fn test_consolidate_eviction_working() {
        let mut cfg = ConsolidationConfig::default();
        cfg.working_capacity = 3;
        cfg.working_to_episodic_threshold = 2.0; // never promote in this test
        let mut engine = ConsolidationEngine::new(cfg);

        // Add 5 records; all have very low importance so none get promoted.
        for i in 0..5_u64 {
            let id = engine.add_memory(
                "x".to_string(),
                unit_vec(4, i as usize),
                MemorySource::User,
                i as f64,
            );
            // Force low importance so promotion threshold is not crossed.
            engine
                .records
                .iter_mut()
                .find(|r| r.id == id)
                .unwrap()
                .importance = 0.1;
        }
        assert_eq!(engine.records().len(), 5);

        engine.consolidate(100.0);

        // After eviction, working count should be <= 3.
        let working = engine
            .records()
            .iter()
            .filter(|r| r.tier == MemoryTier::Working)
            .count();
        assert!(working <= 3, "working count should be ≤ 3, got {working}");
        assert!(engine.stats.total_evictions >= 2, "expected ≥ 2 evictions");
    }

    // ---------------------------------------------------------------------------
    // 10. Consolidate — promotion to Episodic
    // ---------------------------------------------------------------------------
    #[test]
    fn test_consolidate_promotion_to_episodic() {
        let cfg = ConsolidationConfig::default();
        let mut engine = ConsolidationEngine::new(cfg);

        let id = engine.add_memory(
            "important memory".to_string(),
            unit_vec(4, 0),
            MemorySource::Correction,
            0.0,
        );
        // Force importance above threshold.
        engine
            .records
            .iter_mut()
            .find(|r| r.id == id)
            .unwrap()
            .importance = 0.9;

        engine.consolidate(0.0);

        let rec = engine.get_by_id(id).unwrap();
        assert_eq!(
            rec.tier,
            MemoryTier::Episodic,
            "record should have been promoted to Episodic"
        );
        assert!(engine.stats.total_promotions >= 1);
    }

    // ---------------------------------------------------------------------------
    // 11. Consolidate — promotion to Semantic
    // ---------------------------------------------------------------------------
    #[test]
    fn test_consolidate_promotion_to_semantic() {
        let cfg = ConsolidationConfig::default(); // threshold = 10
        let mut engine = ConsolidationEngine::new(cfg);

        let id = engine.add_memory(
            "frequently accessed".to_string(),
            unit_vec(4, 0),
            MemorySource::User,
            0.0,
        );

        // Place record directly in Episodic tier with high access count.
        {
            let rec = engine.records.iter_mut().find(|r| r.id == id).unwrap();
            rec.tier = MemoryTier::Episodic;
            rec.access_count = 11; // > threshold of 10
        }

        engine.consolidate(0.0);

        let rec = engine.get_by_id(id).unwrap();
        assert_eq!(
            rec.tier,
            MemoryTier::Semantic,
            "record should have been promoted to Semantic"
        );
        assert!(engine.stats.total_promotions >= 1);
    }

    // ---------------------------------------------------------------------------
    // 12. access_memory — increments count and timestamp
    // ---------------------------------------------------------------------------
    #[test]
    fn test_access_memory_reactivation() {
        let mut engine = ConsolidationEngine::new(ConsolidationConfig::default());
        let id = engine.add_memory("chunk".to_string(), unit_vec(4, 0), MemorySource::User, 0.0);

        engine.access_memory(id, 5000.0);
        let rec = engine.get_by_id(id).unwrap();
        assert_eq!(rec.access_count, 1);
        assert!((rec.last_accessed - 5000.0_f64).abs() < f64::EPSILON);

        engine.access_memory(id, 9999.0);
        let rec = engine.get_by_id(id).unwrap();
        assert_eq!(rec.access_count, 2);
        assert!((rec.last_accessed - 9999.0_f64).abs() < f64::EPSILON);
    }

    // ---------------------------------------------------------------------------
    // 13. get_stats — counts match record tiers
    // ---------------------------------------------------------------------------
    #[test]
    fn test_get_stats() {
        let mut engine = ConsolidationEngine::new(ConsolidationConfig::default());

        // 2 Working
        engine.add_memory("w1".to_string(), unit_vec(4, 0), MemorySource::User, 0.0);
        engine.add_memory("w2".to_string(), unit_vec(4, 1), MemorySource::User, 0.0);

        // 1 Episodic (manually set)
        let id_e = engine.add_memory("e1".to_string(), unit_vec(4, 2), MemorySource::User, 0.0);
        engine
            .records
            .iter_mut()
            .find(|r| r.id == id_e)
            .unwrap()
            .tier = MemoryTier::Episodic;

        // 1 Semantic (manually set)
        let id_s = engine.add_memory("s1".to_string(), unit_vec(4, 3), MemorySource::User, 0.0);
        engine
            .records
            .iter_mut()
            .find(|r| r.id == id_s)
            .unwrap()
            .tier = MemoryTier::Semantic;

        let stats = engine.get_stats();
        assert_eq!(stats.working_count, 2);
        assert_eq!(stats.episodic_count, 1);
        assert_eq!(stats.semantic_count, 1);
    }

    // ---------------------------------------------------------------------------
    // 14. Consolidate — Episodic eviction over capacity
    // ---------------------------------------------------------------------------
    #[test]
    fn test_consolidate_episodic_eviction() {
        let mut cfg = ConsolidationConfig::default();
        cfg.episodic_capacity = 3;
        cfg.working_to_episodic_threshold = 2.0; // never auto-promote from Working
        let mut engine = ConsolidationEngine::new(cfg);

        // Seed 5 records directly in Episodic.
        for i in 0..5_u64 {
            let id = engine.add_memory(
                "episodic chunk".to_string(),
                unit_vec(4, i as usize),
                MemorySource::User,
                i as f64,
            );
            let rec = engine.records.iter_mut().find(|r| r.id == id).unwrap();
            rec.tier = MemoryTier::Episodic;
            rec.importance = 0.5;
            rec.access_count = 1;
        }

        assert_eq!(engine.records().len(), 5);
        engine.consolidate(100_000.0);

        let episodic = engine
            .records()
            .iter()
            .filter(|r| r.tier == MemoryTier::Episodic)
            .count();
        assert!(
            episodic <= 3,
            "episodic count should be ≤ 3, got {episodic}"
        );
        assert!(
            engine.stats.total_evictions >= 2,
            "expected ≥ 2 episodic evictions"
        );
    }
}
