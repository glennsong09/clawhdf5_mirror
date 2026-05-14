//! Multi-factor re-ranking of retrieval results.
//!
//! Combines temporal recency, source authority, and Hebbian activation weight
//! into a single composite score for each retrieved result.

/// Configuration for the multi-factor re-ranker.
#[derive(Debug, Clone)]
pub struct ReRankConfig {
    /// Weight applied to the temporal decay score (0.0–1.0).
    pub temporal_weight: f32,
    /// Weight applied to the source authority score (0.0–1.0).
    pub authority_weight: f32,
    /// Weight applied to the Hebbian activation score (0.0–1.0).
    pub activation_weight: f32,
    /// Half-life in seconds for temporal exponential decay.
    /// After this many seconds the temporal score is 0.5.
    pub temporal_half_life_secs: f64,
}

impl Default for ReRankConfig {
    fn default() -> Self {
        Self {
            temporal_weight: 0.3,
            authority_weight: 0.2,
            activation_weight: 0.5,
            temporal_half_life_secs: 86_400.0, // 24 hours
        }
    }
}

/// Per-result score breakdown produced by the re-ranker.
#[derive(Debug, Clone)]
pub struct ReRankResult {
    /// Original document index in the corpus.
    pub index: usize,
    /// Final composite re-rank score (weighted sum of factor scores).
    pub combined_score: f32,
    /// Temporal recency score in [0, 1] (1 = very recent, 0 = very old).
    pub temporal_score: f32,
    /// Source authority score in [0, 1].
    pub authority_score: f32,
    /// Normalised Hebbian activation score in [0, 1].
    pub activation_score: f32,
}

/// Compute an exponential decay temporal score.
///
/// Returns a value in `(0, 1]`:
/// - 1.0  when `timestamp == now_timestamp` (no decay).
/// - 0.5  when `now_timestamp - timestamp == half_life_secs`.
/// - Approaches 0 for very old entries.
///
/// # Arguments
///
/// * `timestamp` - Unix-epoch timestamp of the stored entry (seconds, f64).
/// * `now_timestamp` - Current Unix-epoch timestamp (seconds, f64).
/// * `half_life_secs` - Desired half-life in seconds.
pub fn temporal_score(timestamp: f64, now_timestamp: f64, half_life_secs: f64) -> f32 {
    let age_secs = (now_timestamp - timestamp).max(0.0);
    // score = 2^(-age / half_life)
    let exponent = -age_secs / half_life_secs.max(1.0);
    (2.0_f64.powf(exponent)) as f32
}

/// Compute source authority score based on channel type.
///
/// Hierarchy (higher = more authoritative):
/// 1. `"user_correction"` → 1.0
/// 2. `"conversation"` → 0.7
/// 3. `"system"` → 0.4
/// 4. anything else → 0.2
///
/// # Arguments
///
/// * `source_channel` - The `source_channel` field from the memory entry.
pub fn source_authority_score(source_channel: &str) -> f32 {
    match source_channel {
        "user_correction" => 1.0,
        "conversation" => 0.7,
        "system" => 0.4,
        _ => 0.2,
    }
}

/// Pass-through normalisation for Hebbian activation weights.
///
/// Clamps the raw activation weight to `[0, 1]` so downstream arithmetic
/// stays bounded.
///
/// # Arguments
///
/// * `raw_activation` - Raw activation weight (may be > 1 after boosts).
pub fn activation_score(raw_activation: f32) -> f32 {
    raw_activation.clamp(0.0, 1.0)
}

/// A minimal view of a retrieved result supplied to the re-ranker.
#[derive(Debug, Clone)]
pub struct RerankInput {
    /// Document index in the corpus.
    pub index: usize,
    /// Unix-epoch timestamp of the stored entry (seconds).
    pub timestamp: f64,
    /// Source channel string (e.g. `"user_correction"`, `"conversation"`).
    pub source_channel: String,
    /// Raw Hebbian activation weight for this entry.
    pub raw_activation: f32,
}

/// Re-rank a list of retrieval results using multi-factor scoring.
///
/// Returns a `Vec<ReRankResult>` sorted in **descending** order of
/// `combined_score`.
///
/// The combined score is a weighted sum:
///
/// ```text
/// combined = w_t * temporal + w_a * authority + w_act * activation
/// ```
///
/// The weights in `config` do not need to sum to 1.0; results are ranked
/// relative to each other.
///
/// # Arguments
///
/// * `inputs` - Slice of retrieval results to re-rank.
/// * `config` - Re-ranking weights and half-life configuration.
/// * `now_timestamp` - Current Unix-epoch time in seconds.
pub fn rerank(
    inputs: &[RerankInput],
    config: &ReRankConfig,
    now_timestamp: f64,
) -> Vec<ReRankResult> {
    let mut results: Vec<ReRankResult> = inputs
        .iter()
        .map(|inp| {
            let ts = temporal_score(inp.timestamp, now_timestamp, config.temporal_half_life_secs);
            let auth = source_authority_score(&inp.source_channel);
            let act = activation_score(inp.raw_activation);

            let combined = config.temporal_weight * ts
                + config.authority_weight * auth
                + config.activation_weight * act;

            ReRankResult {
                index: inp.index,
                combined_score: combined,
                temporal_score: ts,
                authority_score: auth,
                activation_score: act,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- temporal_score ---

    #[test]
    fn temporal_score_zero_age_is_one() {
        let score = temporal_score(1000.0, 1000.0, 86_400.0);
        assert!((score - 1.0).abs() < 1e-6, "age=0 should give score 1.0");
    }

    #[test]
    fn temporal_score_half_life_gives_half() {
        let half_life = 3600.0_f64;
        let score = temporal_score(0.0, half_life, half_life);
        assert!((score - 0.5).abs() < 1e-6, "age=half_life should give 0.5");
    }

    #[test]
    fn temporal_score_future_timestamp_clamped() {
        // timestamp in the future should not produce negative age
        let score = temporal_score(2000.0, 1000.0, 86_400.0);
        assert!(score <= 1.0 && score > 0.0);
    }

    #[test]
    fn temporal_score_old_entry_near_zero() {
        // 10 half-lives old → 2^-10 ≈ 0.001
        let half_life = 3600.0_f64;
        let score = temporal_score(0.0, 10.0 * half_life, half_life);
        assert!(score < 0.002, "very old entry should have near-zero score");
    }

    // --- source_authority_score ---

    #[test]
    fn authority_user_correction_is_highest() {
        assert_eq!(source_authority_score("user_correction"), 1.0);
    }

    #[test]
    fn authority_conversation() {
        assert!((source_authority_score("conversation") - 0.7).abs() < 1e-6);
    }

    #[test]
    fn authority_system() {
        assert!((source_authority_score("system") - 0.4).abs() < 1e-6);
    }

    #[test]
    fn authority_unknown_channel() {
        assert!((source_authority_score("whatsapp") - 0.2).abs() < 1e-6);
        assert!((source_authority_score("") - 0.2).abs() < 1e-6);
    }

    #[test]
    fn authority_ordering() {
        let uc = source_authority_score("user_correction");
        let cv = source_authority_score("conversation");
        let sy = source_authority_score("system");
        let ot = source_authority_score("other");
        assert!(uc > cv && cv > sy && sy > ot);
    }

    // --- activation_score ---

    #[test]
    fn activation_score_clamps_above_one() {
        assert_eq!(activation_score(5.0), 1.0);
    }

    #[test]
    fn activation_score_clamps_below_zero() {
        assert_eq!(activation_score(-1.0), 0.0);
    }

    #[test]
    fn activation_score_passthrough_in_range() {
        assert!((activation_score(0.6) - 0.6).abs() < 1e-6);
    }

    // --- rerank ---

    fn make_inputs() -> Vec<RerankInput> {
        vec![
            RerankInput {
                index: 0,
                timestamp: 0.0, // very old
                source_channel: "other".to_string(),
                raw_activation: 0.1,
            },
            RerankInput {
                index: 1,
                timestamp: 86_400.0, // one day ago
                source_channel: "conversation".to_string(),
                raw_activation: 0.5,
            },
            RerankInput {
                index: 2,
                timestamp: 172_800.0, // "now"
                source_channel: "user_correction".to_string(),
                raw_activation: 1.0,
            },
        ]
    }

    #[test]
    fn rerank_returns_all_entries() {
        let inputs = make_inputs();
        let config = ReRankConfig::default();
        let results = rerank(&inputs, &config, 172_800.0);
        assert_eq!(results.len(), inputs.len());
    }

    #[test]
    fn rerank_sorted_descending() {
        let inputs = make_inputs();
        let config = ReRankConfig::default();
        let results = rerank(&inputs, &config, 172_800.0);
        for w in results.windows(2) {
            assert!(
                w[0].combined_score >= w[1].combined_score,
                "results must be sorted descending"
            );
        }
    }

    #[test]
    fn rerank_best_entry_is_recent_high_authority() {
        let inputs = make_inputs();
        let config = ReRankConfig::default();
        let results = rerank(&inputs, &config, 172_800.0);
        // index 2 is most recent + user_correction + highest activation
        assert_eq!(results[0].index, 2);
    }

    #[test]
    fn rerank_score_breakdown_matches_manual_calculation() {
        let config = ReRankConfig {
            temporal_weight: 1.0,
            authority_weight: 0.0,
            activation_weight: 0.0,
            temporal_half_life_secs: 3600.0,
        };
        let inputs = vec![RerankInput {
            index: 0,
            timestamp: 0.0,
            source_channel: "other".to_string(),
            raw_activation: 0.5,
        }];
        let now = 3600.0_f64; // exactly one half-life later
        let results = rerank(&inputs, &config, now);
        assert_eq!(results.len(), 1);
        assert!((results[0].temporal_score - 0.5).abs() < 1e-5);
        assert!((results[0].combined_score - 0.5).abs() < 1e-5);
    }

    #[test]
    fn rerank_empty_input() {
        let results = rerank(&[], &ReRankConfig::default(), 0.0);
        assert!(results.is_empty());
    }
}
