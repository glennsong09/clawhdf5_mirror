//! Low-confidence rejection filter for retrieval results.
//!
//! Filters out results whose scores are below configurable thresholds,
//! preventing noisy or irrelevant entries from reaching the caller.

/// Configuration for the confidence rejection filter.
#[derive(Debug, Clone)]
pub struct ConfidenceConfig {
    /// Absolute minimum score a result must have to be returned.
    /// Results below this threshold are always dropped.
    pub min_score: f32,
    /// Maximum allowed score gap between the top-1 result and any
    /// subsequent result.  A result is dropped if:
    ///   `top_score - result_score > min_gap`
    /// Set to `f32::INFINITY` (or a very large value) to disable gap filtering.
    pub min_gap: f32,
    /// Maximum number of results to return after filtering.
    pub max_results: usize,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            min_score: 0.1,
            min_gap: 0.5,
            max_results: 10,
        }
    }
}

/// A scored result that can be passed through the confidence filter.
///
/// The type is intentionally generic: callers may wrap `SearchResult`,
/// `ReRankResult`, or any `(index, score)` pair using this struct.
#[derive(Debug, Clone)]
pub struct ScoredResult {
    /// Document index in the corpus.
    pub index: usize,
    /// Score used for confidence filtering.
    pub score: f32,
}

/// Filter out low-confidence results using absolute threshold and gap checks.
///
/// # Filtering rules (applied in order)
///
/// 1. If `results` is empty, return empty.
/// 2. If the top-1 score is below `config.min_score`, return empty
///    (nothing is good enough).
/// 3. Drop any result whose score is below `config.min_score`.
/// 4. Drop any result where `top_score - result_score > config.min_gap`.
/// 5. Truncate to `config.max_results`.
///
/// The input `results` must be sorted in **descending** order by score;
/// the output preserves that order.
///
/// # Arguments
///
/// * `results` - Slice of scored results sorted descending by score.
/// * `config` - Confidence filter configuration.
pub fn reject_low_confidence(
    results: &[ScoredResult],
    config: &ConfidenceConfig,
) -> Vec<ScoredResult> {
    if results.is_empty() {
        return Vec::new();
    }

    let top_score = results[0].score;

    // Rule 2: nothing is good enough if the very best result is below threshold.
    if top_score < config.min_score {
        return Vec::new();
    }

    // Rules 3 + 4 combined.
    let filtered: Vec<ScoredResult> = results
        .iter()
        .filter(|r| r.score >= config.min_score && (top_score - r.score) <= config.min_gap)
        .cloned()
        .collect();

    // Rule 5.
    filtered.into_iter().take(config.max_results).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(idx: usize, score: f32) -> ScoredResult {
        ScoredResult { index: idx, score }
    }

    // --- empty input ---

    #[test]
    fn empty_input_returns_empty() {
        let result = reject_low_confidence(&[], &ConfidenceConfig::default());
        assert!(result.is_empty());
    }

    // --- rule 2: top below threshold ---

    #[test]
    fn top_below_threshold_returns_empty() {
        let config = ConfidenceConfig {
            min_score: 0.5,
            min_gap: 1.0,
            max_results: 10,
        };
        let results = vec![scored(0, 0.3), scored(1, 0.2)];
        let out = reject_low_confidence(&results, &config);
        assert!(
            out.is_empty(),
            "nothing returned when top score < min_score"
        );
    }

    // --- rule 3: absolute threshold ---

    #[test]
    fn drops_results_below_min_score() {
        let config = ConfidenceConfig {
            min_score: 0.4,
            min_gap: f32::INFINITY,
            max_results: 10,
        };
        let results = vec![
            scored(0, 0.9),
            scored(1, 0.5),
            scored(2, 0.3),
            scored(3, 0.1),
        ];
        let out = reject_low_confidence(&results, &config);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].index, 0);
        assert_eq!(out[1].index, 1);
    }

    // --- rule 4: gap filter ---

    #[test]
    fn drops_results_outside_gap() {
        let config = ConfidenceConfig {
            min_score: 0.0,
            min_gap: 0.2,
            max_results: 10,
        };
        let results = vec![
            scored(0, 1.0),
            scored(1, 0.9),  // gap 0.1 – kept
            scored(2, 0.75), // gap 0.25 – dropped
            scored(3, 0.5),  // gap 0.5 – dropped
        ];
        let out = reject_low_confidence(&results, &config);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].index, 0);
        assert_eq!(out[1].index, 1);
    }

    // --- rule 5: max_results ---

    #[test]
    fn truncates_to_max_results() {
        let config = ConfidenceConfig {
            min_score: 0.0,
            min_gap: f32::INFINITY,
            max_results: 2,
        };
        let results = vec![
            scored(0, 0.9),
            scored(1, 0.8),
            scored(2, 0.7),
            scored(3, 0.6),
        ];
        let out = reject_low_confidence(&results, &config);
        assert_eq!(out.len(), 2);
    }

    // --- combined rules ---

    #[test]
    fn combined_threshold_and_gap() {
        let config = ConfidenceConfig {
            min_score: 0.3,
            min_gap: 0.4,
            max_results: 10,
        };
        let results = vec![
            scored(0, 0.9),
            scored(1, 0.6), // gap 0.3 ≤ 0.4 and ≥ min_score – kept
            scored(2, 0.4), // gap 0.5 > 0.4 – dropped by gap
            scored(3, 0.2), // below min_score – dropped by threshold
        ];
        let out = reject_low_confidence(&results, &config);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].index, 0);
        assert_eq!(out[1].index, 1);
    }

    #[test]
    fn single_result_above_threshold_kept() {
        let config = ConfidenceConfig {
            min_score: 0.5,
            min_gap: 0.3,
            max_results: 10,
        };
        let results = vec![scored(0, 0.8)];
        let out = reject_low_confidence(&results, &config);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].index, 0);
    }

    #[test]
    fn output_preserves_descending_order() {
        let config = ConfidenceConfig {
            min_score: 0.0,
            min_gap: f32::INFINITY,
            max_results: 10,
        };
        let results = vec![scored(0, 0.9), scored(1, 0.7), scored(2, 0.5)];
        let out = reject_low_confidence(&results, &config);
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score, "output must be sorted descending");
        }
    }

    #[test]
    fn all_results_equal_scores_all_kept() {
        let config = ConfidenceConfig {
            min_score: 0.5,
            min_gap: 0.0, // gap=0 means only equal scores to top are kept
            max_results: 10,
        };
        let results = vec![scored(0, 0.8), scored(1, 0.8), scored(2, 0.8)];
        let out = reject_low_confidence(&results, &config);
        assert_eq!(out.len(), 3);
    }
}
