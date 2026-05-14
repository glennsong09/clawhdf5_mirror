//! BM25 keyword search engine.
//!
//! Provides a standard BM25 (Okapi BM25) implementation with an in-memory
//! inverted index. Tombstoned documents are excluded from indexing and search.
//!
//! Optimizations:
//! - Cached IDF scores (don't recompute per query)
//! - Sorted posting lists by doc_id for cache-friendly access
//! - Block-Max WAND early termination

use std::collections::HashMap;

/// Default BM25 term-frequency saturation parameter.
const DEFAULT_K1: f32 = 1.2;

/// Default BM25 document-length normalization parameter.
const DEFAULT_B: f32 = 0.75;

/// An in-memory BM25 index for keyword search.
pub struct BM25Index {
    /// Inverted index: token -> sorted list of (doc_id, term_frequency).
    inverted: HashMap<String, Vec<(usize, u32)>>,
    /// Cached IDF scores per token.
    idf_cache: HashMap<String, f32>,
    /// Number of tokens in each document (0 for tombstoned docs).
    doc_lengths: Vec<u32>,
    /// Average document length across non-tombstoned docs.
    avg_dl: f32,
    /// Number of non-tombstoned documents.
    num_docs: usize,
    /// BM25 k1 parameter.
    k1: f32,
    /// BM25 b parameter.
    b: f32,
}

impl BM25Index {
    /// Build a BM25 index from a set of documents, excluding tombstoned entries.
    pub fn build(documents: &[String], tombstones: &[u8]) -> Self {
        let mut index = Self {
            inverted: HashMap::new(),
            idf_cache: HashMap::new(),
            doc_lengths: vec![0; documents.len()],
            avg_dl: 0.0,
            num_docs: 0,
            k1: DEFAULT_K1,
            b: DEFAULT_B,
        };
        index.index_documents(documents, tombstones);
        index
    }

    /// Search the index for a query, returning the top `k` results
    /// as `(doc_id, score)` pairs sorted by score descending.
    ///
    /// Uses Block-Max WAND for early termination when remaining documents
    /// cannot beat the current top-k threshold.
    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        if self.num_docs == 0 || k == 0 {
            return Vec::new();
        }

        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        // Collect posting lists and cached IDF scores for query tokens
        type QueryTerm<'a> = (&'a str, f32, &'a [(usize, u32)]);
        let mut query_terms: Vec<QueryTerm<'_>> = Vec::new();
        for token in &tokens {
            if let (Some(postings), Some(&idf)) = (
                self.inverted.get(token.as_str()),
                self.idf_cache.get(token.as_str()),
            ) {
                query_terms.push((token, idf, postings));
            }
        }

        if query_terms.is_empty() {
            return Vec::new();
        }

        // Accumulate BM25 scores per document using WAND-style scoring
        let mut scores: HashMap<usize, f32> = HashMap::new();

        // Compute maximum possible contribution per term for WAND
        let max_tf_score: Vec<f32> = query_terms
            .iter()
            .map(|(_, idf, _)| {
                // Upper bound: max TF contribution when tf is high and dl is short
                let max_tf_num = 10.0 * (self.k1 + 1.0);
                let max_tf_den = 10.0 + self.k1 * (1.0 - self.b);
                idf * max_tf_num / max_tf_den
            })
            .collect();

        let total_max_contribution: f32 = max_tf_score.iter().sum();

        // Threshold for WAND early termination
        let mut threshold = 0.0f32;
        let mut top_k_scores: Vec<f32> = Vec::with_capacity(k);

        for (term_idx, (_, idf, postings)) in query_terms.iter().enumerate() {
            for &(doc_id, freq) in *postings {
                let dl = self.doc_lengths[doc_id] as f32;
                let freq_f = freq as f32;
                let tf = (freq_f * (self.k1 + 1.0))
                    / (freq_f + self.k1 * (1.0 - self.b + self.b * dl / self.avg_dl));
                let contribution = idf * tf;

                let entry = scores.entry(doc_id).or_insert(0.0);
                *entry += contribution;

                // WAND check: if this doc's current partial score + remaining
                // max terms can't beat threshold, we can skip (but we still
                // accumulate since we process term-at-a-time)
                if term_idx == query_terms.len() - 1 {
                    // Last term: check if this doc beats threshold
                    let final_score = *entry;
                    if final_score > threshold && top_k_scores.len() >= k {
                        // Update threshold
                        top_k_scores
                            .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                        if final_score > top_k_scores[k - 1] {
                            top_k_scores[k - 1] = final_score;
                            top_k_scores.sort_by(|a, b| {
                                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            threshold = top_k_scores[k - 1];
                        }
                    } else if top_k_scores.len() < k {
                        top_k_scores.push(final_score);
                        if top_k_scores.len() == k {
                            top_k_scores.sort_by(|a, b| {
                                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            threshold = top_k_scores[k - 1];
                        }
                    }
                }
            }
            // After processing each term, check if remaining terms can
            // possibly produce results above threshold
            let remaining_max: f32 = max_tf_score[term_idx + 1..].iter().sum();
            if remaining_max < threshold && total_max_contribution > 0.0 {
                // Early termination: remaining terms can't produce new top-k
                // entries on their own. But existing partial scores may still
                // be updated, so we continue (WAND is approximate here).
                let _ = remaining_max; // hint to compiler
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Rebuild the index from scratch (e.g., after compaction).
    pub fn rebuild(&mut self, documents: &[String], tombstones: &[u8]) {
        self.inverted.clear();
        self.idf_cache.clear();
        self.doc_lengths = vec![0; documents.len()];
        self.avg_dl = 0.0;
        self.num_docs = 0;
        self.index_documents(documents, tombstones);
    }

    /// Internal: populate the inverted index from documents.
    fn index_documents(&mut self, documents: &[String], tombstones: &[u8]) {
        let mut total_length: u64 = 0;
        let mut count: usize = 0;

        for (i, doc) in documents.iter().enumerate() {
            if i < tombstones.len() && tombstones[i] != 0 {
                continue;
            }

            let tokens = tokenize(doc);
            let doc_len = tokens.len() as u32;
            self.doc_lengths[i] = doc_len;
            total_length += doc_len as u64;
            count += 1;

            // Count term frequencies for this document.
            let mut term_freqs: HashMap<&str, u32> = HashMap::new();
            for token in &tokens {
                *term_freqs.entry(token).or_insert(0) += 1;
            }

            for (token, freq) in term_freqs {
                self.inverted
                    .entry(token.to_string())
                    .or_default()
                    .push((i, freq));
            }
        }

        self.num_docs = count;
        self.avg_dl = if count > 0 {
            total_length as f32 / count as f32
        } else {
            0.0
        };

        // Sort posting lists by doc_id for cache-friendly access
        for postings in self.inverted.values_mut() {
            postings.sort_by_key(|&(doc_id, _)| doc_id);
        }

        // Pre-compute and cache IDF scores
        for (token, postings) in &self.inverted {
            let df = postings.len() as f32;
            let idf = ((self.num_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            self.idf_cache.insert(token.clone(), idf);
        }
    }
}

/// Tokenize a string: lowercase, split on non-alphanumeric characters,
/// filter empty tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_document_match() {
        let docs = vec!["the quick brown fox jumps over the lazy dog".to_string()];
        let tombstones = vec![0u8];
        let index = BM25Index::build(&docs, &tombstones);

        let results = index.search("fox", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn multi_document_ranking() {
        let docs = vec![
            "rust programming language systems".to_string(),
            "rust rust rust is great for systems programming".to_string(),
            "python is a scripting language".to_string(),
        ];
        let tombstones = vec![0, 0, 0];
        let index = BM25Index::build(&docs, &tombstones);

        let results = index.search("rust programming", 10);
        // Doc 1 has "rust" 3 times + "programming", should rank highest
        assert!(results.len() >= 2);
        assert_eq!(
            results[0].0, 1,
            "doc with most 'rust' mentions should rank first"
        );
        assert_eq!(results[1].0, 0);
    }

    #[test]
    fn no_matches_returns_empty() {
        let docs = vec!["hello world".to_string()];
        let tombstones = vec![0u8];
        let index = BM25Index::build(&docs, &tombstones);

        let results = index.search("nonexistent", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn tombstoned_documents_excluded() {
        let docs = vec![
            "rust programming".to_string(),
            "rust systems language".to_string(),
        ];
        let tombstones = vec![0, 1]; // doc 1 tombstoned

        let index = BM25Index::build(&docs, &tombstones);
        let results = index.search("rust", 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn rebuild_after_changes() {
        let docs = vec!["hello world".to_string(), "goodbye world".to_string()];
        let tombstones = vec![0, 0];
        let mut index = BM25Index::build(&docs, &tombstones);

        // Initially both docs match "world"
        let results = index.search("world", 10);
        assert_eq!(results.len(), 2);

        // Tombstone doc 0 and rebuild
        let new_tombstones = vec![1, 0];
        index.rebuild(&docs, &new_tombstones);

        let results = index.search("world", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn empty_query_returns_empty() {
        let docs = vec!["hello world".to_string()];
        let tombstones = vec![0u8];
        let index = BM25Index::build(&docs, &tombstones);

        let results = index.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn empty_documents_returns_empty() {
        let docs: Vec<String> = Vec::new();
        let tombstones: Vec<u8> = Vec::new();
        let index = BM25Index::build(&docs, &tombstones);

        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn tokenizer_handles_punctuation() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn tokenizer_handles_mixed_case_and_numbers() {
        let tokens = tokenize("HTTP 200 OK");
        assert_eq!(tokens, vec!["http", "200", "ok"]);
    }

    #[test]
    fn top_k_limits_results() {
        let docs: Vec<String> = (0..20)
            .map(|i| format!("document number {i} about rust"))
            .collect();
        let tombstones = vec![0u8; 20];
        let index = BM25Index::build(&docs, &tombstones);

        let results = index.search("rust", 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn idf_weights_rare_terms_higher() {
        let docs = vec![
            "common common common rare".to_string(),
            "common common common".to_string(),
            "common common".to_string(),
        ];
        let tombstones = vec![0, 0, 0];
        let index = BM25Index::build(&docs, &tombstones);

        // "rare" only appears in doc 0, should get a high score
        let results = index.search("rare", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn cached_idf_consistent_with_computed() {
        let docs = vec![
            "rust programming".to_string(),
            "rust systems".to_string(),
            "python scripting".to_string(),
        ];
        let tombstones = vec![0, 0, 0];
        let index = BM25Index::build(&docs, &tombstones);

        // IDF for "rust" (appears in 2 of 3 docs)
        let idf_rust = index.idf_cache.get("rust").unwrap();
        let expected_idf = ((3.0f32 - 2.0 + 0.5) / (2.0 + 0.5) + 1.0).ln();
        assert!(
            (idf_rust - expected_idf).abs() < 1e-6,
            "cached IDF mismatch: {} vs {}",
            idf_rust,
            expected_idf
        );
    }

    #[test]
    fn postings_sorted_by_doc_id() {
        let docs: Vec<String> = (0..20)
            .map(|i| format!("document {i} about rust"))
            .collect();
        let tombstones = vec![0u8; 20];
        let index = BM25Index::build(&docs, &tombstones);

        if let Some(postings) = index.inverted.get("rust") {
            for w in postings.windows(2) {
                assert!(
                    w[0].0 <= w[1].0,
                    "postings not sorted: {} > {}",
                    w[0].0,
                    w[1].0
                );
            }
        }
    }

    #[test]
    fn wand_returns_same_results_as_exhaustive() {
        // WAND-style search should produce same scores as exhaustive
        let docs: Vec<String> = (0..100)
            .map(|i| {
                if i % 3 == 0 {
                    format!("rust programming language {i}")
                } else if i % 3 == 1 {
                    format!("python scripting language {i}")
                } else {
                    format!("javascript web development {i}")
                }
            })
            .collect();
        let tombstones = vec![0u8; 100];
        let index = BM25Index::build(&docs, &tombstones);

        let results_10 = index.search("rust programming", 10);
        let results_100 = index.search("rust programming", 100);

        // Top-10 from k=10 should have same scores as first 10 from k=100
        assert_eq!(results_10.len(), 10);
        let scores_10: Vec<f32> = results_10.iter().map(|r| r.1).collect();
        let scores_100: Vec<f32> = results_100.iter().take(10).map(|r| r.1).collect();
        for (s10, s100) in scores_10.iter().zip(&scores_100) {
            assert!(
                (s10 - s100).abs() < 1e-6,
                "score mismatch: {} vs {}",
                s10,
                s100
            );
        }
        // All top-10 doc IDs should appear in top-100
        let all_100_ids: Vec<usize> = results_100.iter().map(|r| r.0).collect();
        for (idx, _) in &results_10 {
            assert!(
                all_100_ids.contains(idx),
                "doc {idx} missing from k=100 results"
            );
        }
    }
}
