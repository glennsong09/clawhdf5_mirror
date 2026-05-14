//! Query expansion for broader retrieval recall.
//!
//! Generates related queries from the original query using:
//! - Synonym expansion (built-in word lists)
//! - Acronym expansion/contraction
//! - Temporal expansion (time-related rewrites)
//! - Morphological variants (stemming-like transforms)
//! - Knowledge graph expansion (entity aliases and neighbors)

use crate::knowledge::KnowledgeCache;

/// Configuration for query expansion.
#[derive(Debug, Clone)]
pub struct QueryExpansionConfig {
    /// Maximum number of expanded queries to generate.
    pub max_expansions: usize,
    /// Whether to include synonym expansions.
    pub synonyms: bool,
    /// Whether to expand/contract acronyms.
    pub acronyms: bool,
    /// Whether to generate temporal variants.
    pub temporal: bool,
    /// Whether to apply morphological variants.
    pub morphological: bool,
}

impl Default for QueryExpansionConfig {
    fn default() -> Self {
        Self {
            max_expansions: 5,
            synonyms: true,
            acronyms: true,
            temporal: true,
            morphological: true,
        }
    }
}

/// A generated query variant.
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    /// The rewritten query text.
    pub text: String,
    /// Why this expansion was generated.
    pub expansion_type: String,
    /// How confident we are this is a useful expansion (0-1).
    pub weight: f32,
}

/// Query expander supporting multiple expansion strategies.
pub struct QueryExpander {
    config: QueryExpansionConfig,
}

// ---------------------------------------------------------------------------
// Synonym groups
// ---------------------------------------------------------------------------

/// Groups of interchangeable terms. The first element is treated as the
/// canonical form; all members are substituted for each other during expansion.
const SYNONYM_GROUPS: &[&[&str]] = &[
    &["search", "find", "look for", "query", "retrieve"],
    &["create", "build", "make", "construct", "implement"],
    &["delete", "remove", "drop", "erase", "destroy"],
    &["update", "modify", "change", "edit", "alter"],
    &["fast", "quick", "rapid", "speedy", "performant"],
    &["error", "bug", "issue", "problem", "defect"],
    &["memory", "recall", "remember", "recollection"],
];

// ---------------------------------------------------------------------------
// Acronyms: (acronym, expanded form)
// ---------------------------------------------------------------------------

const ACRONYMS: &[(&str, &str)] = &[
    ("API", "Application Programming Interface"),
    ("DB", "database"),
    ("database", "DB"),
    ("ML", "Machine Learning"),
    ("AI", "Artificial Intelligence"),
    ("LLM", "Large Language Model"),
    ("PR", "Pull Request"),
    ("CI", "Continuous Integration"),
    ("CD", "Continuous Deployment"),
    ("CI/CD", "Continuous Integration / Continuous Deployment"),
    ("ORM", "Object Relational Mapper"),
    ("SDK", "Software Development Kit"),
    ("CLI", "Command Line Interface"),
    ("GUI", "Graphical User Interface"),
    ("HTTP", "Hypertext Transfer Protocol"),
    ("SQL", "Structured Query Language"),
    ("NoSQL", "Not only SQL"),
    ("OS", "Operating System"),
    ("CPU", "Central Processing Unit"),
    ("GPU", "Graphics Processing Unit"),
];

// ---------------------------------------------------------------------------
// Temporal trigger words and their expansions
// ---------------------------------------------------------------------------

const TEMPORAL_EXPANSIONS: &[(&str, &[&str])] = &[
    ("yesterday", &["today", "recent", "last few days"]),
    ("today", &["now", "current", "this moment"]),
    ("last week", &["this week", "recent", "recently"]),
    ("this week", &["recent", "last few days", "lately"]),
    ("recently", &["last few days", "this week", "today"]),
    ("next month", &["upcoming", "soon", "in the future"]),
    ("last month", &["recent", "last few weeks", "previously"]),
];

// ---------------------------------------------------------------------------
// Morphological suffix rules: (suffix_to_strip, replacements)
// The suffix is stripped from the end of a word and each replacement appended.
// ---------------------------------------------------------------------------

const MORPH_RULES: &[(&str, &[&str])] = &[
    ("ing", &["ed", "s", "ion"]),
    ("ed", &["ing", "s"]),
    ("ly", &[""]),        // quickly -> quick
    ("tion", &["te"]),    // creation -> create
    ("ations", &["ate"]), // operations -> operate
    ("ies", &["y", "ied"]),
    ("s", &[""]), // runs -> run (applied last, short suffix)
];

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl QueryExpander {
    /// Create a new `QueryExpander` with the given configuration.
    pub fn new(config: QueryExpansionConfig) -> Self {
        Self { config }
    }

    /// Generate expanded queries from the original.
    ///
    /// Returns up to `config.max_expansions` variants sorted by weight descending.
    pub fn expand(&self, query: &str) -> Vec<ExpandedQuery> {
        let mut results: Vec<ExpandedQuery> = Vec::new();

        if self.config.synonyms {
            results.extend(expand_synonyms(query));
        }
        if self.config.acronyms {
            results.extend(expand_acronyms(query));
        }
        if self.config.temporal {
            results.extend(expand_temporal(query));
        }
        if self.config.morphological {
            results.extend(expand_morphological(query));
        }

        // Remove expansions identical to the original query.
        results.retain(|e| !e.text.eq_ignore_ascii_case(query));

        // Dedup by text.
        dedup_by_text(&mut results);

        // Sort by weight descending.
        results.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.config.max_expansions);
        results
    }

    /// Expand using knowledge graph aliases and connected entity names.
    pub fn expand_with_knowledge(
        &self,
        query: &str,
        knowledge: &KnowledgeCache,
    ) -> Vec<ExpandedQuery> {
        let mut results = self.expand(query);

        // 1. Replace known aliases with canonical entity names.
        let resolved = knowledge.resolve_aliases(query);
        if !resolved.eq_ignore_ascii_case(query) {
            results.push(ExpandedQuery {
                text: resolved,
                expansion_type: "knowledge_alias".to_owned(),
                weight: 0.85,
            });
        }

        // 2. Find entities mentioned in the query (by name match) and include
        //    names of their immediate graph neighbors.
        let query_lower = query.to_lowercase();
        let mut neighbor_expansions: Vec<ExpandedQuery> = Vec::new();
        for entity in &knowledge.entities {
            if query_lower.contains(&entity.name.to_lowercase()) {
                for (neighbor, _depth) in knowledge.bfs_neighbors(entity.id, 1) {
                    let expanded = query.replace(&entity.name, &neighbor.name);
                    if !expanded.eq_ignore_ascii_case(query) {
                        neighbor_expansions.push(ExpandedQuery {
                            text: expanded,
                            expansion_type: "knowledge_graph".to_owned(),
                            weight: 0.85,
                        });
                    }
                }
            }
        }
        results.extend(neighbor_expansions);

        // Filter identical to original and dedup.
        results.retain(|e| !e.text.eq_ignore_ascii_case(query));
        dedup_by_text(&mut results);
        results.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.config.max_expansions);
        results
    }
}

// ---------------------------------------------------------------------------
// Strategy implementations
// ---------------------------------------------------------------------------

fn expand_synonyms(query: &str) -> Vec<ExpandedQuery> {
    let lower = query.to_lowercase();
    let mut results = Vec::new();

    for group in SYNONYM_GROUPS {
        for &term in group.iter() {
            if contains_word(&lower, term) {
                // For each other member of the group, produce a substitution.
                for &other in group.iter() {
                    if other == term {
                        continue;
                    }
                    let new_query = replace_word_case_insensitive(query, term, other);
                    if new_query != query {
                        results.push(ExpandedQuery {
                            text: new_query,
                            expansion_type: "synonym".to_string(),
                            weight: 0.7,
                        });
                    }
                }
            }
        }
    }
    results
}

fn expand_acronyms(query: &str) -> Vec<ExpandedQuery> {
    let mut results = Vec::new();
    for (acronym, expanded) in ACRONYMS {
        // Try expanding the acronym.
        let new_query = replace_word_case_insensitive(query, acronym, expanded);
        if new_query != query {
            results.push(ExpandedQuery {
                text: new_query,
                expansion_type: "acronym".to_string(),
                weight: 0.8,
            });
        }
    }
    results
}

fn expand_temporal(query: &str) -> Vec<ExpandedQuery> {
    let lower = query.to_lowercase();
    let mut results = Vec::new();
    for (trigger, variants) in TEMPORAL_EXPANSIONS {
        if contains_phrase(&lower, trigger) {
            for &variant in *variants {
                let new_query = case_insensitive_replace(query, trigger, variant);
                if new_query != query {
                    results.push(ExpandedQuery {
                        text: new_query,
                        expansion_type: "temporal".to_string(),
                        weight: 0.6,
                    });
                }
            }
        }
    }
    results
}

fn expand_morphological(query: &str) -> Vec<ExpandedQuery> {
    let mut results = Vec::new();
    for word in tokenize(query) {
        let lower_word = word.to_lowercase();
        'rule_loop: for (suffix, replacements) in MORPH_RULES {
            if lower_word.len() > suffix.len() + 2 && lower_word.ends_with(suffix) {
                let stem = &lower_word[..lower_word.len() - suffix.len()];
                for &rep in *replacements {
                    let new_word = format!("{}{}", stem, rep);
                    // Sanity: must be at least 2 chars.
                    if new_word.len() < 2 {
                        continue;
                    }
                    let new_query = replace_word_case_insensitive(query, &word, &new_word);
                    if new_query != query {
                        results.push(ExpandedQuery {
                            text: new_query,
                            expansion_type: "morphological".to_string(),
                            weight: 0.5,
                        });
                        break 'rule_loop;
                    }
                }
            }
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if `text` contains `word` as a complete word (whitespace/punct boundaries).
fn contains_word(text: &str, word: &str) -> bool {
    contains_phrase(text, word)
}

/// Check if `text` contains `phrase` (case-insensitive substring with word boundaries).
fn contains_phrase(text: &str, phrase: &str) -> bool {
    let lower = text.to_lowercase();
    if let Some(pos) = lower.find(phrase) {
        let end = pos + phrase.len();
        let before_ok = pos == 0 || !lower.as_bytes()[pos - 1].is_ascii_alphanumeric();
        let after_ok = end >= lower.len() || !lower.as_bytes()[end].is_ascii_alphanumeric();
        before_ok && after_ok
    } else {
        false
    }
}

/// Replace a phrase in `text` case-insensitively, preserving surrounding case.
fn replace_word_case_insensitive(text: &str, from: &str, to: &str) -> String {
    case_insensitive_replace(text, from, to)
}

fn case_insensitive_replace(text: &str, from: &str, to: &str) -> String {
    let lower = text.to_lowercase();
    let lower_from = from.to_lowercase();
    if let Some(pos) = lower.find(&lower_from) {
        let end = pos + from.len();
        format!("{}{}{}", &text[..pos], to, &text[end..])
    } else {
        text.to_string()
    }
}

/// Simple whitespace/punctuation tokenizer.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Dedup by lowercased text, keeping the first (highest-weight) occurrence.
fn dedup_by_text(items: &mut Vec<ExpandedQuery>) {
    let mut seen = std::collections::HashSet::new();
    items.retain(|e| seen.insert(e.text.to_lowercase()));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::KnowledgeCache;

    fn default_expander() -> QueryExpander {
        QueryExpander::new(QueryExpansionConfig::default())
    }

    // -----------------------------------------------------------------------
    // Synonym expansions
    // -----------------------------------------------------------------------

    #[test]
    fn test_synonym_search() {
        let e = default_expander();
        let expanded = e.expand("how to search for documents");
        let texts: Vec<&str> = expanded.iter().map(|x| x.text.as_str()).collect();
        // Should produce at least one synonym variant.
        let has_variant = texts
            .iter()
            .any(|t| t.contains("find") || t.contains("retrieve") || t.contains("query"));
        assert!(has_variant, "expected synonym variants, got: {:?}", texts);
    }

    #[test]
    fn test_synonym_create() {
        let e = default_expander();
        let expanded = e.expand("create a new entity");
        assert!(
            expanded
                .iter()
                .any(|x| x.text.contains("build") || x.text.contains("make"))
        );
    }

    #[test]
    fn test_synonym_weight() {
        let e = default_expander();
        let expanded = e.expand("find the document");
        let syn = expanded
            .iter()
            .find(|x| x.expansion_type == "synonym")
            .expect("no synonym");
        assert!((syn.weight - 0.7).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Acronym expansions
    // -----------------------------------------------------------------------

    #[test]
    fn test_acronym_expand_api() {
        let e = default_expander();
        let expanded = e.expand("call the API endpoint");
        assert!(
            expanded
                .iter()
                .any(|x| x.text.contains("Application Programming Interface"))
        );
    }

    #[test]
    fn test_acronym_expand_ml() {
        let e = default_expander();
        let expanded = e.expand("ML model training");
        assert!(expanded.iter().any(|x| x.text.contains("Machine Learning")));
    }

    #[test]
    fn test_acronym_expand_llm() {
        let e = default_expander();
        let expanded = e.expand("LLM inference speed");
        assert!(
            expanded
                .iter()
                .any(|x| x.text.contains("Large Language Model"))
        );
    }

    #[test]
    fn test_acronym_weight() {
        let e = default_expander();
        let expanded = e.expand("open a PR for review");
        let acr = expanded.iter().find(|x| x.expansion_type == "acronym");
        if let Some(a) = acr {
            assert!((a.weight - 0.8).abs() < 0.01);
        }
    }

    // -----------------------------------------------------------------------
    // Temporal expansions
    // -----------------------------------------------------------------------

    #[test]
    fn test_temporal_yesterday() {
        let e = default_expander();
        let expanded = e.expand("what happened yesterday");
        assert!(expanded.iter().any(|x| x.expansion_type == "temporal"));
    }

    #[test]
    fn test_temporal_recently() {
        let e = default_expander();
        let expanded = e.expand("recently added features");
        let temporal: Vec<&str> = expanded
            .iter()
            .filter(|x| x.expansion_type == "temporal")
            .map(|x| x.text.as_str())
            .collect();
        assert!(!temporal.is_empty(), "expected temporal variants");
    }

    #[test]
    fn test_temporal_weight() {
        let e = default_expander();
        let expanded = e.expand("what happened last week");
        let t = expanded.iter().find(|x| x.expansion_type == "temporal");
        if let Some(t) = t {
            assert!((t.weight - 0.6).abs() < 0.01);
        }
    }

    // -----------------------------------------------------------------------
    // Morphological expansions
    // -----------------------------------------------------------------------

    #[test]
    fn test_morph_ing_to_ed() {
        let e = default_expander();
        let expanded = e.expand("running the tests");
        // Should produce "runn" + "ed" -> "runned" or similar morph variant.
        let morph: Vec<&ExpandedQuery> = expanded
            .iter()
            .filter(|x| x.expansion_type == "morphological")
            .collect();
        assert!(!morph.is_empty(), "expected morphological variants");
    }

    #[test]
    fn test_morph_weight() {
        let e = default_expander();
        let expanded = e.expand("searching documents");
        let m = expanded
            .iter()
            .find(|x| x.expansion_type == "morphological");
        if let Some(m) = m {
            assert!((m.weight - 0.5).abs() < 0.01);
        }
    }

    // -----------------------------------------------------------------------
    // Config: disabled strategies
    // -----------------------------------------------------------------------

    #[test]
    fn test_synonyms_disabled() {
        let config = QueryExpansionConfig {
            synonyms: false,
            ..Default::default()
        };
        let e = QueryExpander::new(config);
        let expanded = e.expand("search for documents");
        assert!(!expanded.iter().any(|x| x.expansion_type == "synonym"));
    }

    #[test]
    fn test_acronyms_disabled() {
        let config = QueryExpansionConfig {
            acronyms: false,
            ..Default::default()
        };
        let e = QueryExpander::new(config);
        let expanded = e.expand("call the API endpoint");
        assert!(!expanded.iter().any(|x| x.expansion_type == "acronym"));
    }

    #[test]
    fn test_temporal_disabled() {
        let config = QueryExpansionConfig {
            temporal: false,
            ..Default::default()
        };
        let e = QueryExpander::new(config);
        let expanded = e.expand("what happened yesterday");
        assert!(!expanded.iter().any(|x| x.expansion_type == "temporal"));
    }

    // -----------------------------------------------------------------------
    // Empty query
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_query() {
        let e = default_expander();
        let expanded = e.expand("");
        // Should not panic; may return empty or minimal results.
        let _ = expanded;
    }

    // -----------------------------------------------------------------------
    // max_expansions limit
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_expansions_limit() {
        let config = QueryExpansionConfig {
            max_expansions: 2,
            ..Default::default()
        };
        let e = QueryExpander::new(config);
        let expanded = e.expand("search for a bug in the API");
        assert!(expanded.len() <= 2);
    }

    // -----------------------------------------------------------------------
    // Weight ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_ordering() {
        let e = default_expander();
        let expanded = e.expand("call the API to search for documents");
        // Results should be sorted by weight descending.
        for window in expanded.windows(2) {
            assert!(window[0].weight >= window[1].weight);
        }
    }

    // -----------------------------------------------------------------------
    // Knowledge graph expansion
    // -----------------------------------------------------------------------

    #[test]
    fn test_knowledge_alias_expansion() {
        let mut knowledge = KnowledgeCache::new();
        let id = knowledge.add_entity("PostgreSQL", "Technology", -1);
        knowledge.add_alias("pg", id as i64);
        let e = default_expander();
        let expanded = e.expand_with_knowledge("query the pg database", &knowledge);
        // Should contain a variant with "postgresql".
        assert!(
            expanded
                .iter()
                .any(|x| x.text.to_lowercase().contains("postgresql")),
            "expected alias expansion, got: {:?}",
            expanded.iter().map(|x| &x.text).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_knowledge_graph_neighbor_expansion() {
        let mut knowledge = KnowledgeCache::new();
        let pg_id = knowledge.add_entity("PostgreSQL", "Technology", -1);
        let redis_id = knowledge.add_entity("Redis", "Technology", -1);
        knowledge.add_relation(pg_id, redis_id, "related", 1.0);
        let e = default_expander();
        let expanded = e.expand_with_knowledge("connect to PostgreSQL", &knowledge);
        // Should produce a variant mentioning Redis (the neighbor).
        assert!(
            expanded.iter().any(|x| x.text.contains("Redis")),
            "expected neighbor expansion, got: {:?}",
            expanded.iter().map(|x| &x.text).collect::<Vec<_>>()
        );
    }
}
