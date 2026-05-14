//! Lightweight entity extraction from text.
//!
//! Rule-based extractors for common entity types. No ML dependencies.
//! Designed to run on every memory save without perceptible latency.

/// Types of entities that can be extracted.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtractedEntityType {
    Person,
    Organization,
    Location,
    Date,
    Technology,
    Project,
    Custom(String),
}

/// A single extracted entity mention.
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: ExtractedEntityType,
    pub start_offset: usize,
    pub end_offset: usize,
    pub confidence: f32,
}

/// Configuration for the entity extractor.
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Minimum confidence threshold for accepting an extraction.
    pub min_confidence: f32,
    /// Whether to extract dates/times.
    pub extract_dates: bool,
    /// Whether to extract capitalized phrases as potential entities.
    pub extract_capitalized: bool,
    /// Whether to extract technology/tool mentions.
    pub extract_technology: bool,
    /// Custom patterns: (literal string, entity type, confidence).
    pub custom_patterns: Vec<(String, ExtractedEntityType, f32)>,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            extract_dates: true,
            extract_capitalized: true,
            extract_technology: true,
            custom_patterns: Vec::new(),
        }
    }
}

/// Organization suffixes that indicate a capitalized phrase is an org name.
const ORG_SUFFIXES: &[&str] = &[
    "Inc",
    "Corp",
    "LLC",
    "Ltd",
    "Systems",
    "Technologies",
    "Labs",
    "Solutions",
    "Group",
    "Co",
    "Foundation",
    "Institute",
    "Association",
];

/// Known location suffixes.
const LOCATION_SUFFIXES: &[&str] = &[
    "City",
    "State",
    "County",
    "Province",
    "Island",
    "Mountain",
    "River",
    "Lake",
    "Bay",
    "Valley",
    "Park",
    "Street",
    "Avenue",
    "Boulevard",
];

/// Built-in technology word list.
const TECHNOLOGIES: &[&str] = &[
    // Languages
    "Rust",
    "Python",
    "Go",
    "JavaScript",
    "TypeScript",
    "Java",
    "Ruby",
    "Swift",
    "Kotlin",
    "Scala",
    "Haskell",
    "Erlang",
    "Elixir",
    "Clojure",
    "Lua",
    "Perl",
    "PHP",
    "R",
    "Julia",
    "Dart",
    "Zig",
    // C family
    "C++",
    "C#",
    "C",
    // Frameworks
    "React",
    "Vue",
    "Angular",
    "Svelte",
    "Next.js",
    "Nuxt",
    "Remix",
    "Django",
    "Flask",
    "FastAPI",
    "Express",
    "Rails",
    "Spring",
    "Laravel",
    "Actix",
    "Axum",
    "Rocket",
    "Warp",
    // Databases
    "Redis",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "SQLite",
    "DynamoDB",
    "Cassandra",
    "Elasticsearch",
    "Neo4j",
    "InfluxDB",
    "CockroachDB",
    "MariaDB",
    "Oracle",
    "MSSQL",
    // Tools / Platforms
    "Docker",
    "Kubernetes",
    "Git",
    "AWS",
    "GCP",
    "Azure",
    "Terraform",
    "Ansible",
    "Helm",
    "Grafana",
    "Prometheus",
    "Kafka",
    "RabbitMQ",
    "Nginx",
    "Apache",
    "Linux",
    "macOS",
    "Windows",
    "HDF5",
    "OpenClaw",
    "clawhdf5",
    // Other
    "GraphQL",
    "gRPC",
    "REST",
    "WebSocket",
    "OAuth",
    "JWT",
    "WASM",
    "WebAssembly",
    "CUDA",
    "OpenGL",
    "Vulkan",
];

/// Common ISO date separators.
const ISO_SEPS: &[char] = &['-', '/'];

/// Relative date keywords (lower-cased for matching).
const RELATIVE_DATES: &[&str] = &[
    "yesterday",
    "today",
    "tomorrow",
    "last week",
    "next week",
    "this week",
    "last month",
    "next month",
    "this month",
    "last year",
    "next year",
    "this year",
    "recently",
    "soon",
];

/// Month names for English date patterns.
const MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
];

/// Main entity extractor.
pub struct EntityExtractor {
    config: ExtractorConfig,
}

impl EntityExtractor {
    /// Create a new extractor with the given configuration.
    pub fn new(config: ExtractorConfig) -> Self {
        Self { config }
    }

    /// Extract all entities from a text chunk.
    pub fn extract(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities: Vec<ExtractedEntity> = Vec::new();

        if self.config.extract_dates {
            entities.extend(extract_dates(text));
        }
        if self.config.extract_technology {
            entities.extend(extract_technologies(text));
        }
        // Project names (CamelCase / kebab-case)
        entities.extend(extract_projects(text));
        if self.config.extract_capitalized {
            entities.extend(extract_capitalized(text));
        }
        // Custom patterns
        for (pattern, etype, confidence) in &self.config.custom_patterns {
            entities.extend(extract_literal(text, pattern, etype.clone(), *confidence));
        }

        // Filter by min_confidence
        entities.retain(|e| e.confidence >= self.config.min_confidence);

        // Sort by start offset, then deduplicate overlapping spans (keep higher confidence).
        entities.sort_by(|a, b| {
            a.start_offset.cmp(&b.start_offset).then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        dedup_overlapping(entities)
    }

    /// Extract and deduplicate entities from multiple text chunks.
    pub fn extract_batch(&self, texts: &[&str]) -> Vec<ExtractedEntity> {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut result: Vec<ExtractedEntity> = Vec::new();
        for text in texts {
            for entity in self.extract(text) {
                let key = format!("{}\x00{:?}", entity.text.to_lowercase(), entity.entity_type);
                if seen.insert(key) {
                    result.push(entity);
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Private extraction helpers
// ---------------------------------------------------------------------------

/// Remove entities whose spans overlap with a higher-confidence entity.
fn dedup_overlapping(mut entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
    let mut result: Vec<ExtractedEntity> = Vec::new();
    for entity in entities.drain(..) {
        let overlaps = result.iter().any(|existing| {
            existing.start_offset < entity.end_offset && entity.start_offset < existing.end_offset
        });
        if !overlaps {
            result.push(entity);
        }
    }
    result
}

/// Extract technology mentions using the built-in word list.
fn extract_technologies(text: &str) -> Vec<ExtractedEntity> {
    let mut results = Vec::new();
    for tech in TECHNOLOGIES {
        // Find all occurrences (case-sensitive for tech names).
        let mut start = 0;
        while let Some(pos) = text[start..].find(tech) {
            let abs = start + pos;
            let end = abs + tech.len();
            // Verify word boundaries.
            let before_ok = abs == 0
                || !text.as_bytes()[abs - 1].is_ascii_alphanumeric()
                    && text.as_bytes()[abs - 1] != b'_';
            let after_ok = end >= text.len()
                || !text.as_bytes()[end].is_ascii_alphanumeric() && text.as_bytes()[end] != b'_';
            if before_ok && after_ok {
                // Verify there is no hyphen immediately after the tech name
                // (which would mean it is part of a longer compound identifier
                // like "clawhdf5-agent"). The full compound will be captured by
                // the kebab-case project extractor instead.
                let not_hyphen_after = end >= text.len() || text.as_bytes()[end] != b'-';
                if not_hyphen_after {
                    results.push(ExtractedEntity {
                        text: tech.to_string(),
                        entity_type: ExtractedEntityType::Technology,
                        start_offset: abs,
                        end_offset: end,
                        confidence: 0.9,
                    });
                }
            }
            start = abs + 1;
        }
    }
    results
}

/// Extract ISO and English date patterns.
fn extract_dates(text: &str) -> Vec<ExtractedEntity> {
    let mut results = Vec::new();
    let lower = text.to_lowercase();

    // Multi-word relative dates first (longest first to avoid partial matches).
    let mut sorted_rel: Vec<&str> = RELATIVE_DATES.to_vec();
    sorted_rel.sort_by_key(|b| std::cmp::Reverse(b.len()));
    for rel in sorted_rel {
        let mut start = 0;
        while let Some(pos) = lower[start..].find(rel) {
            let abs = start + pos;
            let end = abs + rel.len();
            let before_ok = abs == 0 || !lower.as_bytes()[abs - 1].is_ascii_alphabetic();
            let after_ok = end >= lower.len() || !lower.as_bytes()[end].is_ascii_alphabetic();
            if before_ok && after_ok {
                results.push(ExtractedEntity {
                    text: text[abs..end].to_string(),
                    entity_type: ExtractedEntityType::Date,
                    start_offset: abs,
                    end_offset: end,
                    confidence: 0.95,
                });
            }
            start = abs + 1;
        }
    }

    // ISO dates: YYYY-MM-DD or YYYY/MM/DD
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 10 <= bytes.len() {
        if bytes[i..i + 4].iter().all(|b| b.is_ascii_digit())
            && ISO_SEPS.contains(&(bytes[i + 4] as char))
            && bytes[i + 5..i + 7].iter().all(|b| b.is_ascii_digit())
            && bytes[i + 7] == bytes[i + 4]
            && bytes[i + 8..i + 10].iter().all(|b| b.is_ascii_digit())
        {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let end = i + 10;
            let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
            if before_ok && after_ok {
                results.push(ExtractedEntity {
                    text: text[i..end].to_string(),
                    entity_type: ExtractedEntityType::Date,
                    start_offset: i,
                    end_offset: end,
                    confidence: 0.95,
                });
                i = end;
                continue;
            }
        }
        i += 1;
    }

    // English dates: "Month DD" or "Month DD, YYYY" or "DDth of Month"
    for month in MONTHS {
        let mut start = 0;
        while let Some(pos) = lower[start..].find(month) {
            let abs = start + pos;
            let end_month = abs + month.len();
            // Make sure it's a word boundary
            let before_ok = abs == 0 || !lower.as_bytes()[abs - 1].is_ascii_alphabetic();
            let after_ok =
                end_month >= lower.len() || !lower.as_bytes()[end_month].is_ascii_alphabetic();
            if before_ok && after_ok {
                // Try "Month DD" or "Month DD, YYYY"
                let rest = &text[end_month..];
                let rest_trim = rest.trim_start();
                let ws_len = rest.len() - rest_trim.len();
                if let Some(day_len) = leading_digits_len(rest_trim) {
                    let mut end = end_month + ws_len + day_len;
                    // Optional ordinal suffix
                    let suf = &text[end..];
                    for ord in &["st", "nd", "rd", "th"] {
                        if suf.to_lowercase().starts_with(ord) {
                            end += ord.len();
                            break;
                        }
                    }
                    // Optional ", YYYY"
                    let rest2 = text[end..].trim_start();
                    let ws2 = text[end..].len() - rest2.len();
                    if let Some(after_comma) = rest2.strip_prefix(',') {
                        let rest3 = after_comma.trim_start();
                        if let Some(yr_len) = leading_digits_len(rest3)
                            && yr_len == 4
                        {
                            end = end + ws2 + 1 + (after_comma.len() - rest3.len()) + yr_len;
                        }
                    }
                    results.push(ExtractedEntity {
                        text: text[abs..end].to_string(),
                        entity_type: ExtractedEntityType::Date,
                        start_offset: abs,
                        end_offset: end,
                        confidence: 0.95,
                    });
                }
            }
            start = abs + 1;
        }
    }

    results
}

fn leading_digits_len(s: &str) -> Option<usize> {
    let n = s.bytes().take_while(|b| b.is_ascii_digit()).count();
    if n > 0 { Some(n) } else { None }
}

/// Extract CamelCase or kebab-case project identifiers.
fn extract_projects(text: &str) -> Vec<ExtractedEntity> {
    let mut results = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    // CamelCase: starts with uppercase, has at least one more uppercase-then-lower transition.
    while i < len {
        if bytes[i].is_ascii_uppercase() {
            let start = i;
            // Collect the whole identifier (letters, digits, underscore).
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &text[start..i];
            // Must contain at least one internal uppercase letter after the first char
            // and be at least 4 chars long to avoid false positives.
            if word.len() >= 4 && has_camel_hump(word) {
                // Not just an all-caps acronym
                let lower_count = word.bytes().filter(|b| b.is_ascii_lowercase()).count();
                if lower_count >= 2 {
                    results.push(ExtractedEntity {
                        text: word.to_string(),
                        entity_type: ExtractedEntityType::Project,
                        start_offset: start,
                        end_offset: i,
                        confidence: 0.7,
                    });
                }
            }
        } else {
            i += 1;
        }
    }

    // kebab-case: lowercase-word hyphen lowercase-word (at least 2 segments).
    i = 0;
    while i < len {
        if bytes[i].is_ascii_lowercase() {
            let start = i;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'-') {
                i += 1;
            }
            let candidate = &text[start..i];
            // Must contain at least one hyphen and no spaces.
            if candidate.contains('-') && candidate.len() >= 4 {
                // Each segment must be lowercase alpha.
                let segments: Vec<&str> = candidate.split('-').collect();
                if segments.len() >= 2
                    && segments
                        .iter()
                        .all(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric()))
                {
                    results.push(ExtractedEntity {
                        text: candidate.to_string(),
                        entity_type: ExtractedEntityType::Project,
                        start_offset: start,
                        end_offset: i,
                        confidence: 0.7,
                    });
                }
            }
        } else {
            i += 1;
        }
    }

    results
}

fn has_camel_hump(s: &str) -> bool {
    let bytes = s.as_bytes();
    // After the first char, look for an uppercase letter.
    bytes[1..].iter().any(|b| b.is_ascii_uppercase())
}

/// Extract capitalized phrases (Person, Organization, Location).
fn extract_capitalized(text: &str) -> Vec<ExtractedEntity> {
    let mut results = Vec::new();
    // Split into sentences by '.', '!', '?'
    // We want to skip the first word of each sentence to avoid false positives.
    let sentence_starts = sentence_start_positions(text);

    let words: Vec<(usize, &str)> = word_positions(text);
    let n = words.len();
    let mut i = 0;

    while i < n {
        let (pos, word) = words[i];
        // Skip if at sentence start.
        if sentence_starts.contains(&pos) {
            i += 1;
            continue;
        }
        if is_capitalized(word) && !is_stop_word(word) {
            // Collect a run of up to 3 capitalized words.
            let mut run: Vec<(usize, &str)> = vec![(pos, word)];
            let mut j = i + 1;
            while j < n && run.len() < 3 {
                let (p2, w2) = words[j];
                if is_capitalized(w2) && !is_stop_word(w2) {
                    run.push((p2, w2));
                    j += 1;
                } else {
                    break;
                }
            }
            // Determine entity type from the run.
            let last_word = run.last().unwrap().1;
            let entity_type = if ORG_SUFFIXES
                .iter()
                .any(|s| last_word.eq_ignore_ascii_case(s))
            {
                ExtractedEntityType::Organization
            } else if LOCATION_SUFFIXES
                .iter()
                .any(|s| last_word.eq_ignore_ascii_case(s))
            {
                ExtractedEntityType::Location
            } else if run.len() == 2 {
                ExtractedEntityType::Person
            } else {
                ExtractedEntityType::Organization
            };

            let start = run.first().unwrap().0;
            let last_start = run.last().unwrap().0;
            let last_word_bytes = run.last().unwrap().1;
            let end = last_start + last_word_bytes.len();

            results.push(ExtractedEntity {
                text: text[start..end].to_string(),
                entity_type,
                start_offset: start,
                end_offset: end,
                confidence: 0.6,
            });
            i = j;
        } else {
            i += 1;
        }
    }

    results
}

/// Find the byte offsets of all sentence-starting word positions.
fn sentence_start_positions(text: &str) -> std::collections::HashSet<usize> {
    let mut starts = std::collections::HashSet::new();
    // The very first word is a sentence start.
    let mut after_end = true;
    for (i, ch) in text.char_indices() {
        if after_end && ch.is_alphabetic() {
            starts.insert(i);
            after_end = false;
        }
        if matches!(ch, '.' | '!' | '?' | '\n') {
            after_end = true;
        }
    }
    starts
}

/// Return (byte_offset, word_str) pairs for all word tokens.
fn word_positions(text: &str) -> Vec<(usize, &str)> {
    let mut result = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();
    while i < len {
        if bytes[i].is_ascii_alphabetic() {
            let start = i;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'\'') {
                i += 1;
            }
            result.push((start, &text[start..i]));
        } else {
            i += 1;
        }
    }
    result
}

fn is_capitalized(word: &str) -> bool {
    word.chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false)
}

fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "The"
            | "A"
            | "An"
            | "In"
            | "On"
            | "At"
            | "To"
            | "For"
            | "Of"
            | "And"
            | "Or"
            | "But"
            | "Is"
            | "Are"
            | "Was"
            | "Were"
            | "It"
            | "This"
            | "That"
            | "These"
            | "Those"
            | "He"
            | "She"
            | "They"
            | "We"
            | "I"
            | "My"
            | "His"
            | "Her"
            | "Its"
            | "Our"
    )
}

/// Extract all literal occurrences of `pattern` in `text`.
fn extract_literal(
    text: &str,
    pattern: &str,
    entity_type: ExtractedEntityType,
    confidence: f32,
) -> Vec<ExtractedEntity> {
    let mut results = Vec::new();
    let mut start = 0;
    while let Some(pos) = text[start..].find(pattern) {
        let abs = start + pos;
        let end = abs + pattern.len();
        results.push(ExtractedEntity {
            text: pattern.to_string(),
            entity_type: entity_type.clone(),
            start_offset: abs,
            end_offset: end,
            confidence,
        });
        start = abs + 1;
    }
    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_extractor() -> EntityExtractor {
        EntityExtractor::new(ExtractorConfig::default())
    }

    // -----------------------------------------------------------------------
    // Technology extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_tech_rust() {
        let e = default_extractor();
        let entities = e.extract("We use Rust for the backend.");
        assert!(
            entities
                .iter()
                .any(|x| x.text == "Rust" && x.entity_type == ExtractedEntityType::Technology)
        );
    }

    #[test]
    fn test_tech_multiple() {
        let e = default_extractor();
        let text = "The stack uses Python, PostgreSQL, and Docker.";
        let entities = e.extract(text);
        let techs: Vec<&str> = entities
            .iter()
            .filter(|x| x.entity_type == ExtractedEntityType::Technology)
            .map(|x| x.text.as_str())
            .collect();
        assert!(techs.contains(&"Python"), "missing Python");
        assert!(techs.contains(&"PostgreSQL"), "missing PostgreSQL");
        assert!(techs.contains(&"Docker"), "missing Docker");
    }

    #[test]
    fn test_tech_confidence() {
        let e = default_extractor();
        let entities = e.extract("Using Redis for caching.");
        let tech = entities
            .iter()
            .find(|x| x.text == "Redis")
            .expect("Redis not found");
        assert!((tech.confidence - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_tech_word_boundary() {
        let e = default_extractor();
        // "Rust" inside "Rustic" should NOT match.
        let entities = e.extract("The rustic cabin is beautiful.");
        assert!(!entities.iter().any(|x| x.text == "Rust"));
    }

    #[test]
    fn test_tech_disabled() {
        let mut config = ExtractorConfig::default();
        config.extract_technology = false;
        let e = EntityExtractor::new(config);
        let entities = e.extract("We use Rust and Docker.");
        assert!(
            !entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Technology)
        );
    }

    // -----------------------------------------------------------------------
    // Date extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_date_iso() {
        let e = default_extractor();
        let entities = e.extract("Deployed on 2024-03-19 at noon.");
        assert!(
            entities
                .iter()
                .any(|x| x.text == "2024-03-19" && x.entity_type == ExtractedEntityType::Date)
        );
    }

    #[test]
    fn test_date_iso_slash() {
        let e = default_extractor();
        let entities = e.extract("Report date: 2024/03/19.");
        assert!(
            entities
                .iter()
                .any(|x| x.text == "2024/03/19" && x.entity_type == ExtractedEntityType::Date)
        );
    }

    #[test]
    fn test_date_relative_yesterday() {
        let e = default_extractor();
        let entities = e.extract("I saw it yesterday at the office.");
        assert!(
            entities.iter().any(|x| x.text.to_lowercase() == "yesterday"
                && x.entity_type == ExtractedEntityType::Date)
        );
    }

    #[test]
    fn test_date_relative_last_week() {
        let e = default_extractor();
        let entities = e.extract("We merged the PR last week.");
        assert!(
            entities.iter().any(|x| x.text.to_lowercase() == "last week"
                && x.entity_type == ExtractedEntityType::Date)
        );
    }

    #[test]
    fn test_date_english_month_day() {
        let e = default_extractor();
        let entities = e.extract("Meeting on March 19 to discuss the roadmap.");
        assert!(
            entities
                .iter()
                .any(|x| x.text.starts_with("March") && x.entity_type == ExtractedEntityType::Date)
        );
    }

    #[test]
    fn test_date_disabled() {
        let mut config = ExtractorConfig::default();
        config.extract_dates = false;
        let e = EntityExtractor::new(config);
        let entities = e.extract("Released on 2024-03-19.");
        assert!(
            !entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Date)
        );
    }

    #[test]
    fn test_date_confidence() {
        let e = default_extractor();
        let entities = e.extract("Deadline: 2025-12-31.");
        let d = entities
            .iter()
            .find(|x| x.entity_type == ExtractedEntityType::Date)
            .expect("no date");
        assert!(d.confidence >= 0.9);
    }

    // -----------------------------------------------------------------------
    // Project extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_project_camelcase() {
        let e = default_extractor();
        let entities = e.extract("ClawBrainHub is the core component.");
        assert!(
            entities
                .iter()
                .any(|x| x.text == "ClawBrainHub" && x.entity_type == ExtractedEntityType::Project)
        );
    }

    #[test]
    fn test_project_kebab() {
        let e = default_extractor();
        let entities = e.extract("See the clawhdf5-agent crate for details.");
        assert!(
            entities.iter().any(
                |x| x.text == "clawhdf5-agent" && x.entity_type == ExtractedEntityType::Project
            )
        );
    }

    #[test]
    fn test_project_kebab_multi() {
        let e = default_extractor();
        let entities = e.extract("my-cool-project is production-ready.");
        assert!(
            entities
                .iter()
                .any(|x| x.text == "my-cool-project"
                    && x.entity_type == ExtractedEntityType::Project)
        );
    }

    #[test]
    fn test_project_confidence() {
        let e = default_extractor();
        let entities = e.extract("ClawBrainHub handles memory.");
        let proj = entities
            .iter()
            .find(|x| x.entity_type == ExtractedEntityType::Project)
            .expect("no project");
        assert!((proj.confidence - 0.7).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Capitalized phrase extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_capitalized_person() {
        let e = default_extractor();
        let entities = e.extract("I spoke with John Smith about the project.");
        assert!(
            entities
                .iter()
                .any(|x| x.text == "John Smith" && x.entity_type == ExtractedEntityType::Person)
        );
    }

    #[test]
    fn test_capitalized_org() {
        let e = default_extractor();
        let entities = e.extract("We partnered with Red Hat Systems for support.");
        assert!(
            entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Organization)
        );
    }

    #[test]
    fn test_capitalized_sentence_start_skipped() {
        let e = default_extractor();
        // "The" at sentence start should not be extracted.
        let entities = e.extract("The meeting was held yesterday.");
        assert!(!entities.iter().any(|x| x.text == "The"));
    }

    #[test]
    fn test_all_lowercase_no_cap_entities() {
        let e = default_extractor();
        let entities = e.extract("everything here is lowercase and has no entities.");
        assert!(
            !entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Person)
        );
        assert!(
            !entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Organization)
        );
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_text() {
        let e = default_extractor();
        let entities = e.extract("");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_confidence_filter() {
        let mut config = ExtractorConfig::default();
        config.min_confidence = 0.95;
        let e = EntityExtractor::new(config);
        // Only dates (0.95) and techs (0.9) should survive; 0.9 < 0.95 filters techs.
        let entities = e.extract("We use Rust since 2024-01-01.");
        assert!(
            !entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Technology)
        );
        assert!(
            entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Date)
        );
    }

    #[test]
    fn test_batch_dedup() {
        let e = default_extractor();
        let texts = ["We use Rust.", "Rust is fast.", "Also Rust for safety."];
        let entities = e.extract_batch(&texts.iter().map(|s| *s).collect::<Vec<_>>());
        let rust_count = entities.iter().filter(|x| x.text == "Rust").count();
        assert_eq!(rust_count, 1, "Rust should appear exactly once after dedup");
    }

    #[test]
    fn test_batch_multiple_types() {
        let e = default_extractor();
        let texts = ["Deploy with Docker.", "We merged last week."];
        let entities = e.extract_batch(&texts.iter().map(|s| *s).collect::<Vec<_>>());
        assert!(
            entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Technology)
        );
        assert!(
            entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Date)
        );
    }

    #[test]
    fn test_custom_pattern() {
        let mut config = ExtractorConfig::default();
        config.custom_patterns.push((
            "CRITICAL".to_string(),
            ExtractedEntityType::Custom("Alert".to_string()),
            0.99,
        ));
        let e = EntityExtractor::new(config);
        let entities = e.extract("CRITICAL failure detected.");
        assert!(
            entities
                .iter()
                .any(|x| x.entity_type == ExtractedEntityType::Custom("Alert".to_string()))
        );
    }

    #[test]
    fn test_offsets_correct() {
        let e = default_extractor();
        let text = "Released 2024-06-15 for testing.";
        let entities = e.extract(text);
        let d = entities
            .iter()
            .find(|x| x.entity_type == ExtractedEntityType::Date)
            .expect("no date");
        assert_eq!(&text[d.start_offset..d.end_offset], d.text);
    }

    #[test]
    fn test_no_false_positive_lowercased_tech() {
        let e = default_extractor();
        // "rust" lowercase is NOT in the TECHNOLOGIES list (case-sensitive).
        let entities = e.extract("The rust on the pipes was visible.");
        assert!(
            !entities
                .iter()
                .any(|x| x.text == "rust" && x.entity_type == ExtractedEntityType::Technology)
        );
    }
}
