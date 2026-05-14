//! Knowledge graph data structures and cache.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// RelationType
// ---------------------------------------------------------------------------

/// Typed classification for a knowledge graph relation.
#[derive(Debug, Clone, PartialEq)]
pub enum RelationType {
    Temporal,
    Causal,
    Associative,
    Hierarchical,
    Custom(String),
}

impl RelationType {
    /// Convert a string label to a `RelationType`.
    pub fn from_label(s: &str) -> Self {
        match s {
            "temporal" => RelationType::Temporal,
            "causal" => RelationType::Causal,
            "associative" => RelationType::Associative,
            "hierarchical" => RelationType::Hierarchical,
            other => RelationType::Custom(other.to_string()),
        }
    }

    /// Convert back to a canonical string label.
    pub fn as_str(&self) -> &str {
        match self {
            RelationType::Temporal => "temporal",
            RelationType::Causal => "causal",
            RelationType::Associative => "associative",
            RelationType::Hierarchical => "hierarchical",
            RelationType::Custom(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// Entity
// ---------------------------------------------------------------------------

/// A knowledge graph entity.
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: u64,
    pub name: String,
    pub entity_type: String,
    /// Index into the memory embeddings array, or -1 if none.
    pub embedding_idx: i64,
    /// Arbitrary key-value properties attached to this entity.
    pub properties: HashMap<String, String>,
    /// Optional dense embedding vector stored directly on the entity.
    pub embedding: Option<Vec<f32>>,
    /// Unix timestamp (microseconds) when this entity was first created.
    pub created_at: f64,
    /// Unix timestamp (microseconds) of the most recent update.
    pub updated_at: f64,
}

impl Default for Entity {
    fn default() -> Self {
        let now = current_ts_us();
        Self {
            id: 0,
            name: String::new(),
            entity_type: String::new(),
            embedding_idx: -1,
            properties: HashMap::new(),
            embedding: None,
            created_at: now,
            updated_at: now,
        }
    }
}

// ---------------------------------------------------------------------------
// Relation
// ---------------------------------------------------------------------------

/// A knowledge graph relation between two entities.
#[derive(Debug, Clone)]
pub struct Relation {
    pub src: u64,
    pub tgt: u64,
    pub relation: String,
    pub weight: f32,
    pub ts: f64,
    /// Arbitrary key-value metadata attached to this relation.
    pub metadata: HashMap<String, String>,
}

impl Default for Relation {
    fn default() -> Self {
        Self {
            src: 0,
            tgt: 0,
            relation: String::new(),
            weight: 1.0,
            ts: current_ts_us(),
            metadata: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the current wall-clock time in microseconds since Unix epoch.
/// Falls back to 0.0 if the system clock is unavailable.
fn current_ts_us() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1_000_000.0
}

/// Compute a simple edit-distance between two strings (Levenshtein).
/// Returns the number of single-character edits needed to transform `a` into `b`.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let na = a.len();
    let nb = b.len();

    if na == 0 {
        return nb;
    }
    if nb == 0 {
        return na;
    }

    let mut prev: Vec<usize> = (0..=nb).collect();
    let mut curr = vec![0usize; nb + 1];

    for i in 1..=na {
        curr[0] = i;
        for j in 1..=nb {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1).min(prev[j] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[nb]
}

// ---------------------------------------------------------------------------
// KnowledgeCache
// ---------------------------------------------------------------------------

/// In-memory cache for the /knowledge_graph group.
#[derive(Debug, Clone)]
pub struct KnowledgeCache {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub alias_strings: Vec<String>,
    pub alias_entity_ids: Vec<i64>,
    next_entity_id: u64,
}

impl KnowledgeCache {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            relations: Vec::new(),
            alias_strings: Vec::new(),
            alias_entity_ids: Vec::new(),
            next_entity_id: 0,
        }
    }

    pub fn new_with_next_id(next_id: u64) -> Self {
        Self {
            entities: Vec::new(),
            relations: Vec::new(),
            alias_strings: Vec::new(),
            alias_entity_ids: Vec::new(),
            next_entity_id: next_id,
        }
    }

    // -----------------------------------------------------------------------
    // Entity management
    // -----------------------------------------------------------------------

    /// Add an entity, returns its assigned ID.
    pub fn add_entity(&mut self, name: &str, entity_type: &str, embedding_idx: i64) -> u64 {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        let now = current_ts_us();
        self.entities.push(Entity {
            id,
            name: name.to_owned(),
            entity_type: entity_type.to_owned(),
            embedding_idx,
            properties: HashMap::new(),
            embedding: None,
            created_at: now,
            updated_at: now,
        });
        id
    }

    /// Find an entity by ID.
    pub fn get_entity(&self, id: u64) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    /// Find a mutable entity by ID.
    pub fn get_entity_mut(&mut self, id: u64) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    /// Get entity name by ID.
    pub fn get_entity_name(&self, entity_id: i64) -> Option<&str> {
        self.entities
            .iter()
            .find(|e| e.id == entity_id as u64)
            .map(|e| e.name.as_str())
    }

    // -----------------------------------------------------------------------
    // Relation management
    // -----------------------------------------------------------------------

    /// Add a relation between two entities.
    pub fn add_relation(&mut self, src: u64, tgt: u64, relation: &str, weight: f32) {
        let ts = current_ts_us();
        self.relations.push(Relation {
            src,
            tgt,
            relation: relation.to_owned(),
            weight,
            ts,
            metadata: HashMap::new(),
        });
    }

    /// Find all relations where the given entity is the source.
    pub fn get_relations_from(&self, src_id: u64) -> Vec<&Relation> {
        self.relations.iter().filter(|r| r.src == src_id).collect()
    }

    /// Find all relations where the given entity is the target.
    pub fn get_relations_to(&self, tgt_id: u64) -> Vec<&Relation> {
        self.relations.iter().filter(|r| r.tgt == tgt_id).collect()
    }

    // -----------------------------------------------------------------------
    // Alias management
    // -----------------------------------------------------------------------

    /// Register an alias for an entity. Case-insensitive storage.
    pub fn add_alias(&mut self, alias: &str, entity_id: i64) {
        self.alias_strings.push(alias.to_lowercase());
        self.alias_entity_ids.push(entity_id);
    }

    /// Get all aliases for a given entity.
    pub fn get_aliases(&self, entity_id: i64) -> Vec<&str> {
        self.alias_strings
            .iter()
            .zip(&self.alias_entity_ids)
            .filter(|&(_, id)| *id == entity_id)
            .map(|(s, _)| s.as_str())
            .collect()
    }

    /// Resolve aliases in free text — greedy longest-match replacement.
    pub fn resolve_aliases(&self, query: &str) -> String {
        let lower = query.to_lowercase();
        let mut pairs: Vec<(&str, String)> = self
            .alias_strings
            .iter()
            .zip(&self.alias_entity_ids)
            .filter_map(|(alias, &eid)| {
                self.get_entity_name(eid)
                    .map(|name| (alias.as_str(), name.to_lowercase()))
            })
            .collect();
        pairs.sort_by_key(|b| Reverse(b.0.len()));

        let mut result = lower;
        for (alias, name) in &pairs {
            result = result.replace(alias, name);
        }
        result
    }

    // -----------------------------------------------------------------------
    // Entity resolution
    // -----------------------------------------------------------------------

    /// Find an existing entity whose name is within `max_distance` Levenshtein
    /// edits of `name` (case-insensitive), or create a new entity if none is
    /// found.  Returns `(entity_id, was_created)`.
    pub fn resolve_or_create(
        &mut self,
        name: &str,
        entity_type: &str,
        embedding_idx: i64,
        max_distance: usize,
    ) -> (u64, bool) {
        let lower_name = name.to_lowercase();

        // Search for the closest existing entity.
        let best = self
            .entities
            .iter()
            .map(|e| {
                let dist = levenshtein(&lower_name, &e.name.to_lowercase());
                (e.id, dist)
            })
            .filter(|&(_, dist)| dist <= max_distance)
            .min_by_key(|&(_, dist)| dist);

        if let Some((id, _)) = best {
            return (id, false);
        }

        let id = self.add_entity(name, entity_type, embedding_idx);
        (id, true)
    }

    // -----------------------------------------------------------------------
    // Graph traversal: BFS neighbors
    // -----------------------------------------------------------------------

    /// Return all entities reachable from `entity_id` within `max_depth` hops,
    /// together with their discovered depth.  The seed entity itself is NOT
    /// included.  Traversal follows both outgoing and incoming relation edges.
    pub fn bfs_neighbors(&self, entity_id: u64, max_depth: usize) -> Vec<(Entity, usize)> {
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
        let mut results: Vec<(Entity, usize)> = Vec::new();

        visited.insert(entity_id);
        queue.push_back((entity_id, 0));

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Collect neighbour IDs from outgoing and incoming edges.
            let neighbours: Vec<u64> = self
                .relations
                .iter()
                .filter_map(|r| {
                    if r.src == current_id {
                        Some(r.tgt)
                    } else if r.tgt == current_id {
                        Some(r.src)
                    } else {
                        None
                    }
                })
                .collect();

            for neighbour_id in neighbours {
                if visited.insert(neighbour_id)
                    && let Some(entity) = self.get_entity(neighbour_id)
                {
                    results.push((entity.clone(), depth + 1));
                    queue.push_back((neighbour_id, depth + 1));
                }
            }
        }

        results
    }

    // -----------------------------------------------------------------------
    // Graph traversal: subgraph extraction
    // -----------------------------------------------------------------------

    /// Extract the subgraph reachable from any of `seed_ids` within `max_depth`
    /// hops.  Returns `(entities, relations)` where `relations` contains only
    /// those edges whose both endpoints are in the entity set.
    pub fn get_subgraph(&self, seed_ids: &[u64], max_depth: usize) -> (Vec<Entity>, Vec<Relation>) {
        let mut entity_ids: HashSet<u64> = HashSet::new();

        // Seed all starting nodes.
        for &seed in seed_ids {
            entity_ids.insert(seed);
        }

        // BFS from each seed, collecting entity IDs.
        for &seed in seed_ids {
            for (entity, _depth) in self.bfs_neighbors(seed, max_depth) {
                entity_ids.insert(entity.id);
            }
        }

        let entities: Vec<Entity> = self
            .entities
            .iter()
            .filter(|e| entity_ids.contains(&e.id))
            .cloned()
            .collect();

        let relations: Vec<Relation> = self
            .relations
            .iter()
            .filter(|r| entity_ids.contains(&r.src) && entity_ids.contains(&r.tgt))
            .cloned()
            .collect();

        (entities, relations)
    }

    // -----------------------------------------------------------------------
    // Spreading activation
    // -----------------------------------------------------------------------

    /// Compute spreading activation scores starting from `seed_ids`.
    ///
    /// The algorithm works as follows:
    /// 1. Each seed receives an initial activation of 1.0.
    /// 2. At each step, every activated node spreads activation to its
    ///    neighbours proportional to `relation.weight * decay_factor`.
    /// 3. A node's total activation is the sum of all received signals.
    /// 4. Propagation continues for up to `max_steps` rounds or until no node
    ///    has activation above `min_activation`.
    ///
    /// Returns `Vec<(entity_id, activation_score)>` sorted descending by score,
    /// excluding entities whose final score is below `min_activation`.
    pub fn spreading_activation(
        &self,
        seed_ids: &[u64],
        decay_factor: f32,
        min_activation: f32,
        max_steps: usize,
    ) -> Vec<(u64, f32)> {
        let mut activation: HashMap<u64, f32> = HashMap::new();

        // Initialise seeds with activation 1.0.
        for &seed in seed_ids {
            *activation.entry(seed).or_insert(0.0) += 1.0;
        }

        for _step in 0..max_steps {
            // Snapshot current activations.
            let current: Vec<(u64, f32)> = activation
                .iter()
                .filter(|&(_, &score)| score >= min_activation)
                .map(|(&id, &score)| (id, score))
                .collect();

            if current.is_empty() {
                break;
            }

            let mut any_spread = false;

            for (source_id, source_score) in current {
                // Spread to all neighbours via outgoing and incoming edges.
                for rel in &self.relations {
                    let neighbour_id = if rel.src == source_id {
                        rel.tgt
                    } else if rel.tgt == source_id {
                        rel.src
                    } else {
                        continue;
                    };

                    let delta = source_score * rel.weight * decay_factor;
                    if delta >= min_activation {
                        *activation.entry(neighbour_id).or_insert(0.0) += delta;
                        any_spread = true;
                    }
                }
            }

            if !any_spread {
                break;
            }
        }

        // Collect results, drop entries below threshold.
        let mut result: Vec<(u64, f32)> = activation
            .into_iter()
            .filter(|&(_, score)| score >= min_activation)
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    // -----------------------------------------------------------------------
    // Graph-aware search context
    // -----------------------------------------------------------------------

    /// Return a human-readable context string describing the neighbourhood of
    /// `entity_id`.  Includes the entity itself, all directly connected entities,
    /// and the relations linking them.  Suitable for injection into retrieval
    /// results.
    pub fn get_entity_context(&self, entity_id: u64) -> String {
        let entity = match self.get_entity(entity_id) {
            Some(e) => e,
            None => return format!("[entity {entity_id} not found]"),
        };

        let mut lines: Vec<String> = Vec::new();

        lines.push(format!(
            "Entity: {} (id={}, type={})",
            entity.name, entity.id, entity.entity_type
        ));

        if !entity.properties.is_empty() {
            let mut props: Vec<String> = entity
                .properties
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            props.sort();
            lines.push(format!("  Properties: {}", props.join(", ")));
        }

        // Outgoing relations.
        let outgoing = self.get_relations_from(entity_id);
        for rel in outgoing {
            if let Some(tgt) = self.get_entity(rel.tgt) {
                lines.push(format!(
                    "  -[{}]-> {} (id={}, weight={:.3})",
                    rel.relation, tgt.name, tgt.id, rel.weight
                ));
            }
        }

        // Incoming relations.
        let incoming = self.get_relations_to(entity_id);
        for rel in incoming {
            if let Some(src) = self.get_entity(rel.src) {
                lines.push(format!(
                    "  <-[{}]- {} (id={}, weight={:.3})",
                    rel.relation, src.name, src.id, rel.weight
                ));
            }
        }

        lines.join("\n")
    }
}

impl Default for KnowledgeCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Original tests — must remain passing
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_and_get_aliases() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        cache.add_alias("my son", id as i64);
        cache.add_alias("the kid", id as i64);

        let aliases = cache.get_aliases(id as i64);
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains(&"my son"));
        assert!(aliases.contains(&"the kid"));
    }

    #[test]
    fn test_resolve_single_alias() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        cache.add_alias("my son", id as i64);

        let resolved = cache.resolve_aliases("what does my son do?");
        assert!(
            resolved.contains("henry"),
            "expected 'henry' in '{resolved}'"
        );
    }

    #[test]
    fn test_resolve_multiple_aliases() {
        let mut cache = KnowledgeCache::new();
        let h = cache.add_entity("Henry", "person", -1);
        let a = cache.add_entity("Acme Corp", "company", -1);
        cache.add_alias("my son", h as i64);
        cache.add_alias("our main client", a as i64);

        let resolved = cache.resolve_aliases("what does my son do at our main client?");
        assert!(
            resolved.contains("henry"),
            "expected 'henry' in '{resolved}'"
        );
        assert!(
            resolved.contains("acme corp"),
            "expected 'acme corp' in '{resolved}'"
        );
    }

    #[test]
    fn test_longest_match_wins() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        cache.add_alias("my son", id as i64);
        cache.add_alias("my son henry", id as i64);

        let resolved = cache.resolve_aliases("ask my son henry");
        // The longer alias "my son henry" should match first, producing "ask henry"
        assert_eq!(resolved, "ask henry");
    }

    #[test]
    fn test_unregistered_alias_passthrough() {
        let mut cache = KnowledgeCache::new();
        cache.add_entity("Henry", "person", -1);
        let resolved = cache.resolve_aliases("unknown phrase here");
        assert_eq!(resolved, "unknown phrase here");
    }

    #[test]
    fn test_case_insensitive_resolve() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        cache.add_alias("henry", id as i64);

        let resolved = cache.resolve_aliases("Tell HENRY about it");
        assert!(
            resolved.contains("henry"),
            "expected 'henry' in '{resolved}'"
        );
    }

    #[test]
    fn test_empty_aliases() {
        let cache = KnowledgeCache::new();
        let resolved = cache.resolve_aliases("hello");
        assert_eq!(resolved, "hello");
    }

    #[test]
    fn test_get_entity_name() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        assert_eq!(cache.get_entity_name(id as i64), Some("Henry"));
        assert_eq!(cache.get_entity_name(999), None);
    }

    // -----------------------------------------------------------------------
    // Entity properties
    // -----------------------------------------------------------------------

    #[test]
    fn test_entity_properties_default_empty() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Alice", "person", -1);
        let entity = cache.get_entity(id).unwrap();
        assert!(entity.properties.is_empty());
    }

    #[test]
    fn test_entity_properties_set_and_get() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Alice", "person", -1);
        {
            let entity = cache.get_entity_mut(id).unwrap();
            entity
                .properties
                .insert("role".to_string(), "engineer".to_string());
        }
        let entity = cache.get_entity(id).unwrap();
        assert_eq!(
            entity.properties.get("role").map(String::as_str),
            Some("engineer")
        );
    }

    // -----------------------------------------------------------------------
    // Entity embeddings
    // -----------------------------------------------------------------------

    #[test]
    fn test_entity_embedding_default_none() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Bob", "person", -1);
        let entity = cache.get_entity(id).unwrap();
        assert!(entity.embedding.is_none());
    }

    #[test]
    fn test_entity_embedding_set_and_get() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Bob", "person", -1);
        let vec = vec![0.1_f32, 0.2, 0.3];
        {
            let entity = cache.get_entity_mut(id).unwrap();
            entity.embedding = Some(vec.clone());
        }
        let entity = cache.get_entity(id).unwrap();
        assert_eq!(entity.embedding.as_deref(), Some(vec.as_slice()));
    }

    // -----------------------------------------------------------------------
    // Entity timestamps
    // -----------------------------------------------------------------------

    #[test]
    fn test_entity_timestamps_set_on_creation() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Carol", "person", -1);
        let entity = cache.get_entity(id).unwrap();
        assert!(entity.created_at > 0.0);
        assert!(entity.updated_at > 0.0);
        assert_eq!(entity.created_at, entity.updated_at);
    }

    #[test]
    fn test_entity_updated_at_can_be_changed() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Carol", "person", -1);
        let original_created = cache.get_entity(id).unwrap().created_at;
        {
            let entity = cache.get_entity_mut(id).unwrap();
            entity.updated_at = original_created + 1000.0;
        }
        let entity = cache.get_entity(id).unwrap();
        assert_eq!(entity.created_at, original_created);
        assert!(entity.updated_at > entity.created_at);
    }

    // -----------------------------------------------------------------------
    // Relation metadata
    // -----------------------------------------------------------------------

    #[test]
    fn test_relation_metadata_default_empty() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        cache.add_relation(a, b, "link", 1.0);
        let rels = cache.get_relations_from(a);
        assert_eq!(rels.len(), 1);
        assert!(rels[0].metadata.is_empty());
    }

    // -----------------------------------------------------------------------
    // RelationType enum
    // -----------------------------------------------------------------------

    #[test]
    fn test_relation_type_from_str_known_variants() {
        assert_eq!(RelationType::from_label("temporal"), RelationType::Temporal);
        assert_eq!(RelationType::from_label("causal"), RelationType::Causal);
        assert_eq!(
            RelationType::from_label("associative"),
            RelationType::Associative
        );
        assert_eq!(
            RelationType::from_label("hierarchical"),
            RelationType::Hierarchical
        );
    }

    #[test]
    fn test_relation_type_custom() {
        let rt = RelationType::from_label("something_else");
        assert_eq!(rt, RelationType::Custom("something_else".to_string()));
        assert_eq!(rt.as_str(), "something_else");
    }

    #[test]
    fn test_relation_type_round_trip() {
        let variants = [
            RelationType::Temporal,
            RelationType::Causal,
            RelationType::Associative,
            RelationType::Hierarchical,
            RelationType::Custom("my_type".to_string()),
        ];
        for v in &variants {
            assert_eq!(RelationType::from_label(v.as_str()), *v);
        }
    }

    // -----------------------------------------------------------------------
    // Levenshtein helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty_strings() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn test_levenshtein_one_edit() {
        assert_eq!(levenshtein("cat", "bat"), 1);
    }

    #[test]
    fn test_levenshtein_insertions() {
        assert_eq!(levenshtein("ab", "abc"), 1);
    }

    // -----------------------------------------------------------------------
    // resolve_or_create
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_or_create_creates_new() {
        let mut cache = KnowledgeCache::new();
        let (id, created) = cache.resolve_or_create("NewEntity", "thing", -1, 0);
        assert!(created);
        assert!(cache.get_entity(id).is_some());
    }

    #[test]
    fn test_resolve_or_create_finds_exact_match() {
        let mut cache = KnowledgeCache::new();
        let orig_id = cache.add_entity("Alice", "person", -1);
        let (id, created) = cache.resolve_or_create("Alice", "person", -1, 0);
        assert!(!created);
        assert_eq!(id, orig_id);
    }

    #[test]
    fn test_resolve_or_create_fuzzy_match_within_threshold() {
        let mut cache = KnowledgeCache::new();
        let orig_id = cache.add_entity("Alice", "person", -1);
        // "Alyce" has edit distance 1 from "Alice"
        let (id, created) = cache.resolve_or_create("Alyce", "person", -1, 2);
        assert!(!created, "expected fuzzy match, got a new entity");
        assert_eq!(id, orig_id);
    }

    #[test]
    fn test_resolve_or_create_no_match_beyond_threshold() {
        let mut cache = KnowledgeCache::new();
        cache.add_entity("Alice", "person", -1);
        // "Zebra" is far from "Alice"
        let (_id, created) = cache.resolve_or_create("Zebra", "animal", -1, 2);
        assert!(created, "expected a new entity to be created");
        assert_eq!(cache.entities.len(), 2);
    }

    #[test]
    fn test_resolve_or_create_case_insensitive() {
        let mut cache = KnowledgeCache::new();
        let orig_id = cache.add_entity("Alice", "person", -1);
        // Exact match after lower-casing
        let (id, created) = cache.resolve_or_create("ALICE", "person", -1, 0);
        assert!(!created);
        assert_eq!(id, orig_id);
    }

    // -----------------------------------------------------------------------
    // bfs_neighbors
    // -----------------------------------------------------------------------

    #[test]
    fn test_bfs_neighbors_empty_graph() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Solo", "node", -1);
        let result = cache.bfs_neighbors(id, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bfs_neighbors_depth_one() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        let c = cache.add_entity("C", "node", -1);
        cache.add_relation(a, b, "link", 1.0);
        cache.add_relation(a, c, "link", 1.0);

        let result = cache.bfs_neighbors(a, 1);
        assert_eq!(result.len(), 2);
        let depths: Vec<usize> = result.iter().map(|(_, d)| *d).collect();
        assert!(depths.iter().all(|&d| d == 1));
    }

    #[test]
    fn test_bfs_neighbors_max_depth_zero_returns_empty() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        cache.add_relation(a, b, "link", 1.0);

        let result = cache.bfs_neighbors(a, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bfs_neighbors_respects_max_depth() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        let c = cache.add_entity("C", "node", -1);
        cache.add_relation(a, b, "link", 1.0);
        cache.add_relation(b, c, "link", 1.0);

        // depth=1: only B reachable
        let result = cache.bfs_neighbors(a, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.id, b);

        // depth=2: both B and C reachable
        let result2 = cache.bfs_neighbors(a, 2);
        assert_eq!(result2.len(), 2);
    }

    #[test]
    fn test_bfs_neighbors_follows_incoming_edges() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        // Relation goes B -> A, but BFS from A should still find B.
        cache.add_relation(b, a, "link", 1.0);

        let result = cache.bfs_neighbors(a, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.id, b);
    }

    #[test]
    fn test_bfs_neighbors_no_duplicate_visits() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        // Two edges between A and B.
        cache.add_relation(a, b, "link1", 1.0);
        cache.add_relation(a, b, "link2", 1.0);

        let result = cache.bfs_neighbors(a, 2);
        assert_eq!(result.len(), 1, "B should appear exactly once");
    }

    // -----------------------------------------------------------------------
    // get_subgraph
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_subgraph_single_seed_no_neighbours() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);

        let (entities, relations) = cache.get_subgraph(&[a], 2);
        assert_eq!(entities.len(), 1);
        assert!(relations.is_empty());
    }

    #[test]
    fn test_get_subgraph_includes_seed_and_neighbours() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        let c = cache.add_entity("C", "node", -1);
        // D is disconnected.
        let _d = cache.add_entity("D", "node", -1);
        cache.add_relation(a, b, "link", 1.0);
        cache.add_relation(b, c, "link", 1.0);

        let (entities, relations) = cache.get_subgraph(&[a], 2);
        let ids: HashSet<u64> = entities.iter().map(|e| e.id).collect();
        assert!(ids.contains(&a));
        assert!(ids.contains(&b));
        assert!(ids.contains(&c));
        assert!(!ids.contains(&_d));
        assert_eq!(relations.len(), 2);
    }

    #[test]
    fn test_get_subgraph_multiple_seeds() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        let c = cache.add_entity("C", "node", -1);
        // No edges, but both seeds should appear.
        let (entities, _) = cache.get_subgraph(&[a, b], 1);
        let ids: HashSet<u64> = entities.iter().map(|e| e.id).collect();
        assert!(ids.contains(&a));
        assert!(ids.contains(&b));
        assert!(!ids.contains(&c));
    }

    // -----------------------------------------------------------------------
    // spreading_activation
    // -----------------------------------------------------------------------

    #[test]
    fn test_spreading_activation_single_seed_no_edges() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);

        let result = cache.spreading_activation(&[a], 0.5, 0.01, 5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, a);
        assert!((result[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spreading_activation_propagates_to_neighbour() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        cache.add_relation(a, b, "link", 1.0);

        let result = cache.spreading_activation(&[a], 0.5, 0.01, 3);
        // B should have received activation from A.
        let b_score = result.iter().find(|&&(id, _)| id == b).map(|&(_, s)| s);
        assert!(b_score.is_some());
        assert!(b_score.unwrap() > 0.0);
    }

    #[test]
    fn test_spreading_activation_decay_reduces_signal() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        let c = cache.add_entity("C", "node", -1);
        cache.add_relation(a, b, "link", 1.0);
        cache.add_relation(b, c, "link", 1.0);

        // Use 1 step to test pure decay without iterative backflow accumulation.
        let result = cache.spreading_activation(&[a], 0.5, 0.01, 1);

        let score_of = |id: u64| -> f32 {
            result
                .iter()
                .find(|&&(eid, _)| eid == id)
                .map(|&(_, s)| s)
                .unwrap_or(0.0)
        };

        // A starts with 1.0; B gets 0.5 after 1 step; C gets nothing (2 hops, only 1 step).
        assert!(
            score_of(a) >= score_of(b),
            "A should have higher or equal activation than B"
        );
        assert!(
            score_of(b) > score_of(c),
            "B should have more activation than C (C unreachable in 1 step)"
        );
    }

    #[test]
    fn test_spreading_activation_sorted_descending() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        let c = cache.add_entity("C", "node", -1);
        cache.add_relation(a, b, "link", 1.0);
        cache.add_relation(b, c, "link", 1.0);

        let result = cache.spreading_activation(&[a], 0.5, 0.01, 5);
        let scores: Vec<f32> = result.iter().map(|&(_, s)| s).collect();
        for window in scores.windows(2) {
            assert!(window[0] >= window[1], "results must be sorted descending");
        }
    }

    #[test]
    fn test_spreading_activation_min_activation_filter() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("A", "node", -1);
        let b = cache.add_entity("B", "node", -1);
        cache.add_relation(a, b, "link", 0.01);

        // With a very high min_activation, only the seed should appear.
        let result = cache.spreading_activation(&[a], 0.5, 10.0, 5);
        assert_eq!(
            result.len(),
            0,
            "all activations below min threshold should be filtered"
        );
    }

    // -----------------------------------------------------------------------
    // get_entity_context
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_entity_context_unknown_entity() {
        let cache = KnowledgeCache::new();
        let ctx = cache.get_entity_context(999);
        assert!(ctx.contains("not found"));
    }

    #[test]
    fn test_get_entity_context_basic_info() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Alice", "person", -1);
        let ctx = cache.get_entity_context(id);
        assert!(ctx.contains("Alice"));
        assert!(ctx.contains("person"));
    }

    #[test]
    fn test_get_entity_context_includes_outgoing_relations() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("Alice", "person", -1);
        let b = cache.add_entity("Bob", "person", -1);
        cache.add_relation(a, b, "knows", 0.9);

        let ctx = cache.get_entity_context(a);
        assert!(
            ctx.contains("knows"),
            "context should mention the relation type"
        );
        assert!(
            ctx.contains("Bob"),
            "context should mention the target entity"
        );
    }

    #[test]
    fn test_get_entity_context_includes_incoming_relations() {
        let mut cache = KnowledgeCache::new();
        let a = cache.add_entity("Alice", "person", -1);
        let b = cache.add_entity("Bob", "person", -1);
        cache.add_relation(b, a, "trusts", 0.8);

        let ctx = cache.get_entity_context(a);
        assert!(ctx.contains("trusts"));
        assert!(ctx.contains("Bob"));
    }

    #[test]
    fn test_get_entity_context_includes_properties() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Alice", "person", -1);
        {
            let entity = cache.get_entity_mut(id).unwrap();
            entity
                .properties
                .insert("occupation".to_string(), "engineer".to_string());
        }
        let ctx = cache.get_entity_context(id);
        assert!(ctx.contains("occupation"));
        assert!(ctx.contains("engineer"));
    }
}
