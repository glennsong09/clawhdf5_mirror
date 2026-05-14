//! Temporal reasoning primitives for agent memory.
//!
//! Provides:
//! - [`TemporalIndex`]   — sorted (id, timestamp) index with range/slice queries
//! - [`SessionDAG`]      — directed-acyclic-graph of sessions linked by continuation
//! - [`TemporalReRanker`] — boost search scores based on a temporal query hint
//! - [`EntityTimeline`]  — property-level change log with point-in-time state reconstruction

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TemporalIndex
// ---------------------------------------------------------------------------

/// A sorted index mapping `record_id -> timestamp`.
///
/// Internally keeps a `Vec<(f64, u64)>` (timestamp-first for cheap sorting)
/// maintained in ascending timestamp order via `binary_search`.
#[derive(Debug, Default, Clone)]
pub struct TemporalIndex {
    /// Sorted ascending by timestamp. Secondary sort by id for determinism.
    entries: Vec<(f64, u64)>,
}

impl TemporalIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a (record_id, timestamp) pair, maintaining sorted order.
    /// Duplicate (id, timestamp) pairs are allowed — callers should deduplicate
    /// via [`remove`] before re-inserting if they want upsert semantics.
    pub fn insert(&mut self, id: u64, timestamp: f64) {
        let key = (timestamp, id);
        let pos = self.entries.partition_point(|&e| e < key);
        self.entries.insert(pos, key);
    }

    /// Remove the entry for `id` from the index (all occurrences).
    pub fn remove(&mut self, id: u64) {
        self.entries.retain(|&(_, eid)| eid != id);
    }

    /// All record IDs whose timestamp is within `[start_ts, end_ts]`.
    pub fn range_query(&self, start_ts: f64, end_ts: f64) -> Vec<u64> {
        let lo = self.entries.partition_point(|&(ts, _)| ts < start_ts);
        let hi = self.entries.partition_point(|&(ts, _)| ts <= end_ts);
        self.entries[lo..hi].iter().map(|&(_, id)| id).collect()
    }

    /// Return the `n` most recent record IDs (highest timestamps), newest first.
    pub fn latest(&self, n: usize) -> Vec<u64> {
        self.entries
            .iter()
            .rev()
            .take(n)
            .map(|&(_, id)| id)
            .collect()
    }

    /// Return the `n` oldest record IDs (lowest timestamps), oldest first.
    pub fn earliest(&self, n: usize) -> Vec<u64> {
        self.entries.iter().take(n).map(|&(_, id)| id).collect()
    }

    /// Return up to `n` records strictly *before* `ts`, most-recent-first.
    pub fn before(&self, ts: f64, n: usize) -> Vec<u64> {
        let hi = self.entries.partition_point(|&(t, _)| t < ts);
        self.entries[..hi]
            .iter()
            .rev()
            .take(n)
            .map(|&(_, id)| id)
            .collect()
    }

    /// Return up to `n` records strictly *after* `ts`, oldest-first.
    pub fn after(&self, ts: f64, n: usize) -> Vec<u64> {
        let lo = self.entries.partition_point(|&(t, _)| t <= ts);
        self.entries[lo..]
            .iter()
            .take(n)
            .map(|&(_, id)| id)
            .collect()
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SessionDAG
// ---------------------------------------------------------------------------

/// A single node in the session DAG.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionNode {
    pub session_id: String,
    pub start_ts: f64,
    pub end_ts: Option<f64>,
    pub parent_session: Option<String>,
    pub tags: Vec<String>,
}

/// A directed-acyclic graph of sessions linked by "continuation" edges
/// (parent → child).
#[derive(Debug, Default)]
pub struct SessionDAG {
    /// session_id → node
    nodes: HashMap<String, SessionNode>,
    /// parent_id → list of child_ids
    children: HashMap<String, Vec<String>>,
}

impl SessionDAG {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a session node.  If a node with the same id already exists it is
    /// replaced.
    pub fn add_session(&mut self, node: SessionNode) {
        self.nodes.insert(node.session_id.clone(), node);
    }

    /// Mark `child_id` as a continuation of `parent_id`.
    ///
    /// Updates the child node's `parent_session` field and records the edge.
    pub fn link_continuation(&mut self, parent_id: &str, child_id: &str) {
        if let Some(child) = self.nodes.get_mut(child_id) {
            child.parent_session = Some(parent_id.to_owned());
        }
        self.children
            .entry(parent_id.to_owned())
            .or_default()
            .push(child_id.to_owned());
    }

    /// Walk the parent chain from `session_id` to the root, returning the
    /// chain in root-first order.
    pub fn get_session_chain(&self, session_id: &str) -> Vec<SessionNode> {
        let mut chain = Vec::new();
        let mut current = session_id.to_owned();
        let mut visited = std::collections::HashSet::new();

        loop {
            if visited.contains(&current) {
                break; // cycle guard
            }
            visited.insert(current.clone());

            match self.nodes.get(&current) {
                None => break,
                Some(node) => {
                    chain.push(node.clone());
                    match &node.parent_session {
                        None => break,
                        Some(p) => current = p.clone(),
                    }
                }
            }
        }

        chain.reverse(); // root-first
        chain
    }

    /// Direct children of `session_id`.
    pub fn get_children(&self, session_id: &str) -> Vec<SessionNode> {
        self.children
            .get(session_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.nodes.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// All sessions whose interval overlaps `[start, end]`.
    ///
    /// A session with no `end_ts` is treated as still-open (end = +∞).
    pub fn get_sessions_in_range(&self, start: f64, end: f64) -> Vec<SessionNode> {
        let mut result: Vec<SessionNode> = self
            .nodes
            .values()
            .filter(|n| {
                let session_end = n.end_ts.unwrap_or(f64::INFINITY);
                // overlap: session_start <= end AND session_end >= start
                n.start_ts <= end && session_end >= start
            })
            .cloned()
            .collect();
        result.sort_by(|a, b| a.start_ts.partial_cmp(&b.start_ts).unwrap());
        result
    }

    /// All sessions sorted by `start_ts` ascending.
    pub fn get_all_sessions_sorted(&self) -> Vec<SessionNode> {
        let mut all: Vec<SessionNode> = self.nodes.values().cloned().collect();
        all.sort_by(|a, b| a.start_ts.partial_cmp(&b.start_ts).unwrap());
        all
    }
}

// ---------------------------------------------------------------------------
// TemporalReRanker
// ---------------------------------------------------------------------------

/// A hint describing the temporal preference of a query.
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalHint {
    /// Prefer recent records.
    Latest,
    /// Prefer old records.
    Earliest,
    /// Prefer records near this timestamp.
    Around(f64),
    /// Prefer records whose timestamp falls in [lo, hi].
    Between(f64, f64),
    /// No temporal preference; boost is always 0.
    None,
}

/// Computes a temporal boost score in `[-1.0, 1.0]` for a single result.
pub struct TemporalReRanker;

impl TemporalReRanker {
    /// Returns a boost in `[0.0, 1.0]`.
    ///
    /// - `result_timestamp` — the timestamp of the candidate record.
    /// - `query_hint`       — the caller's temporal preference.
    /// - `now`              — the current wall-clock timestamp (same units as
    ///   all other timestamps in the system).
    pub fn temporal_boost(result_timestamp: f64, query_hint: &TemporalHint, now: f64) -> f32 {
        match query_hint {
            TemporalHint::None => 0.0,

            TemporalHint::Latest => {
                // Sigmoid-style decay: newer → closer to 1.
                // age ∈ [0, ∞), boost ∈ (0, 1]
                let age = (now - result_timestamp).max(0.0);
                // half-life of 86_400 s (one day) by default
                let half_life = 86_400.0_f64;
                (-(age / half_life) * std::f64::consts::LN_2).exp() as f32
            }

            TemporalHint::Earliest => {
                // Inverse of Latest: older → closer to 1.
                let age = (now - result_timestamp).max(0.0);
                let half_life = 86_400.0_f64;
                let recency = (-(age / half_life) * std::f64::consts::LN_2).exp();
                (1.0 - recency) as f32
            }

            TemporalHint::Around(target) => {
                // Gaussian centred on `target` with σ = 1 day.
                let sigma = 86_400.0_f64;
                let diff = result_timestamp - target;
                (-(diff * diff) / (2.0 * sigma * sigma)).exp() as f32
            }

            TemporalHint::Between(lo, hi) => {
                if result_timestamp >= *lo && result_timestamp <= *hi {
                    1.0_f32
                } else {
                    // Decay linearly from the nearest boundary.
                    let dist = if result_timestamp < *lo {
                        lo - result_timestamp
                    } else {
                        result_timestamp - hi
                    };
                    let sigma = 86_400.0_f64;
                    (-(dist / sigma)).exp() as f32
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EntityTimeline
// ---------------------------------------------------------------------------

/// A single property change event.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyChange {
    pub timestamp: f64,
    pub property_key: String,
    pub old_value: String,
    pub new_value: String,
}

/// Tracks the change history of named entities and can reconstruct state at
/// any point in time.
#[derive(Debug, Default)]
pub struct EntityTimeline {
    /// entity_id → sorted list of changes
    history: HashMap<String, Vec<PropertyChange>>,
}

impl EntityTimeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a property change for `entity_id` at `timestamp`.
    pub fn track_entity_state(
        &mut self,
        entity_id: &str,
        timestamp: f64,
        property_key: &str,
        old_value: &str,
        new_value: &str,
    ) {
        let changes = self.history.entry(entity_id.to_owned()).or_default();
        let change = PropertyChange {
            timestamp,
            property_key: property_key.to_owned(),
            old_value: old_value.to_owned(),
            new_value: new_value.to_owned(),
        };
        // Keep sorted by timestamp
        let pos = changes.partition_point(|c| c.timestamp <= timestamp);
        changes.insert(pos, change);
    }

    /// Full change log for `entity_id`, sorted by timestamp ascending.
    pub fn get_entity_history(&self, entity_id: &str) -> Vec<(f64, String, String, String)> {
        self.history
            .get(entity_id)
            .map(|changes| {
                changes
                    .iter()
                    .map(|c| {
                        (
                            c.timestamp,
                            c.property_key.clone(),
                            c.old_value.clone(),
                            c.new_value.clone(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Reconstruct the state of `entity_id` at `timestamp` by replaying all
    /// changes whose timestamp is ≤ `timestamp`.
    ///
    /// Returns a map of `property_key → current_value` at that instant.
    pub fn get_entity_state_at(&self, entity_id: &str, timestamp: f64) -> HashMap<String, String> {
        let mut state: HashMap<String, String> = HashMap::new();

        if let Some(changes) = self.history.get(entity_id) {
            for change in changes {
                if change.timestamp > timestamp {
                    break;
                }
                state.insert(change.property_key.clone(), change.new_value.clone());
            }
        }

        state
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TemporalIndex
    // -----------------------------------------------------------------------

    #[test]
    fn test_temporal_index_insert_and_range() {
        let mut idx = TemporalIndex::new();
        idx.insert(1, 100.0);
        idx.insert(2, 200.0);
        idx.insert(3, 300.0);
        idx.insert(4, 400.0);

        let r = idx.range_query(150.0, 350.0);
        assert_eq!(r, vec![2, 3]);
    }

    #[test]
    fn test_temporal_index_range_inclusive_boundaries() {
        let mut idx = TemporalIndex::new();
        idx.insert(10, 100.0);
        idx.insert(20, 200.0);
        idx.insert(30, 300.0);

        // Both boundaries inclusive
        assert_eq!(idx.range_query(100.0, 300.0), vec![10, 20, 30]);
        assert_eq!(idx.range_query(100.0, 100.0), vec![10]);
        assert_eq!(idx.range_query(300.0, 300.0), vec![30]);
    }

    #[test]
    fn test_temporal_index_latest() {
        let mut idx = TemporalIndex::new();
        idx.insert(1, 1.0);
        idx.insert(2, 2.0);
        idx.insert(3, 3.0);
        idx.insert(4, 4.0);

        assert_eq!(idx.latest(2), vec![4, 3]);
        assert_eq!(idx.latest(10), vec![4, 3, 2, 1]); // clamps to available
    }

    #[test]
    fn test_temporal_index_earliest() {
        let mut idx = TemporalIndex::new();
        idx.insert(1, 10.0);
        idx.insert(2, 20.0);
        idx.insert(3, 30.0);

        assert_eq!(idx.earliest(2), vec![1, 2]);
    }

    #[test]
    fn test_temporal_index_before() {
        let mut idx = TemporalIndex::new();
        for i in 1u64..=5 {
            idx.insert(i, i as f64 * 10.0);
        }
        // Before ts=35: entries at 10, 20, 30 — most-recent-first
        let r = idx.before(35.0, 2);
        assert_eq!(r, vec![3, 2]);
    }

    #[test]
    fn test_temporal_index_after() {
        let mut idx = TemporalIndex::new();
        for i in 1u64..=5 {
            idx.insert(i, i as f64 * 10.0);
        }
        // After ts=30: entries at 40, 50 — oldest-first
        let r = idx.after(30.0, 2);
        assert_eq!(r, vec![4, 5]);
    }

    #[test]
    fn test_temporal_index_remove() {
        let mut idx = TemporalIndex::new();
        idx.insert(1, 1.0);
        idx.insert(2, 2.0);
        idx.insert(3, 3.0);

        idx.remove(2);
        assert_eq!(idx.range_query(0.0, 10.0), vec![1, 3]);
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_temporal_index_empty() {
        let idx = TemporalIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.range_query(0.0, 100.0), vec![]);
        assert_eq!(idx.latest(5), vec![]);
        assert_eq!(idx.earliest(5), vec![]);
    }

    #[test]
    fn test_temporal_index_insert_order_invariant() {
        let mut idx = TemporalIndex::new();
        // Insert out of order
        idx.insert(3, 300.0);
        idx.insert(1, 100.0);
        idx.insert(2, 200.0);

        assert_eq!(idx.earliest(3), vec![1, 2, 3]);
        assert_eq!(idx.latest(3), vec![3, 2, 1]);
    }

    // -----------------------------------------------------------------------
    // SessionDAG
    // -----------------------------------------------------------------------

    fn make_session(id: &str, start: f64, end: Option<f64>) -> SessionNode {
        SessionNode {
            session_id: id.to_owned(),
            start_ts: start,
            end_ts: end,
            parent_session: None,
            tags: vec![],
        }
    }

    #[test]
    fn test_session_dag_add_and_sorted() {
        let mut dag = SessionDAG::new();
        dag.add_session(make_session("b", 200.0, Some(300.0)));
        dag.add_session(make_session("a", 100.0, Some(150.0)));
        dag.add_session(make_session("c", 300.0, None));

        let sorted = dag.get_all_sessions_sorted();
        assert_eq!(
            sorted
                .iter()
                .map(|n| n.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn test_session_dag_chain() {
        let mut dag = SessionDAG::new();
        dag.add_session(make_session("root", 0.0, Some(100.0)));
        dag.add_session(make_session("mid", 100.0, Some(200.0)));
        dag.add_session(make_session("leaf", 200.0, None));

        dag.link_continuation("root", "mid");
        dag.link_continuation("mid", "leaf");

        let chain = dag.get_session_chain("leaf");
        assert_eq!(
            chain
                .iter()
                .map(|n| n.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["root", "mid", "leaf"]
        );
    }

    #[test]
    fn test_session_dag_chain_single_node() {
        let mut dag = SessionDAG::new();
        dag.add_session(make_session("solo", 0.0, None));

        let chain = dag.get_session_chain("solo");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].session_id, "solo");
    }

    #[test]
    fn test_session_dag_chain_missing() {
        let dag = SessionDAG::new();
        assert!(dag.get_session_chain("nonexistent").is_empty());
    }

    #[test]
    fn test_session_dag_get_children() {
        let mut dag = SessionDAG::new();
        dag.add_session(make_session("p", 0.0, Some(100.0)));
        dag.add_session(make_session("c1", 100.0, Some(200.0)));
        dag.add_session(make_session("c2", 100.0, Some(200.0)));

        dag.link_continuation("p", "c1");
        dag.link_continuation("p", "c2");

        let mut children: Vec<String> = dag
            .get_children("p")
            .into_iter()
            .map(|n| n.session_id)
            .collect();
        children.sort();
        assert_eq!(children, vec!["c1", "c2"]);
    }

    #[test]
    fn test_session_dag_in_range() {
        let mut dag = SessionDAG::new();
        dag.add_session(make_session("early", 0.0, Some(50.0)));
        dag.add_session(make_session("overlap", 40.0, Some(120.0)));
        dag.add_session(make_session("late", 200.0, None));

        let r = dag.get_sessions_in_range(45.0, 100.0);
        let ids: Vec<&str> = r.iter().map(|n| n.session_id.as_str()).collect();
        assert!(ids.contains(&"overlap"));
        // "early" ends at 50 which overlaps [45, 100]
        assert!(ids.contains(&"early"));
        // "late" starts at 200, after range end
        assert!(!ids.contains(&"late"));
    }

    #[test]
    fn test_session_dag_open_session_in_range() {
        let mut dag = SessionDAG::new();
        // Open session started before range — should appear
        dag.add_session(make_session("open", 50.0, None));

        let r = dag.get_sessions_in_range(100.0, 200.0);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].session_id, "open");
    }

    #[test]
    fn test_session_dag_parent_field_updated() {
        let mut dag = SessionDAG::new();
        dag.add_session(make_session("parent", 0.0, None));
        dag.add_session(make_session("child", 10.0, None));
        dag.link_continuation("parent", "child");

        assert_eq!(
            dag.nodes.get("child").unwrap().parent_session,
            Some("parent".to_owned())
        );
    }

    // -----------------------------------------------------------------------
    // TemporalReRanker
    // -----------------------------------------------------------------------

    #[test]
    fn test_reranker_none() {
        let boost = TemporalReRanker::temporal_boost(1000.0, &TemporalHint::None, 2000.0);
        assert_eq!(boost, 0.0);
    }

    #[test]
    fn test_reranker_latest_recent_beats_old() {
        let now = 1_000_000.0_f64;
        let recent = now - 3600.0; // 1 hour ago
        let old = now - 864_000.0; // 10 days ago

        let b_recent = TemporalReRanker::temporal_boost(recent, &TemporalHint::Latest, now);
        let b_old = TemporalReRanker::temporal_boost(old, &TemporalHint::Latest, now);
        assert!(b_recent > b_old, "recent={b_recent} should > old={b_old}");
    }

    #[test]
    fn test_reranker_earliest_old_beats_recent() {
        let now = 1_000_000.0_f64;
        let recent = now - 3600.0;
        let old = now - 864_000.0;

        let b_recent = TemporalReRanker::temporal_boost(recent, &TemporalHint::Earliest, now);
        let b_old = TemporalReRanker::temporal_boost(old, &TemporalHint::Earliest, now);
        assert!(b_old > b_recent, "old={b_old} should > recent={b_recent}");
    }

    #[test]
    fn test_reranker_around_peak_at_target() {
        let target = 500_000.0_f64;
        let hint = TemporalHint::Around(target);
        let now = 1_000_000.0;

        let at_target = TemporalReRanker::temporal_boost(target, &hint, now);
        let off = TemporalReRanker::temporal_boost(target + 86_400.0, &hint, now);

        assert!(at_target > off);
        assert!((at_target - 1.0).abs() < 1e-5, "should be 1.0 at target");
    }

    #[test]
    fn test_reranker_between_inside_is_one() {
        let hint = TemporalHint::Between(100.0, 200.0);
        let now = 500.0;

        let inside = TemporalReRanker::temporal_boost(150.0, &hint, now);
        assert_eq!(inside, 1.0);
    }

    #[test]
    fn test_reranker_between_outside_decays() {
        let hint = TemporalHint::Between(100.0, 200.0);
        let now = 500.0;

        let outside = TemporalReRanker::temporal_boost(0.0, &hint, now);
        assert!(outside > 0.0 && outside < 1.0);
    }

    #[test]
    fn test_reranker_boost_range() {
        // All boosts must be in [0, 1]
        let cases = vec![
            TemporalHint::Latest,
            TemporalHint::Earliest,
            TemporalHint::Around(500.0),
            TemporalHint::Between(100.0, 200.0),
            TemporalHint::None,
        ];
        let now = 1000.0_f64;
        for hint in &cases {
            for ts in [0.0, 100.0, 500.0, 999.0, 1000.0, 2000.0] {
                let b = TemporalReRanker::temporal_boost(ts, hint, now);
                assert!(
                    (0.0..=1.0).contains(&b),
                    "hint={hint:?} ts={ts} boost={b} out of [0,1]"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // EntityTimeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_entity_timeline_track_and_history() {
        let mut tl = EntityTimeline::new();
        tl.track_entity_state("alice", 1.0, "status", "", "active");
        tl.track_entity_state("alice", 2.0, "role", "", "admin");
        tl.track_entity_state("alice", 3.0, "status", "active", "inactive");

        let hist = tl.get_entity_history("alice");
        assert_eq!(hist.len(), 3);
        assert_eq!(hist[0], (1.0, "status".into(), "".into(), "active".into()));
        assert_eq!(hist[1], (2.0, "role".into(), "".into(), "admin".into()));
        assert_eq!(
            hist[2],
            (3.0, "status".into(), "active".into(), "inactive".into())
        );
    }

    #[test]
    fn test_entity_timeline_history_empty() {
        let tl = EntityTimeline::new();
        assert!(tl.get_entity_history("nobody").is_empty());
    }

    #[test]
    fn test_entity_state_at_early() {
        let mut tl = EntityTimeline::new();
        tl.track_entity_state("bob", 1.0, "color", "", "red");
        tl.track_entity_state("bob", 3.0, "color", "red", "blue");
        tl.track_entity_state("bob", 5.0, "size", "", "large");

        // At ts=2: only first change applied
        let state = tl.get_entity_state_at("bob", 2.0);
        assert_eq!(state.get("color").map(String::as_str), Some("red"));
        assert!(!state.contains_key("size"));
    }

    #[test]
    fn test_entity_state_at_mid() {
        let mut tl = EntityTimeline::new();
        tl.track_entity_state("bob", 1.0, "color", "", "red");
        tl.track_entity_state("bob", 3.0, "color", "red", "blue");
        tl.track_entity_state("bob", 5.0, "size", "", "large");

        // At ts=4: color=blue, no size yet
        let state = tl.get_entity_state_at("bob", 4.0);
        assert_eq!(state.get("color").map(String::as_str), Some("blue"));
        assert!(!state.contains_key("size"));
    }

    #[test]
    fn test_entity_state_at_latest() {
        let mut tl = EntityTimeline::new();
        tl.track_entity_state("bob", 1.0, "color", "", "red");
        tl.track_entity_state("bob", 3.0, "color", "red", "blue");
        tl.track_entity_state("bob", 5.0, "size", "", "large");

        // At ts=10: all applied
        let state = tl.get_entity_state_at("bob", 10.0);
        assert_eq!(state.get("color").map(String::as_str), Some("blue"));
        assert_eq!(state.get("size").map(String::as_str), Some("large"));
    }

    #[test]
    fn test_entity_state_at_before_any_changes() {
        let mut tl = EntityTimeline::new();
        tl.track_entity_state("carol", 10.0, "x", "", "1");

        let state = tl.get_entity_state_at("carol", 5.0);
        assert!(state.is_empty());
    }

    #[test]
    fn test_entity_state_at_exact_boundary() {
        let mut tl = EntityTimeline::new();
        tl.track_entity_state("dave", 10.0, "a", "", "v1");
        tl.track_entity_state("dave", 20.0, "a", "v1", "v2");

        // Exactly at ts=10
        let state = tl.get_entity_state_at("dave", 10.0);
        assert_eq!(state.get("a").map(String::as_str), Some("v1"));

        // Exactly at ts=20
        let state2 = tl.get_entity_state_at("dave", 20.0);
        assert_eq!(state2.get("a").map(String::as_str), Some("v2"));
    }

    #[test]
    fn test_entity_timeline_multiple_entities_isolated() {
        let mut tl = EntityTimeline::new();
        tl.track_entity_state("x", 1.0, "k", "", "vx");
        tl.track_entity_state("y", 1.0, "k", "", "vy");

        let sx = tl.get_entity_state_at("x", 10.0);
        let sy = tl.get_entity_state_at("y", 10.0);
        assert_eq!(sx["k"], "vx");
        assert_eq!(sy["k"], "vy");
    }

    #[test]
    fn test_entity_timeline_out_of_order_insert() {
        let mut tl = EntityTimeline::new();
        // Insert in reverse order
        tl.track_entity_state("e", 30.0, "p", "b", "c");
        tl.track_entity_state("e", 10.0, "p", "", "a");
        tl.track_entity_state("e", 20.0, "p", "a", "b");

        let hist = tl.get_entity_history("e");
        // Should be sorted by timestamp
        assert_eq!(hist[0].0, 10.0);
        assert_eq!(hist[1].0, 20.0);
        assert_eq!(hist[2].0, 30.0);

        // State reconstruction should still work
        let state = tl.get_entity_state_at("e", 25.0);
        assert_eq!(state["p"], "b");
    }
}
