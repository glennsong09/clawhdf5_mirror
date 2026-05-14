//! Ephemeral working memory tier — pure in-memory, no disk persistence.
//!
//! All data lives only in the process heap. Entries expire via TTL,
//! are evicted by LFU/LRU when the store is at capacity, and can be
//! promoted to the persistent HDF5 tier by the caller.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an [`EphemeralStore`].
#[derive(Debug, Clone)]
pub struct EphemeralConfig {
    /// Maximum number of entries before eviction is triggered.
    pub max_entries: usize,
    /// Default TTL in seconds applied when the caller does not specify one.
    pub default_ttl_secs: f64,
    /// When `true`, `get` increments [`EphemeralEntry::access_count`].
    pub track_access: bool,
}

impl Default for EphemeralConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            default_ttl_secs: 3600.0,
            track_access: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry
// ─────────────────────────────────────────────────────────────────────────────

/// A single ephemeral memory record.
#[derive(Debug, Clone)]
pub struct EphemeralEntry {
    /// Raw value bytes.
    pub value: Vec<u8>,
    /// Optional decoded text (set by [`EphemeralStore::set_text`] /
    /// [`EphemeralStore::set_with_embedding`]).
    pub text: Option<String>,
    /// Optional embedding vector for semantic search.
    pub embedding: Option<Vec<f32>>,
    /// Unix-epoch seconds when this entry was created.
    pub created_at: f64,
    /// Unix-epoch seconds of the most recent successful get.
    pub last_accessed: f64,
    /// Unix-epoch seconds after which the entry is considered expired.
    /// `0.0` means "never expires".
    pub expires_at: f64,
    /// Number of times this entry has been successfully retrieved.
    pub access_count: u32,
    /// Free-form tags.
    pub tags: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of [`EphemeralStore`] health.
#[derive(Debug, Clone, Default)]
pub struct EphemeralStats {
    /// Number of live (non-expired) entries currently in the store.
    pub total_entries: usize,
    /// Total bytes occupied by all stored values.
    pub total_bytes: usize,
    /// Entries removed by TTL expiry in the most recent [`EphemeralStore::cleanup_expired`] call.
    pub expired_count: usize,
    /// Entries removed by capacity eviction since the store was created.
    pub evicted_count: usize,
    /// Age in seconds of the oldest entry (`0.0` when empty).
    pub oldest_entry_age_secs: f64,
    /// Cumulative successful get calls.
    pub hit_count: u64,
    /// Cumulative failed get calls (key absent or expired).
    pub miss_count: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory key-value store with TTL, capacity management, and embedding search.
pub struct EphemeralStore {
    config: EphemeralConfig,
    pub(crate) entries: HashMap<String, EphemeralEntry>,
    hit_count: u64,
    miss_count: u64,
    last_expired_count: usize,
    last_evicted_count: usize,
}

impl EphemeralStore {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new store with the given configuration.
    pub fn new(config: EphemeralConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            hit_count: 0,
            miss_count: 0,
            last_expired_count: 0,
            last_evicted_count: 0,
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Return current Unix epoch time as fractional seconds.
    fn now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    /// L2 norm of a vector.
    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Remove all expired entries without updating `last_expired_count`.
    /// Returns the number removed.  Accepts a pre-computed `now` to avoid
    /// repeated syscalls within a single logical operation.
    fn cleanup_expired_internal(&mut self, now: f64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, e| e.expires_at == 0.0 || now <= e.expires_at);
        before - self.entries.len()
    }

    /// Evict one entry when the store is at capacity.
    ///
    /// Strategy: lowest `access_count` first; ties broken by oldest
    /// `last_accessed` (ascending).
    fn evict_one(&mut self) {
        let key_to_evict = self
            .entries
            .iter()
            .min_by(|a, b| {
                a.1.access_count.cmp(&b.1.access_count).then_with(|| {
                    a.1.last_accessed
                        .partial_cmp(&b.1.last_accessed)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .map(|(k, _)| k.clone());

        if let Some(k) = key_to_evict {
            self.entries.remove(&k);
            self.last_evicted_count += 1;
        }
    }

    // ── Writes ───────────────────────────────────────────────────────────────

    /// Store raw bytes under `key`.
    ///
    /// If the store is at capacity, expired entries are removed first; if
    /// still full, the least-frequently-used entry is evicted.
    pub fn set(&mut self, key: &str, value: Vec<u8>, ttl_secs: Option<f64>) {
        let now = Self::now();

        // Only evict if we are inserting a brand-new key.
        if !self.entries.contains_key(key) && self.entries.len() >= self.config.max_entries {
            let removed = self.cleanup_expired_internal(now);
            if removed == 0 && self.entries.len() >= self.config.max_entries {
                self.evict_one();
            }
        }

        let ttl = ttl_secs.unwrap_or(self.config.default_ttl_secs);
        let expires_at = if ttl > 0.0 { now + ttl } else { 0.0 };

        let entry = EphemeralEntry {
            value,
            text: None,
            embedding: None,
            created_at: now,
            last_accessed: now,
            expires_at,
            access_count: 0,
            tags: Vec::new(),
        };
        self.entries.insert(key.to_string(), entry);
    }

    /// Store a UTF-8 string under `key`.
    ///
    /// The value bytes are the UTF-8 encoding of `text`; `entry.text` is also
    /// populated for fast text search.
    pub fn set_text(&mut self, key: &str, text: &str, ttl_secs: Option<f64>) {
        self.set(key, text.as_bytes().to_vec(), ttl_secs);
        if let Some(e) = self.entries.get_mut(key) {
            e.text = Some(text.to_string());
        }
    }

    /// Store a string together with a pre-computed embedding vector.
    pub fn set_with_embedding(
        &mut self,
        key: &str,
        text: &str,
        embedding: Vec<f32>,
        ttl_secs: Option<f64>,
    ) {
        self.set_text(key, text, ttl_secs);
        if let Some(e) = self.entries.get_mut(key) {
            e.embedding = Some(embedding);
        }
    }

    /// Store multiple entries in a single call.
    pub fn set_batch(&mut self, entries: Vec<(String, Vec<u8>, Option<f64>)>) {
        for (key, value, ttl) in entries {
            self.set(&key, value, ttl);
        }
    }

    // ── Reads ─────────────────────────────────────────────────────────────────

    /// Retrieve raw bytes for `key`.
    ///
    /// Returns `None` if the key is absent or the entry has expired (the
    /// expired entry is removed).  Updates access statistics on hit.
    pub fn get(&mut self, key: &str) -> Option<&[u8]> {
        let now = Self::now();
        let is_expired = self
            .entries
            .get(key)
            .map(|e| e.expires_at > 0.0 && now > e.expires_at);

        match is_expired {
            None => {
                self.miss_count += 1;
                return None;
            }
            Some(true) => {
                self.entries.remove(key);
                self.miss_count += 1;
                return None;
            }
            Some(false) => {}
        }

        self.hit_count += 1;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = now;
            if self.config.track_access {
                entry.access_count += 1;
            }
        }
        self.entries.get(key).map(|e| e.value.as_slice())
    }

    /// Retrieve the text field for `key`.
    ///
    /// Returns `None` if the key is absent, expired, or the entry has no text.
    /// Updates access statistics on hit.
    pub fn get_text(&mut self, key: &str) -> Option<&str> {
        let now = Self::now();
        let is_expired = self
            .entries
            .get(key)
            .map(|e| e.expires_at > 0.0 && now > e.expires_at);

        match is_expired {
            None => {
                self.miss_count += 1;
                return None;
            }
            Some(true) => {
                self.entries.remove(key);
                self.miss_count += 1;
                return None;
            }
            Some(false) => {}
        }

        // Only count as a hit if the entry has a text field.
        if self
            .entries
            .get(key)
            .and_then(|e| e.text.as_ref())
            .is_none()
        {
            self.miss_count += 1;
            return None;
        }

        self.hit_count += 1;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = now;
            if self.config.track_access {
                entry.access_count += 1;
            }
        }
        self.entries.get(key).and_then(|e| e.text.as_deref())
    }

    /// Return a reference to the raw [`EphemeralEntry`] for `key`, if present
    /// and not expired.
    ///
    /// Does **not** update access statistics.
    pub fn get_entry(&self, key: &str) -> Option<&EphemeralEntry> {
        let entry = self.entries.get(key)?;
        let now = Self::now();
        if entry.expires_at > 0.0 && now > entry.expires_at {
            return None;
        }
        Some(entry)
    }

    /// Retrieve multiple keys at once.
    ///
    /// Each tuple in the result is `(key, Some(bytes))` on hit or
    /// `(key, None)` on miss / expiry.
    pub fn get_batch(&mut self, keys: &[&str]) -> Vec<(String, Option<Vec<u8>>)> {
        keys.iter()
            .map(|&k| {
                let val = self.get(k).map(|b| b.to_vec());
                (k.to_string(), val)
            })
            .collect()
    }

    // ── Existence & deletion ─────────────────────────────────────────────────

    /// Returns `true` if the key exists and is not expired.
    ///
    /// Does **not** update access statistics.
    pub fn exists(&self, key: &str) -> bool {
        match self.entries.get(key) {
            None => false,
            Some(e) => {
                if e.expires_at == 0.0 {
                    return true;
                }
                let now = Self::now();
                now <= e.expires_at
            }
        }
    }

    /// Remove the entry for `key`.  Returns `true` if the key existed.
    pub fn delete(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    // ── TTL management ───────────────────────────────────────────────────────

    /// Update the TTL for an existing key.
    ///
    /// Returns `true` if the key was found and updated, `false` otherwise.
    /// A `ttl_secs` of `0.0` sets the entry to never expire.
    pub fn set_ttl(&mut self, key: &str, ttl_secs: f64) -> bool {
        let now = Self::now();
        match self.entries.get_mut(key) {
            None => false,
            Some(entry) => {
                entry.expires_at = if ttl_secs > 0.0 { now + ttl_secs } else { 0.0 };
                true
            }
        }
    }

    /// Return the remaining TTL in seconds for `key`.
    ///
    /// Returns `None` if the key does not exist or is already expired.
    /// Returns `Some(f64::INFINITY)` if the entry never expires.
    pub fn ttl(&self, key: &str) -> Option<f64> {
        let entry = self.entries.get(key)?;
        if entry.expires_at == 0.0 {
            return Some(f64::INFINITY);
        }
        let now = Self::now();
        let remaining = entry.expires_at - now;
        if remaining <= 0.0 {
            None
        } else {
            Some(remaining)
        }
    }

    // ── Maintenance ──────────────────────────────────────────────────────────

    /// Remove all expired entries.  Returns the number removed.
    pub fn cleanup_expired(&mut self) -> usize {
        let now = Self::now();
        let removed = self.cleanup_expired_internal(now);
        self.last_expired_count = removed;
        removed
    }

    // ── Search ───────────────────────────────────────────────────────────────

    /// Simple token-match search over all text fields.
    ///
    /// Splits `query` on whitespace; a document scores 1 point per matching
    /// token (case-insensitive).  Returns up to `k` results sorted by score
    /// descending.  Expired entries are lazily removed during iteration.
    pub fn search_text(&mut self, query: &str, k: usize) -> Vec<(String, f32)> {
        let tokens: Vec<String> = query.split_whitespace().map(|t| t.to_lowercase()).collect();

        if tokens.is_empty() || k == 0 {
            return Vec::new();
        }

        let now = Self::now();

        // Collect (key, score) without mutating the map while iterating.
        let mut expired_keys: Vec<String> = Vec::new();
        let mut scored: Vec<(String, f32)> = Vec::new();

        for (key, entry) in &self.entries {
            // Skip expired.
            if entry.expires_at > 0.0 && now > entry.expires_at {
                expired_keys.push(key.clone());
                continue;
            }
            let text = match &entry.text {
                Some(t) => t.to_lowercase(),
                None => continue,
            };
            let score: f32 = tokens.iter().filter(|t| text.contains(t.as_str())).count() as f32;
            if score > 0.0 {
                scored.push((key.clone(), score));
            }
        }

        // Lazily evict expired entries found during iteration.
        for k in expired_keys {
            self.entries.remove(&k);
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Cosine-similarity search over entries that have an embedding vector.
    ///
    /// Uses `clawhdf5_accel::dot_product` for the inner product.  Returns up
    /// to `k` results sorted by cosine similarity descending.  Does **not**
    /// update access statistics.
    pub fn search_embedding(&self, query_embedding: &[f32], k: usize) -> Vec<(String, f32)> {
        if query_embedding.is_empty() || k == 0 {
            return Vec::new();
        }

        let query_norm = Self::l2_norm(query_embedding);
        if query_norm == 0.0 {
            return Vec::new();
        }

        let now = Self::now();
        let mut scored: Vec<(String, f32)> = self
            .entries
            .iter()
            .filter(|(_, e)| {
                // Skip expired.
                if e.expires_at > 0.0 && now > e.expires_at {
                    return false;
                }
                e.embedding.is_some()
            })
            .filter_map(|(key, entry)| {
                let emb = entry.embedding.as_ref()?;
                let emb_norm = Self::l2_norm(emb);
                if emb_norm == 0.0 {
                    return None;
                }
                let dot = clawhdf5_accel::dot_product(query_embedding, emb);
                let cosine = dot / (query_norm * emb_norm);
                Some((key.clone(), cosine))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    // ── Promotion helpers ────────────────────────────────────────────────────

    /// Return the keys of all non-expired entries whose access count is at
    /// least `min_access_count`.
    pub fn promotion_candidates(&self, min_access_count: u32) -> Vec<String> {
        let now = Self::now();
        self.entries
            .iter()
            .filter(|(_, e)| {
                let alive = e.expires_at == 0.0 || now <= e.expires_at;
                alive && e.access_count >= min_access_count
            })
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Remove and return the entry for `key` so the caller can promote it to
    /// the persistent tier.
    pub fn take_for_promotion(&mut self, key: &str) -> Option<EphemeralEntry> {
        self.entries.remove(key)
    }

    // ── Introspection ────────────────────────────────────────────────────────

    /// Return a snapshot of store statistics.
    pub fn stats(&self) -> EphemeralStats {
        let now = Self::now();
        let live_entries: Vec<&EphemeralEntry> = self
            .entries
            .values()
            .filter(|e| e.expires_at == 0.0 || now <= e.expires_at)
            .collect();

        let total_bytes = live_entries.iter().map(|e| e.value.len()).sum();

        let oldest_entry_age_secs = live_entries
            .iter()
            .map(|e| now - e.created_at)
            .fold(0.0_f64, f64::max);

        EphemeralStats {
            total_entries: live_entries.len(),
            total_bytes,
            expired_count: self.last_expired_count,
            evicted_count: self.last_evicted_count,
            oldest_entry_age_secs,
            hit_count: self.hit_count,
            miss_count: self.miss_count,
        }
    }

    /// Return the keys of all live (non-expired) entries.
    pub fn keys(&self) -> Vec<String> {
        let now = Self::now();
        self.entries
            .iter()
            .filter(|(_, e)| e.expires_at == 0.0 || now <= e.expires_at)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Number of entries currently in the store (including not-yet-evicted
    /// expired ones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when there are no entries in the store.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove **all** entries and return the count that was removed.
    pub fn clear(&mut self) -> usize {
        let count = self.entries.len();
        self.entries.clear();
        count
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> EphemeralStore {
        EphemeralStore::new(EphemeralConfig::default())
    }

    fn small_store(max: usize) -> EphemeralStore {
        EphemeralStore::new(EphemeralConfig {
            max_entries: max,
            default_ttl_secs: 3600.0,
            track_access: true,
        })
    }

    // ── set / get round-trip ─────────────────────────────────────────────────

    #[test]
    fn set_and_get_bytes() {
        let mut s = store();
        s.set("k1", b"hello".to_vec(), None);
        assert_eq!(s.get("k1"), Some(b"hello".as_ref()));
    }

    #[test]
    fn get_missing_key_returns_none() {
        let mut s = store();
        assert_eq!(s.get("nope"), None);
    }

    #[test]
    fn set_text_roundtrip() {
        let mut s = store();
        s.set_text("t1", "world", None);
        assert_eq!(s.get_text("t1"), Some("world"));
    }

    #[test]
    fn get_text_missing_returns_none() {
        let mut s = store();
        assert_eq!(s.get_text("gone"), None);
    }

    #[test]
    fn set_text_also_stores_bytes() {
        let mut s = store();
        s.set_text("tb", "abc", None);
        assert_eq!(s.get("tb"), Some(b"abc".as_ref()));
    }

    // ── TTL expiry ───────────────────────────────────────────────────────────

    #[test]
    fn get_returns_none_after_ttl() {
        let mut s = store();
        // Set an entry that expired 1 second ago.
        let past = EphemeralEntry {
            value: b"old".to_vec(),
            text: Some("old".into()),
            embedding: None,
            created_at: 0.0,
            last_accessed: 0.0,
            expires_at: 1.0, // Jan 1 1970 — definitely expired
            access_count: 0,
            tags: Vec::new(),
        };
        s.entries.insert("exp".into(), past);
        assert_eq!(s.get("exp"), None);
        // Entry must have been removed.
        assert!(!s.entries.contains_key("exp"));
    }

    #[test]
    fn get_text_returns_none_after_ttl() {
        let mut s = store();
        let past = EphemeralEntry {
            value: b"stale".to_vec(),
            text: Some("stale".into()),
            embedding: None,
            created_at: 0.0,
            last_accessed: 0.0,
            expires_at: 1.0,
            access_count: 0,
            tags: Vec::new(),
        };
        s.entries.insert("stale".into(), past);
        assert_eq!(s.get_text("stale"), None);
    }

    #[test]
    fn exists_returns_false_for_expired() {
        let mut s = store();
        let past = EphemeralEntry {
            value: vec![],
            text: None,
            embedding: None,
            created_at: 0.0,
            last_accessed: 0.0,
            expires_at: 1.0,
            access_count: 0,
            tags: Vec::new(),
        };
        s.entries.insert("x".into(), past);
        assert!(!s.exists("x"));
    }

    #[test]
    fn ttl_returns_none_for_expired() {
        let s = EphemeralStore::new(EphemeralConfig::default());
        // Not in map at all.
        assert!(s.ttl("absent").is_none());
    }

    #[test]
    fn ttl_returns_infinity_for_no_expire() {
        let mut s = store();
        s.set("forever", b"x".to_vec(), Some(0.0));
        assert_eq!(s.ttl("forever"), Some(f64::INFINITY));
    }

    // ── delete ───────────────────────────────────────────────────────────────

    #[test]
    fn delete_existing_key() {
        let mut s = store();
        s.set("del", b"v".to_vec(), None);
        assert!(s.delete("del"));
        assert_eq!(s.get("del"), None);
    }

    #[test]
    fn delete_missing_key_returns_false() {
        let mut s = store();
        assert!(!s.delete("nada"));
    }

    // ── access tracking ──────────────────────────────────────────────────────

    #[test]
    fn access_count_increments_on_get() {
        let mut s = store();
        s.set_text("ac", "value", None);
        s.get("ac");
        s.get("ac");
        s.get("ac");
        assert_eq!(s.entries["ac"].access_count, 3);
    }

    #[test]
    fn miss_count_increments_on_absent_key() {
        let mut s = store();
        s.get("ghost");
        s.get("ghost");
        assert_eq!(s.stats().miss_count, 2);
    }

    #[test]
    fn hit_count_increments_on_successful_get() {
        let mut s = store();
        s.set("hit", b"y".to_vec(), None);
        s.get("hit");
        s.get("hit");
        assert_eq!(s.stats().hit_count, 2);
    }

    // ── capacity & eviction ──────────────────────────────────────────────────

    #[test]
    fn eviction_respects_max_entries() {
        let mut s = small_store(3);
        s.set("a", b"1".to_vec(), None);
        s.set("b", b"2".to_vec(), None);
        s.set("c", b"3".to_vec(), None);
        // Inserting a 4th entry must evict one.
        s.set("d", b"4".to_vec(), None);
        assert_eq!(s.len(), 3, "store must not exceed max_entries");
    }

    #[test]
    fn eviction_prefers_expired_entries() {
        let mut s = small_store(2);
        // Add entry that expires far in the future.
        s.set("keep", b"k".to_vec(), Some(9999.0));
        // Add an already-expired entry directly.
        let dead = EphemeralEntry {
            value: b"dead".to_vec(),
            text: None,
            embedding: None,
            created_at: 0.0,
            last_accessed: 0.0,
            expires_at: 1.0, // expired
            access_count: 0,
            tags: Vec::new(),
        };
        s.entries.insert("dead".into(), dead);
        // Inserting a third entry: should evict the expired one.
        s.set("new", b"n".to_vec(), Some(9999.0));
        assert!(s.entries.contains_key("keep"), "keep should survive");
        assert!(s.entries.contains_key("new"), "new should be inserted");
        assert!(!s.entries.contains_key("dead"), "dead should be evicted");
    }

    // ── batch operations ─────────────────────────────────────────────────────

    #[test]
    fn set_batch_inserts_all() {
        let mut s = store();
        s.set_batch(vec![
            ("k1".into(), b"v1".to_vec(), None),
            ("k2".into(), b"v2".to_vec(), None),
            ("k3".into(), b"v3".to_vec(), None),
        ]);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn get_batch_returns_hits_and_misses() {
        let mut s = store();
        s.set("a", b"alpha".to_vec(), None);
        s.set("b", b"beta".to_vec(), None);
        let results = s.get_batch(&["a", "b", "c"]);
        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|(k, v)| k == "a" && v.is_some()));
        assert!(results.iter().any(|(k, v)| k == "c" && v.is_none()));
    }

    // ── cleanup_expired ──────────────────────────────────────────────────────

    #[test]
    fn cleanup_expired_removes_dead_entries() {
        let mut s = store();
        s.set("live", b"l".to_vec(), Some(9999.0));
        let dead = EphemeralEntry {
            value: b"d".to_vec(),
            text: None,
            embedding: None,
            created_at: 0.0,
            last_accessed: 0.0,
            expires_at: 1.0,
            access_count: 0,
            tags: Vec::new(),
        };
        s.entries.insert("dead".into(), dead.clone());
        s.entries.insert("dead2".into(), dead);
        let removed = s.cleanup_expired();
        assert_eq!(removed, 2);
        assert!(s.entries.contains_key("live"));
        assert_eq!(s.stats().expired_count, 2);
    }

    // ── text search ──────────────────────────────────────────────────────────

    #[test]
    fn search_text_finds_matching_entries() {
        let mut s = store();
        s.set_text("doc1", "the quick brown fox", None);
        s.set_text("doc2", "a lazy dog", None);
        s.set_text("doc3", "quick fox jumps", None);

        let results = s.search_text("quick fox", 10);
        let keys: Vec<&str> = results.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"doc1"), "doc1 should match 'quick fox'");
        assert!(keys.contains(&"doc3"), "doc3 should match 'quick fox'");
        assert!(!keys.contains(&"doc2"), "doc2 should not match 'quick fox'");
    }

    #[test]
    fn search_text_respects_k_limit() {
        let mut s = store();
        for i in 0..10 {
            s.set_text(&format!("d{i}"), "match term", None);
        }
        let results = s.search_text("match", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_text_empty_query_returns_empty() {
        let mut s = store();
        s.set_text("x", "hello", None);
        let results = s.search_text("", 10);
        assert!(results.is_empty());
    }

    // ── embedding search ─────────────────────────────────────────────────────

    #[test]
    fn search_embedding_ranks_by_cosine() {
        let mut s = store();
        // Embedding identical to query → cosine = 1.0
        s.set_with_embedding("perfect", "p", vec![1.0, 0.0, 0.0, 0.0], None);
        // Orthogonal → cosine = 0.0
        s.set_with_embedding("ortho", "o", vec![0.0, 1.0, 0.0, 0.0], None);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = s.search_embedding(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "perfect");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn search_embedding_empty_query_returns_empty() {
        let mut s = store();
        s.set_with_embedding("x", "v", vec![1.0, 0.0], None);
        assert!(s.search_embedding(&[], 5).is_empty());
    }

    // ── promotion ────────────────────────────────────────────────────────────

    #[test]
    fn promotion_candidates_filters_by_access_count() {
        let mut s = store();
        s.set_text("frequent", "f", None);
        // Manually bump access count.
        s.entries.get_mut("frequent").unwrap().access_count = 5;
        s.set_text("rare", "r", None);
        s.entries.get_mut("rare").unwrap().access_count = 1;

        let candidates = s.promotion_candidates(3);
        assert!(candidates.contains(&"frequent".to_string()));
        assert!(!candidates.contains(&"rare".to_string()));
    }

    #[test]
    fn take_for_promotion_removes_entry() {
        let mut s = store();
        s.set_text("promo", "data", None);
        let entry = s.take_for_promotion("promo");
        assert!(entry.is_some());
        assert!(!s.entries.contains_key("promo"));
    }

    // ── set_ttl / ttl ────────────────────────────────────────────────────────

    #[test]
    fn set_ttl_updates_expiry() {
        let mut s = store();
        s.set("key", b"v".to_vec(), Some(100.0));
        let original = s.entries["key"].expires_at;
        // Re-set a shorter TTL.
        assert!(s.set_ttl("key", 10.0));
        let updated = s.entries["key"].expires_at;
        // The updated expiry should be less than the original (shorter TTL).
        assert!(updated < original);
    }

    #[test]
    fn set_ttl_missing_key_returns_false() {
        let mut s = store();
        assert!(!s.set_ttl("ghost", 60.0));
    }

    // ── stats ────────────────────────────────────────────────────────────────

    #[test]
    fn stats_reflects_current_state() {
        let mut s = store();
        s.set("a", b"hello".to_vec(), None);
        s.set("b", b"world!".to_vec(), None);
        let st = s.stats();
        assert_eq!(st.total_entries, 2);
        assert_eq!(st.total_bytes, 11); // 5 + 6
    }

    // ── clear ────────────────────────────────────────────────────────────────

    #[test]
    fn clear_empties_store() {
        let mut s = store();
        s.set("x", b"1".to_vec(), None);
        s.set("y", b"2".to_vec(), None);
        let removed = s.clear();
        assert_eq!(removed, 2);
        assert!(s.is_empty());
    }

    // ── keys / len / is_empty ─────────────────────────────────────────────────

    #[test]
    fn keys_returns_live_keys() {
        let mut s = store();
        s.set("live1", b"a".to_vec(), Some(9999.0));
        s.set("live2", b"b".to_vec(), Some(9999.0));
        // Add expired entry.
        let dead = EphemeralEntry {
            value: vec![],
            text: None,
            embedding: None,
            created_at: 0.0,
            last_accessed: 0.0,
            expires_at: 1.0,
            access_count: 0,
            tags: Vec::new(),
        };
        s.entries.insert("dead".into(), dead);

        let ks = s.keys();
        assert!(ks.contains(&"live1".to_string()));
        assert!(ks.contains(&"live2".to_string()));
        assert!(!ks.contains(&"dead".to_string()));
    }

    #[test]
    fn is_empty_on_fresh_store() {
        let s = store();
        assert!(s.is_empty());
    }

    #[test]
    fn len_reflects_all_entries_including_expired() {
        let mut s = store();
        s.set("a", b"1".to_vec(), None);
        let dead = EphemeralEntry {
            value: vec![],
            text: None,
            embedding: None,
            created_at: 0.0,
            last_accessed: 0.0,
            expires_at: 1.0,
            access_count: 0,
            tags: Vec::new(),
        };
        s.entries.insert("dead".into(), dead);
        assert_eq!(s.len(), 2);
    }

    // ── with_embedding stores correctly ──────────────────────────────────────

    #[test]
    fn set_with_embedding_stores_embedding() {
        let mut s = store();
        s.set_with_embedding("emb", "text here", vec![1.0, 2.0, 3.0], None);
        let entry = &s.entries["emb"];
        assert_eq!(entry.embedding, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(entry.text.as_deref(), Some("text here"));
    }
}
