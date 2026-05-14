//! In-memory cache for memory entries, sessions, and knowledge graph.

use crate::vector_search;

/// In-memory cache for the /memory group data.
#[derive(Debug, Clone)]
pub struct MemoryCache {
    pub chunks: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
    pub source_channels: Vec<String>,
    pub timestamps: Vec<f64>,
    pub session_ids: Vec<String>,
    pub tags: Vec<String>,
    pub tombstones: Vec<u8>,
    pub embedding_dim: usize,
    /// Pre-computed L2 norms for each embedding.
    pub norms: Vec<f32>,
    /// Hebbian activation weights (default 1.0 per entry).
    pub activation_weights: Vec<f32>,
}

impl MemoryCache {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            chunks: Vec::new(),
            embeddings: Vec::new(),
            source_channels: Vec::new(),
            timestamps: Vec::new(),
            session_ids: Vec::new(),
            tags: Vec::new(),
            tombstones: Vec::new(),
            embedding_dim,
            norms: Vec::new(),
            activation_weights: Vec::new(),
        }
    }

    /// Total number of entries (including tombstoned).
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Number of active (non-tombstoned) entries.
    pub fn count_active(&self) -> usize {
        self.tombstones.iter().filter(|&&t| t == 0).count()
    }

    /// Push a new entry, returns its index.
    pub fn push(
        &mut self,
        chunk: String,
        embedding: Vec<f32>,
        source_channel: String,
        timestamp: f64,
        session_id: String,
        tags: String,
    ) -> usize {
        let idx = self.chunks.len();
        let norm = vector_search::compute_norm(&embedding);
        self.chunks.push(chunk);
        self.embeddings.push(embedding);
        self.source_channels.push(source_channel);
        self.timestamps.push(timestamp);
        self.session_ids.push(session_id);
        self.tags.push(tags);
        self.tombstones.push(0);
        self.norms.push(norm);
        self.activation_weights.push(1.0);
        idx
    }

    /// Find an active (non-tombstoned) entry by tags (used as key for dedup).
    /// Returns the index of the first matching active entry, or None.
    pub fn find_by_tags(&self, tags: &str) -> Option<usize> {
        if tags.is_empty() {
            return None;
        }
        for (i, t) in self.tags.iter().enumerate() {
            if self.tombstones.get(i).copied().unwrap_or(1) == 0 && t == tags {
                return Some(i);
            }
        }
        None
    }

    /// Update an existing entry in-place (for upsert dedup).
    pub fn update(
        &mut self,
        idx: usize,
        chunk: String,
        embedding: Vec<f32>,
        source_channel: String,
        timestamp: f64,
        session_id: String,
    ) {
        if idx < self.chunks.len() {
            let norm = vector_search::compute_norm(&embedding);
            self.chunks[idx] = chunk;
            self.embeddings[idx] = embedding;
            self.source_channels[idx] = source_channel;
            self.timestamps[idx] = timestamp;
            self.session_ids[idx] = session_id;
            self.norms[idx] = norm;
            self.activation_weights[idx] = 1.0; // reset activation on update
        }
    }

    /// Mark an entry as deleted (tombstoned).
    pub fn mark_deleted(&mut self, id: usize) -> bool {
        if id < self.tombstones.len() && self.tombstones[id] == 0 {
            self.tombstones[id] = 1;
            true
        } else {
            false
        }
    }

    /// Fraction of entries that are tombstoned.
    pub fn tombstone_fraction(&self) -> f32 {
        if self.chunks.is_empty() {
            return 0.0;
        }
        let tombstoned = self.tombstones.iter().filter(|&&t| t == 1).count();
        tombstoned as f32 / self.chunks.len() as f32
    }

    /// Remove all tombstoned entries, returns number removed.
    /// Also returns a mapping from old indices to new indices (None if removed).
    /// Recomputes norms for remaining entries.
    pub fn compact(&mut self) -> (usize, Vec<Option<usize>>) {
        let old_len = self.chunks.len();
        let mut index_map = vec![None; old_len];
        let mut new_idx = 0usize;

        let mut new_chunks = Vec::new();
        let mut new_embeddings = Vec::new();
        let mut new_source_channels = Vec::new();
        let mut new_timestamps = Vec::new();
        let mut new_session_ids = Vec::new();
        let mut new_tags = Vec::new();
        let mut new_tombstones = Vec::new();
        let mut new_norms = Vec::new();
        let mut new_activation_weights = Vec::new();

        for (i, slot) in index_map.iter_mut().enumerate() {
            if self.tombstones[i] == 0 {
                *slot = Some(new_idx);
                new_idx += 1;
                let norm = vector_search::compute_norm(&self.embeddings[i]);
                new_chunks.push(self.chunks[i].clone());
                new_embeddings.push(self.embeddings[i].clone());
                new_source_channels.push(self.source_channels[i].clone());
                new_timestamps.push(self.timestamps[i]);
                new_session_ids.push(self.session_ids[i].clone());
                new_tags.push(self.tags[i].clone());
                new_tombstones.push(0u8);
                new_norms.push(norm);
                new_activation_weights.push(self.activation_weights[i]);
            }
        }

        let removed = old_len - new_chunks.len();
        self.chunks = new_chunks;
        self.embeddings = new_embeddings;
        self.source_channels = new_source_channels;
        self.timestamps = new_timestamps;
        self.session_ids = new_session_ids;
        self.tags = new_tags;
        self.tombstones = new_tombstones;
        self.norms = new_norms;
        self.activation_weights = new_activation_weights;

        (removed, index_map)
    }

    /// Flatten all embeddings into a single Vec<f32> for HDF5 storage.
    pub fn flat_embeddings(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.embeddings.len() * self.embedding_dim);
        for emb in &self.embeddings {
            flat.extend_from_slice(emb);
        }
        flat
    }
}
