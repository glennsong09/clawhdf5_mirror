//! Multi-Modal Memory — Track 6
//!
//! Supports storing and searching memories across multiple modalities:
//! Text, Image, Audio, Video, and Structured data.
//!
//! Each record can carry multiple embeddings (e.g. a CLIP image embedding
//! alongside a text embedding for the same document), enabling both
//! within-modality and cross-modal retrieval.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Modality
// ---------------------------------------------------------------------------

/// The sensory / semantic modality of a memory record.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Structured,
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Modality::Text => write!(f, "text"),
            Modality::Image => write!(f, "image"),
            Modality::Audio => write!(f, "audio"),
            Modality::Video => write!(f, "video"),
            Modality::Structured => write!(f, "structured"),
        }
    }
}

// ---------------------------------------------------------------------------
// ModalEmbedding
// ---------------------------------------------------------------------------

/// A dense float embedding produced by a specific model for a specific modality.
#[derive(Debug, Clone)]
pub struct ModalEmbedding {
    pub modality: Modality,
    /// Raw embedding values.
    pub embedding: Vec<f32>,
    /// Expected dimensionality (must equal `embedding.len()`).
    pub dimension: usize,
    /// Identifier of the model that produced this embedding,
    /// e.g. `"clip-vit-large"`, `"whisper-base"`, `"text-embedding-3-small"`.
    pub model_id: String,
}

impl ModalEmbedding {
    /// Create a new embedding, setting `dimension` from the vector length.
    pub fn new(modality: Modality, embedding: Vec<f32>, model_id: impl Into<String>) -> Self {
        let dimension = embedding.len();
        Self {
            modality,
            embedding,
            dimension,
            model_id: model_id.into(),
        }
    }

    /// L2 norm of the embedding.
    #[inline]
    pub fn norm(&self) -> f32 {
        self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// MediaRefType / MediaRef
// ---------------------------------------------------------------------------

/// Where the raw media bytes live.
#[derive(Debug, Clone)]
pub enum MediaRefType {
    /// A path on the local filesystem.
    Path(String),
    /// A remote URL.
    Url(String),
    /// Bytes stored inline.
    Inline(Vec<u8>),
}

/// A reference to the raw media associated with a record.
#[derive(Debug, Clone)]
pub struct MediaRef {
    pub ref_type: MediaRefType,
    pub mime_type: String,
    pub size_bytes: Option<u64>,
    /// FNV-1a 64-bit hash of the content (for Inline) or of the path/URL string.
    pub checksum: Option<u64>,
}

impl MediaRef {
    /// Construct a `Path` reference, computing a checksum of the path string.
    pub fn path(path: impl Into<String>, mime_type: impl Into<String>) -> Self {
        let p = path.into();
        let cs = fnv1a_64(p.as_bytes());
        Self {
            ref_type: MediaRefType::Path(p),
            mime_type: mime_type.into(),
            size_bytes: None,
            checksum: Some(cs),
        }
    }

    /// Construct a `Url` reference, computing a checksum of the URL string.
    pub fn url(url: impl Into<String>, mime_type: impl Into<String>) -> Self {
        let u = url.into();
        let cs = fnv1a_64(u.as_bytes());
        Self {
            ref_type: MediaRefType::Url(u),
            mime_type: mime_type.into(),
            size_bytes: None,
            checksum: Some(cs),
        }
    }

    /// Construct an `Inline` reference, computing a checksum of the bytes.
    pub fn inline(data: Vec<u8>, mime_type: impl Into<String>) -> Self {
        let cs = fnv1a_64(&data);
        let sz = data.len() as u64;
        Self {
            ref_type: MediaRefType::Inline(data),
            mime_type: mime_type.into(),
            size_bytes: Some(sz),
            checksum: Some(cs),
        }
    }
}

// ---------------------------------------------------------------------------
// FNV-1a helper (no external deps)
// ---------------------------------------------------------------------------

/// 64-bit FNV-1a hash.
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut h = OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// What an agent perceived versus what it concluded from a modality.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Literal description of what was perceived (e.g. "I see a red stop-sign").
    pub raw_perception: String,
    /// Higher-level interpretation (e.g. "the vehicle must stop").
    pub interpretation: String,
    /// Confidence in the interpretation, clamped to [0.0, 1.0].
    pub confidence: f32,
    pub modality: Modality,
}

impl Observation {
    pub fn new(
        raw_perception: impl Into<String>,
        interpretation: impl Into<String>,
        confidence: f32,
        modality: Modality,
    ) -> Self {
        Self {
            raw_perception: raw_perception.into(),
            interpretation: interpretation.into(),
            confidence: confidence.clamp(0.0, 1.0),
            modality,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiModalRecord
// ---------------------------------------------------------------------------

/// A single memory record that may span multiple modalities.
///
/// A record can hold embeddings from several models/modalities simultaneously,
/// enabling cross-modal nearest-neighbour queries.
#[derive(Debug, Clone)]
pub struct MultiModalRecord {
    pub id: u64,
    pub primary_modality: Modality,
    /// Textual content (caption, transcript, document text, …).
    pub text_content: Option<String>,
    /// Reference to the raw media artifact.
    pub media_ref: Option<MediaRef>,
    /// One or more embeddings, potentially from different models/modalities.
    pub embeddings: Vec<ModalEmbedding>,
    /// Optional agent observation attached to this record.
    pub observation: Option<Observation>,
    /// Unix timestamp (seconds, float for sub-second precision).
    pub timestamp: f64,
    pub metadata: HashMap<String, String>,
}

impl MultiModalRecord {
    /// Iterate over embeddings that belong to a particular modality.
    pub fn embeddings_for(&self, modality: &Modality) -> impl Iterator<Item = &ModalEmbedding> {
        self.embeddings
            .iter()
            .filter(move |e| &e.modality == modality)
    }
}

// ---------------------------------------------------------------------------
// MultiModalStore
// ---------------------------------------------------------------------------

/// In-memory store for multi-modal memory records with cosine search.
pub struct MultiModalStore {
    records: Vec<MultiModalRecord>,
    next_id: u64,
}

impl MultiModalStore {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            next_id: 1,
        }
    }

    // ------------------------------------------------------------------
    // Mutations
    // ------------------------------------------------------------------

    /// Add a record to the store.  The `id` field on the record is ignored
    /// and replaced with the store's auto-incremented counter.
    pub fn add_record(&mut self, mut record: MultiModalRecord) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        record.id = id;
        self.records.push(record);
        id
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Retrieve a record by id.
    pub fn get_record(&self, id: u64) -> Option<&MultiModalRecord> {
        self.records.iter().find(|r| r.id == id)
    }

    /// All records whose primary modality matches.
    pub fn get_by_modality(&self, modality: &Modality) -> Vec<&MultiModalRecord> {
        self.records
            .iter()
            .filter(|r| &r.primary_modality == modality)
            .collect()
    }

    /// All `Observation`s attached to records of a specific modality.
    pub fn get_observations(&self, modality: &Modality) -> Vec<&Observation> {
        self.records
            .iter()
            .filter_map(|r| r.observation.as_ref().filter(|o| &o.modality == modality))
            .collect()
    }

    /// Total number of records.
    pub fn count(&self) -> usize {
        self.records.len()
    }

    /// Number of records whose primary modality matches.
    pub fn count_by_modality(&self, modality: &Modality) -> usize {
        self.records
            .iter()
            .filter(|r| &r.primary_modality == modality)
            .count()
    }

    // ------------------------------------------------------------------
    // Vector search
    // ------------------------------------------------------------------

    /// Cosine nearest-neighbour search restricted to embeddings of `modality`.
    ///
    /// For each record, the highest cosine similarity across all embeddings
    /// that match `modality` is used as the record's score.
    ///
    /// Returns up to `k` `(record_id, similarity)` pairs sorted descending.
    pub fn search_by_modality(
        &self,
        modality: &Modality,
        query: &[f32],
        k: usize,
    ) -> Vec<(u64, f32)> {
        let q_norm = l2_norm(query);
        let mut scored: Vec<(u64, f32)> = self
            .records
            .iter()
            .filter_map(|r| {
                let best = r
                    .embeddings_for(modality)
                    .map(|e| cosine_sim_prenorm(query, q_norm, &e.embedding))
                    .fold(f32::NEG_INFINITY, f32::max);
                if best.is_finite() {
                    Some((r.id, best))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Cross-modal cosine search — considers all embeddings regardless of modality.
    ///
    /// Returns up to `k` `(record_id, similarity)` pairs sorted descending.
    pub fn search_cross_modal(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let q_norm = l2_norm(query);
        let mut scored: Vec<(u64, f32)> = self
            .records
            .iter()
            .filter_map(|r| {
                let best = r
                    .embeddings
                    .iter()
                    .map(|e| cosine_sim_prenorm(query, q_norm, &e.embedding))
                    .fold(f32::NEG_INFINITY, f32::max);
                if best.is_finite() {
                    Some((r.id, best))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

impl Default for MultiModalStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[inline]
fn cosine_sim_prenorm(query: &[f32], q_norm: f32, candidate: &[f32]) -> f32 {
    let c_norm = l2_norm(candidate);
    let denom = q_norm * c_norm;
    if denom == 0.0 {
        return 0.0;
    }
    let dot: f32 = query.iter().zip(candidate.iter()).map(|(a, b)| a * b).sum();
    dot / denom
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Modality
    // -----------------------------------------------------------------------

    #[test]
    fn modality_display() {
        assert_eq!(Modality::Text.to_string(), "text");
        assert_eq!(Modality::Image.to_string(), "image");
        assert_eq!(Modality::Audio.to_string(), "audio");
        assert_eq!(Modality::Video.to_string(), "video");
        assert_eq!(Modality::Structured.to_string(), "structured");
    }

    #[test]
    fn modality_equality() {
        assert_eq!(Modality::Text, Modality::Text);
        assert_ne!(Modality::Text, Modality::Image);
    }

    #[test]
    fn modality_clone_debug() {
        let m = Modality::Audio;
        let c = m.clone();
        assert_eq!(m, c);
        let _ = format!("{:?}", Modality::Video);
    }

    // -----------------------------------------------------------------------
    // ModalEmbedding
    // -----------------------------------------------------------------------

    #[test]
    fn modal_embedding_dimension() {
        let e = ModalEmbedding::new(Modality::Text, vec![1.0, 0.0, 0.0], "text-embed-small");
        assert_eq!(e.dimension, 3);
        assert_eq!(e.model_id, "text-embed-small");
    }

    #[test]
    fn modal_embedding_norm() {
        let e = ModalEmbedding::new(Modality::Image, vec![3.0, 4.0], "clip-vit-large");
        assert!((e.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn modal_embedding_zero_norm() {
        let e = ModalEmbedding::new(Modality::Text, vec![0.0, 0.0], "m");
        assert_eq!(e.norm(), 0.0);
    }

    // -----------------------------------------------------------------------
    // MediaRef
    // -----------------------------------------------------------------------

    #[test]
    fn media_ref_path_checksum_is_set() {
        let r = MediaRef::path("/tmp/photo.jpg", "image/jpeg");
        assert!(r.checksum.is_some());
        assert_eq!(r.mime_type, "image/jpeg");
        assert!(r.size_bytes.is_none());
    }

    #[test]
    fn media_ref_url_checksum_is_set() {
        let r = MediaRef::url("https://example.com/audio.mp3", "audio/mpeg");
        assert!(r.checksum.is_some());
    }

    #[test]
    fn media_ref_inline_size_and_checksum() {
        let bytes = vec![1u8, 2, 3, 4, 5];
        let r = MediaRef::inline(bytes, "application/octet-stream");
        assert_eq!(r.size_bytes, Some(5));
        assert!(r.checksum.is_some());
    }

    #[test]
    fn media_ref_inline_checksum_deterministic() {
        let b1 = vec![42u8; 16];
        let b2 = vec![42u8; 16];
        let r1 = MediaRef::inline(b1, "application/octet-stream");
        let r2 = MediaRef::inline(b2, "application/octet-stream");
        assert_eq!(r1.checksum, r2.checksum);
    }

    #[test]
    fn media_ref_inline_checksum_differs() {
        let r1 = MediaRef::inline(vec![1u8], "application/octet-stream");
        let r2 = MediaRef::inline(vec![2u8], "application/octet-stream");
        assert_ne!(r1.checksum, r2.checksum);
    }

    // -----------------------------------------------------------------------
    // FNV-1a
    // -----------------------------------------------------------------------

    #[test]
    fn fnv1a_known_value() {
        // FNV-1a 64-bit hash of empty string is the offset basis
        assert_eq!(fnv1a_64(b""), 14_695_981_039_346_656_037);
    }

    #[test]
    fn fnv1a_deterministic() {
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }

    #[test]
    fn fnv1a_different_inputs() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
    }

    // -----------------------------------------------------------------------
    // Observation
    // -----------------------------------------------------------------------

    #[test]
    fn observation_confidence_clamped_high() {
        let o = Observation::new("saw fire", "fire detected", 2.5, Modality::Image);
        assert!((o.confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn observation_confidence_clamped_low() {
        let o = Observation::new("heard something", "noise detected", -0.5, Modality::Audio);
        assert!((o.confidence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn observation_normal_confidence() {
        let o = Observation::new("text block", "english paragraph", 0.85, Modality::Text);
        assert!((o.confidence - 0.85).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // MultiModalStore — basic ops
    // -----------------------------------------------------------------------

    fn make_text_record(text: &str, emb: Vec<f32>) -> MultiModalRecord {
        MultiModalRecord {
            id: 0,
            primary_modality: Modality::Text,
            text_content: Some(text.to_string()),
            media_ref: None,
            embeddings: vec![ModalEmbedding::new(Modality::Text, emb, "text-embed-small")],
            observation: None,
            timestamp: 0.0,
            metadata: HashMap::new(),
        }
    }

    fn make_image_record(emb: Vec<f32>) -> MultiModalRecord {
        MultiModalRecord {
            id: 0,
            primary_modality: Modality::Image,
            text_content: None,
            media_ref: Some(MediaRef::path("/img/cat.jpg", "image/jpeg")),
            embeddings: vec![ModalEmbedding::new(Modality::Image, emb, "clip-vit-large")],
            observation: Some(Observation::new(
                "cat on mat",
                "domestic cat",
                0.9,
                Modality::Image,
            )),
            timestamp: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn add_record_assigns_ids() {
        let mut store = MultiModalStore::new();
        let id1 = store.add_record(make_text_record("hello", vec![1.0, 0.0]));
        let id2 = store.add_record(make_text_record("world", vec![0.0, 1.0]));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn get_record_found_and_not_found() {
        let mut store = MultiModalStore::new();
        let id = store.add_record(make_text_record("hello", vec![1.0, 0.0]));
        assert!(store.get_record(id).is_some());
        assert!(store.get_record(id + 99).is_none());
    }

    #[test]
    fn get_record_id_is_correct() {
        let mut store = MultiModalStore::new();
        let id = store.add_record(make_text_record("hi", vec![1.0]));
        assert_eq!(store.get_record(id).unwrap().id, id);
    }

    #[test]
    fn get_by_modality_filters_correctly() {
        let mut store = MultiModalStore::new();
        store.add_record(make_text_record("a", vec![1.0]));
        store.add_record(make_text_record("b", vec![0.5]));
        store.add_record(make_image_record(vec![1.0, 0.0]));

        let texts = store.get_by_modality(&Modality::Text);
        let images = store.get_by_modality(&Modality::Image);
        let audios = store.get_by_modality(&Modality::Audio);

        assert_eq!(texts.len(), 2);
        assert_eq!(images.len(), 1);
        assert_eq!(audios.len(), 0);
    }

    #[test]
    fn count_by_modality() {
        let mut store = MultiModalStore::new();
        store.add_record(make_text_record("a", vec![1.0]));
        store.add_record(make_image_record(vec![0.0, 1.0]));
        assert_eq!(store.count_by_modality(&Modality::Text), 1);
        assert_eq!(store.count_by_modality(&Modality::Image), 1);
        assert_eq!(store.count_by_modality(&Modality::Audio), 0);
    }

    #[test]
    fn get_observations_filters_by_modality() {
        let mut store = MultiModalStore::new();
        store.add_record(make_image_record(vec![1.0, 0.0])); // has observation
        store.add_record(make_text_record("no obs", vec![0.0, 1.0])); // no observation
        let obs = store.get_observations(&Modality::Image);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].interpretation, "domestic cat");

        let text_obs = store.get_observations(&Modality::Text);
        assert_eq!(text_obs.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Vector search
    // -----------------------------------------------------------------------

    #[test]
    fn search_by_modality_returns_top_k() {
        let mut store = MultiModalStore::new();
        // query = [1, 0]; record A is [1,0] (perfect match), B is [0,1] (orthogonal)
        store.add_record(make_text_record("A", vec![1.0, 0.0]));
        store.add_record(make_text_record("B", vec![0.0, 1.0]));

        let results = store.search_by_modality(&Modality::Text, &[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // Best match should be record A (sim ≈ 1.0)
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 1e-5);
        // Second should be B (sim ≈ 0.0)
        assert_eq!(results[1].0, 2);
    }

    #[test]
    fn search_by_modality_k_limits_results() {
        let mut store = MultiModalStore::new();
        for i in 0..5 {
            store.add_record(make_text_record("x", vec![i as f32, 1.0]));
        }
        let results = store.search_by_modality(&Modality::Text, &[1.0, 1.0], 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_by_modality_ignores_other_modalities() {
        let mut store = MultiModalStore::new();
        // Image record with identical embedding to query — should NOT appear
        store.add_record(make_image_record(vec![1.0, 0.0]));
        // Text record
        store.add_record(make_text_record("txt", vec![0.5, 0.5]));

        let results = store.search_by_modality(&Modality::Text, &[1.0, 0.0], 5);
        // Only the text record should be returned
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_cross_modal_sees_all_modalities() {
        let mut store = MultiModalStore::new();
        store.add_record(make_image_record(vec![1.0, 0.0]));
        store.add_record(make_text_record("txt", vec![0.0, 1.0]));

        let results = store.search_cross_modal(&[1.0, 0.0], 5);
        assert_eq!(results.len(), 2);
        // Image should score higher (sim ≈ 1.0)
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn search_cross_modal_k_limits() {
        let mut store = MultiModalStore::new();
        for i in 0..10 {
            store.add_record(make_text_record("x", vec![i as f32, 0.0]));
        }
        let results = store.search_cross_modal(&[1.0, 0.0], 4);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn search_empty_store_returns_empty() {
        let store = MultiModalStore::new();
        assert!(
            store
                .search_by_modality(&Modality::Text, &[1.0, 0.0], 5)
                .is_empty()
        );
        assert!(store.search_cross_modal(&[1.0, 0.0], 5).is_empty());
    }

    #[test]
    fn search_zero_query_returns_zero_similarity() {
        let mut store = MultiModalStore::new();
        store.add_record(make_text_record("a", vec![1.0, 0.0]));
        let results = store.search_by_modality(&Modality::Text, &[0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn search_sorted_descending() {
        let mut store = MultiModalStore::new();
        store.add_record(make_text_record("low", vec![0.0, 1.0])); // sim ≈ 0
        store.add_record(make_text_record("high", vec![1.0, 0.0])); // sim ≈ 1
        store.add_record(make_text_record("mid", vec![1.0, 1.0])); // sim ≈ 0.707

        let results = store.search_by_modality(&Modality::Text, &[1.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    // -----------------------------------------------------------------------
    // MultiModalRecord with multiple embeddings
    // -----------------------------------------------------------------------

    #[test]
    fn record_with_multiple_embeddings() {
        let mut store = MultiModalStore::new();
        let record = MultiModalRecord {
            id: 0,
            primary_modality: Modality::Image,
            text_content: Some("a cat sitting on a mat".to_string()),
            media_ref: None,
            embeddings: vec![
                ModalEmbedding::new(Modality::Image, vec![1.0, 0.0, 0.0], "clip-vit-large"),
                ModalEmbedding::new(
                    Modality::Text,
                    vec![0.0, 1.0, 0.0],
                    "text-embedding-3-small",
                ),
            ],
            observation: None,
            timestamp: 42.0,
            metadata: HashMap::new(),
        };
        let id = store.add_record(record);

        // Cross-modal query aligned with image embedding
        let img_results = store.search_cross_modal(&[1.0, 0.0, 0.0], 5);
        assert_eq!(img_results.len(), 1);
        assert_eq!(img_results[0].0, id);
        assert!((img_results[0].1 - 1.0).abs() < 1e-5);

        // Per-modality: text embedding search
        let txt_results = store.search_by_modality(&Modality::Text, &[0.0, 1.0, 0.0], 5);
        assert_eq!(txt_results.len(), 1);
        assert!((txt_results[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn embeddings_for_iterator() {
        let record = MultiModalRecord {
            id: 1,
            primary_modality: Modality::Image,
            text_content: None,
            media_ref: None,
            embeddings: vec![
                ModalEmbedding::new(Modality::Image, vec![1.0], "clip"),
                ModalEmbedding::new(Modality::Text, vec![0.5], "text"),
                ModalEmbedding::new(Modality::Image, vec![0.8], "clip-v2"),
            ],
            observation: None,
            timestamp: 0.0,
            metadata: HashMap::new(),
        };
        let image_embs: Vec<_> = record.embeddings_for(&Modality::Image).collect();
        assert_eq!(image_embs.len(), 2);
        let text_embs: Vec<_> = record.embeddings_for(&Modality::Text).collect();
        assert_eq!(text_embs.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Default / metadata
    // -----------------------------------------------------------------------

    #[test]
    fn store_default() {
        let store = MultiModalStore::default();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn record_metadata() {
        let mut store = MultiModalStore::new();
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "camera-1".to_string());
        let record = MultiModalRecord {
            id: 0,
            primary_modality: Modality::Image,
            text_content: None,
            media_ref: None,
            embeddings: vec![],
            observation: None,
            timestamp: 0.0,
            metadata: meta,
        };
        let id = store.add_record(record);
        let r = store.get_record(id).unwrap();
        assert_eq!(r.metadata.get("source").unwrap(), "camera-1");
    }
}
