//! Product Quantization (PQ) for approximate nearest neighbor search.
//!
//! Compresses high-dimensional vectors into compact codes by splitting each
//! vector into subvectors and quantizing each subvector to its nearest
//! centroid from a learned codebook.
//!
//! Default: 384-dim → 48 subvectors × 256 centroids = 48 bytes per vector (8x compression).

use crate::cosine_similarity_prenorm;

/// Product Quantizer with learned codebooks.
pub struct ProductQuantizer {
    /// Number of sub-vector segments.
    pub num_subvectors: usize,
    /// Number of centroids per sub-vector (max 256 for u8 codes).
    pub num_centroids: usize,
    /// Original vector dimension.
    pub dim: usize,
    /// Dimension of each sub-vector.
    pub sub_dim: usize,
    /// Codebook: `[num_subvectors][num_centroids][sub_dim]` stored flat.
    /// Layout: codebook[sv * num_centroids * sub_dim + c * sub_dim + d]
    pub codebook: Vec<f32>,
}

impl ProductQuantizer {
    /// Train a product quantizer from a set of vectors using k-means.
    ///
    /// `num_subvectors` must evenly divide the vector dimension.
    /// `num_centroids` must be <= 256 (for u8 encoding).
    pub fn train(
        vectors: &[Vec<f32>],
        dim: usize,
        num_subvectors: usize,
        num_centroids: usize,
    ) -> Self {
        assert!(num_centroids <= 256, "num_centroids must be <= 256");
        assert!(
            dim.is_multiple_of(num_subvectors),
            "dim must be divisible by num_subvectors"
        );
        assert!(!vectors.is_empty(), "need at least one vector to train");

        let sub_dim = dim / num_subvectors;
        let mut codebook = vec![0.0f32; num_subvectors * num_centroids * sub_dim];

        let actual_centroids = num_centroids.min(vectors.len());

        for sv in 0..num_subvectors {
            let offset = sv * sub_dim;
            // Extract sub-vectors for this segment
            let sub_vecs: Vec<&[f32]> = vectors
                .iter()
                .map(|v| &v[offset..offset + sub_dim])
                .collect();

            // Initialize centroids from first `actual_centroids` vectors
            let cb_offset = sv * num_centroids * sub_dim;
            for c in 0..actual_centroids {
                let src = sub_vecs[c % sub_vecs.len()];
                let dst = &mut codebook[cb_offset + c * sub_dim..cb_offset + (c + 1) * sub_dim];
                dst.copy_from_slice(src);
            }
            // Duplicate if we have fewer vectors than centroids
            for c in actual_centroids..num_centroids {
                let src_c = c % actual_centroids;
                let (src_start, dst_start) = (cb_offset + src_c * sub_dim, cb_offset + c * sub_dim);
                for d in 0..sub_dim {
                    codebook[dst_start + d] = codebook[src_start + d];
                }
            }

            // K-means iterations
            let max_iters = 10;
            let mut assignments = vec![0u8; sub_vecs.len()];

            for _ in 0..max_iters {
                // Assignment step
                let mut changed = false;
                for (vi, sv_data) in sub_vecs.iter().enumerate() {
                    let mut best_c = 0u8;
                    let mut best_dist = f32::MAX;
                    for c in 0..actual_centroids {
                        let cb_start = cb_offset + c * sub_dim;
                        let centroid = &codebook[cb_start..cb_start + sub_dim];
                        let dist = l2_sq(sv_data, centroid);
                        if dist < best_dist {
                            best_dist = dist;
                            best_c = c as u8;
                        }
                    }
                    if assignments[vi] != best_c {
                        assignments[vi] = best_c;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }

                // Update step: recompute centroids as mean of assigned vectors
                let mut counts = vec![0u32; actual_centroids];
                // Zero out centroids
                for c in 0..actual_centroids {
                    let start = cb_offset + c * sub_dim;
                    for d in 0..sub_dim {
                        codebook[start + d] = 0.0;
                    }
                }
                for (vi, sv_data) in sub_vecs.iter().enumerate() {
                    let c = assignments[vi] as usize;
                    counts[c] += 1;
                    let start = cb_offset + c * sub_dim;
                    for d in 0..sub_dim {
                        codebook[start + d] += sv_data[d];
                    }
                }
                for (c, &count) in counts.iter().enumerate().take(actual_centroids) {
                    if count > 0 {
                        let start = cb_offset + c * sub_dim;
                        let cnt = count as f32;
                        for d in 0..sub_dim {
                            codebook[start + d] /= cnt;
                        }
                    }
                }
            }
        }

        Self {
            num_subvectors,
            num_centroids,
            dim,
            sub_dim,
            codebook,
        }
    }

    /// Encode a vector into PQ codes (one u8 per subvector).
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim);
        let mut codes = Vec::with_capacity(self.num_subvectors);

        for sv in 0..self.num_subvectors {
            let v_offset = sv * self.sub_dim;
            let sub = &vector[v_offset..v_offset + self.sub_dim];
            let cb_offset = sv * self.num_centroids * self.sub_dim;

            let mut best_c = 0u8;
            let mut best_dist = f32::MAX;
            for c in 0..self.num_centroids {
                let c_start = cb_offset + c * self.sub_dim;
                let centroid = &self.codebook[c_start..c_start + self.sub_dim];
                let dist = l2_sq(sub, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c as u8;
                }
            }
            codes.push(best_c);
        }
        codes
    }

    /// Decode PQ codes back to an approximate vector.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        assert_eq!(codes.len(), self.num_subvectors);
        let mut result = Vec::with_capacity(self.dim);

        for (sv, &code) in codes.iter().enumerate() {
            let cb_offset = sv * self.num_centroids * self.sub_dim;
            let c_start = cb_offset + code as usize * self.sub_dim;
            result.extend_from_slice(&self.codebook[c_start..c_start + self.sub_dim]);
        }
        result
    }

    /// Precompute distance table for asymmetric distance computation.
    ///
    /// Returns a table of shape `[num_subvectors][num_centroids]` (stored flat)
    /// containing the squared L2 distance from each query sub-vector to each
    /// centroid.
    pub fn precompute_distance_table(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim);
        let mut table = Vec::with_capacity(self.num_subvectors * self.num_centroids);

        for sv in 0..self.num_subvectors {
            let q_offset = sv * self.sub_dim;
            let q_sub = &query[q_offset..q_offset + self.sub_dim];
            let cb_offset = sv * self.num_centroids * self.sub_dim;

            for c in 0..self.num_centroids {
                let c_start = cb_offset + c * self.sub_dim;
                let centroid = &self.codebook[c_start..c_start + self.sub_dim];
                table.push(l2_sq(q_sub, centroid));
            }
        }
        table
    }

    /// Compute asymmetric distance between query and encoded vector.
    ///
    /// Uses a precomputed distance table for speed — this is just
    /// `num_subvectors` table lookups + additions.
    pub fn asymmetric_distance_with_table(&self, table: &[f32], codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        for (sv, &code) in codes.iter().enumerate() {
            dist += table[sv * self.num_centroids + code as usize];
        }
        dist
    }

    /// Compute asymmetric distance between a query and an encoded vector.
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let table = self.precompute_distance_table(query);
        self.asymmetric_distance_with_table(&table, codes)
    }

    /// Search a collection of PQ-encoded vectors and return the top-k nearest
    /// by asymmetric distance (smallest distance = most similar).
    ///
    /// `all_codes` is a flat buffer: `[n_vectors * num_subvectors]`.
    /// `tombstones` marks deleted vectors.
    pub fn search(
        &self,
        query: &[f32],
        all_codes: &[u8],
        tombstones: &[u8],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let table = self.precompute_distance_table(query);
        let n = all_codes.len() / self.num_subvectors;
        let mut results: Vec<(usize, f32)> = Vec::with_capacity(n);

        for i in 0..n {
            if i < tombstones.len() && tombstones[i] != 0 {
                continue;
            }
            let codes = &all_codes[i * self.num_subvectors..(i + 1) * self.num_subvectors];
            let dist = self.asymmetric_distance_with_table(&table, codes);
            results.push((i, dist));
        }

        // Sort by distance ascending (smaller = closer)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Search with PQ then re-rank top candidates with exact cosine similarity.
    ///
    /// Returns `(index, cosine_score)` pairs sorted by score descending.
    pub fn search_rerank(
        &self,
        query: &[f32],
        all_codes: &[u8],
        vectors: &[Vec<f32>],
        tombstones: &[u8],
        candidates: usize,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let pq_results = self.search(query, all_codes, tombstones, candidates);
        let query_norm = clawhdf5_accel::vector_norm(query);

        let mut reranked: Vec<(usize, f32)> = pq_results
            .iter()
            .map(|&(idx, _)| {
                let vec_norm = clawhdf5_accel::vector_norm(&vectors[idx]);
                let score = cosine_similarity_prenorm(query, query_norm, &vectors[idx], vec_norm);
                (idx, score)
            })
            .collect();

        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        reranked
    }

    /// Encode all vectors and return flat code buffer.
    pub fn encode_all(&self, vectors: &[Vec<f32>]) -> Vec<u8> {
        let mut all_codes = Vec::with_capacity(vectors.len() * self.num_subvectors);
        for v in vectors {
            all_codes.extend(self.encode(v));
        }
        all_codes
    }

    /// Serialize the quantizer state to flat data for HDF5 storage.
    /// Returns (codebook_flat, metadata: [num_subvectors, num_centroids, dim]).
    pub fn to_hdf5_data(&self) -> (&[f32], [i64; 3]) {
        (
            &self.codebook,
            [
                self.num_subvectors as i64,
                self.num_centroids as i64,
                self.dim as i64,
            ],
        )
    }

    /// Reconstruct from HDF5 data.
    pub fn from_hdf5_data(codebook: Vec<f32>, metadata: [i64; 3]) -> Self {
        let num_subvectors = metadata[0] as usize;
        let num_centroids = metadata[1] as usize;
        let dim = metadata[2] as usize;
        let sub_dim = dim / num_subvectors;
        Self {
            num_subvectors,
            num_centroids,
            dim,
            sub_dim,
            codebook,
        }
    }
}

/// Squared L2 distance between two slices.
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize, seed: u32) -> Vec<Vec<f32>> {
        let mut s = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as f32) / 65536.0 - 0.5
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    #[test]
    fn encode_decode_roundtrip() {
        let dim = 384;
        let vectors = make_vectors(200, dim, 42);
        let pq = ProductQuantizer::train(&vectors, dim, 48, 256);

        // Check reconstruction error
        let mut total_error = 0.0f32;
        for v in &vectors {
            let codes = pq.encode(v);
            let decoded = pq.decode(&codes);
            assert_eq!(decoded.len(), dim);
            let error: f32 = v.iter().zip(&decoded).map(|(a, b)| (a - b) * (a - b)).sum();
            total_error += error;
        }
        let avg_error = total_error / vectors.len() as f32 / dim as f32;
        // Reconstruction error should be reasonable
        assert!(
            avg_error < 0.1,
            "avg per-dim reconstruction error too high: {avg_error}"
        );
    }

    #[test]
    fn pq_code_size() {
        let dim = 384;
        let num_sub = 48;
        let vectors = make_vectors(100, dim, 42);
        let pq = ProductQuantizer::train(&vectors, dim, num_sub, 256);
        let codes = pq.encode(&vectors[0]);
        assert_eq!(codes.len(), num_sub); // 48 bytes per vector
    }

    #[test]
    fn asymmetric_distance_basic() {
        let dim = 16;
        let vectors = make_vectors(50, dim, 42);
        let pq = ProductQuantizer::train(&vectors, dim, 4, 16);

        let query = &vectors[0];
        let codes = pq.encode(&vectors[1]);

        let dist = pq.asymmetric_distance(query, &codes);
        assert!(dist >= 0.0, "distance should be non-negative");
    }

    #[test]
    fn pq_search_returns_closest() {
        let dim = 32;
        let mut vectors = make_vectors(100, dim, 42);
        // Make vectors[0] identical to query
        let query = vectors[0].clone();
        vectors[0] = query.clone();

        let pq = ProductQuantizer::train(&vectors, dim, 8, 32);
        let all_codes = pq.encode_all(&vectors);
        let tombstones = vec![0u8; 100];

        let results = pq.search(&query, &all_codes, &tombstones, 10);
        assert!(!results.is_empty());
        // The query itself (index 0) should be in top results
        let top_indices: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(
            top_indices.contains(&0),
            "query vector should be in top results"
        );
    }

    #[test]
    fn pq_search_respects_tombstones() {
        let dim = 16;
        let vectors = make_vectors(20, dim, 42);
        let pq = ProductQuantizer::train(&vectors, dim, 4, 16);
        let all_codes = pq.encode_all(&vectors);
        let mut tombstones = vec![0u8; 20];
        tombstones[0] = 1;

        let results = pq.search(&vectors[0], &all_codes, &tombstones, 20);
        assert!(results.iter().all(|r| r.0 != 0));
    }

    #[test]
    fn pq_search_rerank_improves_quality() {
        let dim = 64;
        let vectors = make_vectors(500, dim, 42);
        let query = vectors[0].clone();

        let pq = ProductQuantizer::train(&vectors, dim, 8, 64);
        let all_codes = pq.encode_all(&vectors);
        let tombstones = vec![0u8; 500];

        let reranked = pq.search_rerank(&query, &all_codes, &vectors, &tombstones, 100, 10);
        assert!(reranked.len() <= 10);
        // First result should have high cosine similarity (it's the query itself)
        assert!(reranked[0].1 > 0.9, "top reranked score: {}", reranked[0].1);
    }

    #[test]
    fn distance_table_precomputation() {
        let dim = 16;
        let vectors = make_vectors(50, dim, 42);
        let pq = ProductQuantizer::train(&vectors, dim, 4, 16);

        let query = &vectors[0];
        let codes = pq.encode(&vectors[1]);

        // Distance with table should equal without table
        let table = pq.precompute_distance_table(query);
        let dist_table = pq.asymmetric_distance_with_table(&table, &codes);
        let dist_direct = pq.asymmetric_distance(query, &codes);
        assert!((dist_table - dist_direct).abs() < 1e-6);
    }

    #[test]
    fn pq_hdf5_roundtrip() {
        let dim = 32;
        let vectors = make_vectors(50, dim, 42);
        let pq = ProductQuantizer::train(&vectors, dim, 8, 32);

        let (cb, meta) = pq.to_hdf5_data();
        let pq2 = ProductQuantizer::from_hdf5_data(cb.to_vec(), meta);

        assert_eq!(pq.num_subvectors, pq2.num_subvectors);
        assert_eq!(pq.num_centroids, pq2.num_centroids);
        assert_eq!(pq.dim, pq2.dim);
        assert_eq!(pq.codebook, pq2.codebook);
    }

    #[test]
    fn pq_asymmetric_ranking_reasonable_recall() {
        // Check that PQ ranking has reasonable overlap with exact ranking
        let dim = 64;
        let n = 500;
        let vectors = make_vectors(n, dim, 42);
        let query = vectors[0].clone();

        // Exact top-10 by cosine similarity
        let mut exact: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, clawhdf5_accel::cosine_similarity(&query, v)))
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let exact_top10: Vec<usize> = exact.iter().take(10).map(|r| r.0).collect();

        // PQ approximate top-20 then check overlap with exact top-10
        let pq = ProductQuantizer::train(&vectors, dim, 8, 64);
        let all_codes = pq.encode_all(&vectors);
        let tombstones = vec![0u8; n];
        let pq_top20 = pq.search(&query, &all_codes, &tombstones, 20);
        let pq_indices: Vec<usize> = pq_top20.iter().map(|r| r.0).collect();

        let overlap = exact_top10
            .iter()
            .filter(|i| pq_indices.contains(i))
            .count();
        // Recall should be at least 50% (5 out of 10)
        assert!(
            overlap >= 5,
            "PQ recall too low: {overlap}/10 overlap with exact top-10 in PQ top-20"
        );
    }
}
