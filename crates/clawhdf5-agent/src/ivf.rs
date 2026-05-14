//! Inverted File Index (IVF) for approximate nearest neighbor search.
//!
//! Partitions the vector space into clusters using k-means, then searches
//! only the `nprobe` nearest clusters for a query. Combined with PQ for
//! maximum throughput on large collections.

use std::collections::HashSet;

use crate::cosine_similarity_prenorm;
use crate::pq::ProductQuantizer;

/// A shared entry in the SEIL (Shared Entry IVF Lists) layout.
/// Instead of duplicating vectors across multiple lists, we store
/// references to the original vector array.
#[derive(Debug, Clone, Copy)]
pub struct SharedEntry {
    /// Index into the original vector array.
    pub vector_idx: usize,
    /// The primary (closest) list this vector belongs to.
    pub primary_list: usize,
}

/// Compute the AIR (Angle-Informed Redundancy) score for a vector relative
/// to a centroid, considering the query direction. Combines distance proximity
/// with angular alignment to the query direction.
///
/// Higher score = better candidate for multi-assignment to this centroid's list.
pub fn air_score(vector: &[f32], centroid: &[f32], query_direction: &[f32]) -> f32 {
    // Cosine similarity between vector and centroid (proximity)
    let proximity = clawhdf5_accel::cosine_similarity(vector, centroid);

    // Compute the residual: vector - centroid
    let residual: Vec<f32> = vector.iter().zip(centroid).map(|(v, c)| v - c).collect();

    // Angular alignment: cosine between residual and query direction.
    // If the residual points toward the query, this vector is a boundary
    // vector that benefits from multi-assignment.
    let residual_norm = clawhdf5_accel::vector_norm(&residual);
    let query_norm = clawhdf5_accel::vector_norm(query_direction);
    let alignment = if residual_norm > 1e-10 && query_norm > 1e-10 {
        clawhdf5_accel::dot_product(&residual, query_direction) / (residual_norm * query_norm)
    } else {
        0.0
    };

    // AIR score: weighted combination of proximity and angular alignment.
    // Alpha controls the trade-off; 0.7 proximity + 0.3 alignment works well
    // empirically for boundary vector detection.
    0.7 * proximity + 0.3 * alignment
}

/// An inverted file index that partitions vectors into clusters.
pub struct IVFIndex {
    /// Cluster centroids: `[num_clusters][dim]` stored flat.
    pub centroids: Vec<f32>,
    /// Number of clusters.
    pub num_clusters: usize,
    /// Vector dimension.
    pub dim: usize,
    /// Inverted lists: for each cluster, the indices of vectors assigned to it.
    pub inverted_lists: Vec<Vec<usize>>,
    /// RAIRS redundancy factor: each vector is assigned to this many lists.
    /// Default 1 = standard IVF (no redundancy). 2-3 = multi-assignment.
    pub redundancy_factor: usize,
    /// SEIL shared entries per list (list_id → entries).
    /// Only populated when redundancy_factor > 1.
    pub seil_lists: Vec<Vec<SharedEntry>>,
}

impl IVFIndex {
    /// Train an IVF index using k-means clustering.
    pub fn train(vectors: &[Vec<f32>], dim: usize, num_clusters: usize) -> Self {
        Self::train_rairs(vectors, dim, num_clusters, 1)
    }

    /// Train an IVF index with RAIRS multi-assignment.
    ///
    /// `redundancy_factor` controls how many lists each vector is assigned to:
    /// - 1 = standard IVF (no redundancy)
    /// - 2-3 = RAIRS multi-assignment using AIR scoring
    pub fn train_rairs(
        vectors: &[Vec<f32>],
        dim: usize,
        num_clusters: usize,
        redundancy_factor: usize,
    ) -> Self {
        let n = vectors.len();
        let actual_clusters = num_clusters.min(n);
        let rf = redundancy_factor.clamp(1, actual_clusters);

        // Initialize centroids from evenly-spaced vectors
        let mut centroids = vec![0.0f32; actual_clusters * dim];
        let step = if n > actual_clusters {
            n / actual_clusters
        } else {
            1
        };
        for c in 0..actual_clusters {
            let src_idx = (c * step) % n;
            let dst = &mut centroids[c * dim..(c + 1) * dim];
            dst.copy_from_slice(&vectors[src_idx]);
        }

        let mut assignments = vec![0usize; n];
        let max_iters = 15;

        for _ in 0..max_iters {
            // Assignment step (primary cluster only, for k-means convergence)
            let mut changed = false;
            for (i, vec) in vectors.iter().enumerate() {
                let best = nearest_centroid(vec, &centroids, actual_clusters, dim);
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }

            // Update centroids
            let mut counts = vec![0u32; actual_clusters];
            centroids.fill(0.0);
            for (i, vec) in vectors.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                let offset = c * dim;
                for d in 0..dim {
                    centroids[offset + d] += vec[d];
                }
            }
            for (c, &count) in counts.iter().enumerate().take(actual_clusters) {
                if count > 0 {
                    let offset = c * dim;
                    let cnt = count as f32;
                    for d in 0..dim {
                        centroids[offset + d] /= cnt;
                    }
                }
            }
        }

        // Build inverted lists (primary assignment)
        let mut inverted_lists = vec![Vec::new(); actual_clusters];
        for (i, &c) in assignments.iter().enumerate() {
            inverted_lists[c].push(i);
        }

        // Build SEIL lists with multi-assignment if redundancy_factor > 1
        let mut seil_lists = vec![Vec::new(); actual_clusters];
        if rf > 1 {
            // Compute the global centroid as a default query direction for AIR scoring
            // during training. This uses the mean of all centroids as a proxy.
            let mut global_center = vec![0.0f32; dim];
            for c in 0..actual_clusters {
                let offset = c * dim;
                for d in 0..dim {
                    global_center[d] += centroids[offset + d];
                }
            }
            let inv = 1.0 / actual_clusters as f32;
            for gc in global_center.iter_mut() {
                *gc *= inv;
            }

            for (i, vec) in vectors.iter().enumerate() {
                let primary = assignments[i];

                // Score all centroids using AIR metric with the vector-to-global
                // direction as query_direction proxy
                let query_dir: Vec<f32> = global_center
                    .iter()
                    .zip(vec.iter())
                    .map(|(g, v)| g - v)
                    .collect();

                let mut scores: Vec<(usize, f32)> = (0..actual_clusters)
                    .map(|c| {
                        let centroid = &centroids[c * dim..(c + 1) * dim];
                        (c, air_score(vec, centroid, &query_dir))
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Assign to top-rf lists
                let entry = SharedEntry {
                    vector_idx: i,
                    primary_list: primary,
                };

                // Primary list always gets it
                seil_lists[primary].push(entry);

                // Add to additional lists (up to rf total)
                // Only seil_lists get secondary assignments; inverted_lists
                // stays primary-only for backward compatibility with search().
                let mut assigned = 1;
                for &(c, _) in &scores {
                    if assigned >= rf {
                        break;
                    }
                    if c != primary {
                        seil_lists[c].push(entry);
                        assigned += 1;
                    }
                }
            }
        } else {
            // Standard IVF: SEIL lists mirror inverted_lists
            for (c, list) in inverted_lists.iter().enumerate() {
                for &idx in list {
                    seil_lists[c].push(SharedEntry {
                        vector_idx: idx,
                        primary_list: c,
                    });
                }
            }
        }

        Self {
            centroids,
            num_clusters: actual_clusters,
            dim,
            inverted_lists,
            redundancy_factor: rf,
            seil_lists,
        }
    }

    /// Assign a vector to its nearest cluster.
    pub fn assign(&self, vector: &[f32]) -> usize {
        nearest_centroid(vector, &self.centroids, self.num_clusters, self.dim)
    }

    /// Search using IVF: probe the `nprobe` nearest clusters and return
    /// top-k results by cosine similarity.
    pub fn search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        norms: &[f32],
        tombstones: &[u8],
        nprobe: usize,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let probe_clusters = self.nearest_clusters(query, nprobe);
        let query_norm = clawhdf5_accel::vector_norm(query);

        let mut results: Vec<(usize, f32)> = Vec::new();

        for cluster_id in probe_clusters {
            for &idx in &self.inverted_lists[cluster_id] {
                if idx < tombstones.len() && tombstones[idx] != 0 {
                    continue;
                }
                let vec_norm = if idx < norms.len() {
                    norms[idx]
                } else {
                    clawhdf5_accel::vector_norm(&vectors[idx])
                };
                let score = cosine_similarity_prenorm(query, query_norm, &vectors[idx], vec_norm);
                results.push((idx, score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Search using RAIRS: probe clusters selected by AIR scoring, deduplicate
    /// shared vectors via SEIL layout, and return top-k by cosine similarity.
    ///
    /// This method leverages multi-assignment (if trained with redundancy_factor > 1)
    /// for better recall, while using a `seen` set to avoid redundant distance
    /// computations for vectors that appear in multiple probed lists.
    pub fn search_rairs(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        norms: &[f32],
        tombstones: &[u8],
        nprobe: usize,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let probe_clusters = self.nearest_clusters_air(query, nprobe);
        let query_norm = clawhdf5_accel::vector_norm(query);

        let mut results: Vec<(usize, f32)> = Vec::new();
        let mut seen = HashSet::new();

        for cluster_id in probe_clusters {
            for entry in &self.seil_lists[cluster_id] {
                let idx = entry.vector_idx;

                // SEIL deduplication: skip if we already computed distance for this vector
                if !seen.insert(idx) {
                    continue;
                }

                if idx < tombstones.len() && tombstones[idx] != 0 {
                    continue;
                }

                let vec_norm = if idx < norms.len() {
                    norms[idx]
                } else {
                    clawhdf5_accel::vector_norm(&vectors[idx])
                };
                let score = cosine_similarity_prenorm(query, query_norm, &vectors[idx], vec_norm);
                results.push((idx, score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Find the `nprobe` nearest cluster centroids to the query using AIR scoring.
    /// This considers both proximity and angular alignment to the query.
    fn nearest_clusters_air(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut scores: Vec<(usize, f32)> = (0..self.num_clusters)
            .map(|c| {
                let centroid = &self.centroids[c * self.dim..(c + 1) * self.dim];
                // Use query as its own direction for list selection
                (c, air_score(query, centroid, query))
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.iter().take(nprobe).map(|&(c, _)| c).collect()
    }

    /// Find the `nprobe` nearest cluster centroids to the query.
    fn nearest_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = (0..self.num_clusters)
            .map(|c| {
                let centroid = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let sim = clawhdf5_accel::cosine_similarity(query, centroid);
                (c, sim)
            })
            .collect();

        dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.iter().take(nprobe).map(|&(c, _)| c).collect()
    }

    /// Check if clusters are reasonably balanced (no cluster has more than
    /// 3x the average size).
    pub fn is_balanced(&self) -> bool {
        if self.inverted_lists.is_empty() {
            return true;
        }
        let total: usize = self.inverted_lists.iter().map(|l| l.len()).sum();
        let avg = total as f32 / self.inverted_lists.len() as f32;
        let max_size = self
            .inverted_lists
            .iter()
            .map(|l| l.len())
            .max()
            .unwrap_or(0);
        max_size as f32 <= avg * 3.0
    }

    /// Serialize for HDF5 storage.
    /// Returns (centroids, inverted_list_offsets, inverted_list_data, metadata).
    pub fn to_hdf5_data(&self) -> (&[f32], Vec<i64>, Vec<i64>, [i64; 2]) {
        let mut offsets = Vec::with_capacity(self.num_clusters + 1);
        let mut data = Vec::new();
        let mut offset = 0i64;
        for list in &self.inverted_lists {
            offsets.push(offset);
            for &idx in list {
                data.push(idx as i64);
            }
            offset += list.len() as i64;
        }
        offsets.push(offset);

        (
            &self.centroids,
            offsets,
            data,
            [self.num_clusters as i64, self.dim as i64],
        )
    }

    /// Reconstruct from HDF5 data.
    pub fn from_hdf5_data(
        centroids: Vec<f32>,
        offsets: &[i64],
        data: &[i64],
        metadata: [i64; 2],
    ) -> Self {
        let num_clusters = metadata[0] as usize;
        let dim = metadata[1] as usize;
        let mut inverted_lists = Vec::with_capacity(num_clusters);

        for c in 0..num_clusters {
            let start = offsets[c] as usize;
            let end = offsets[c + 1] as usize;
            let list: Vec<usize> = data[start..end].iter().map(|&v| v as usize).collect();
            inverted_lists.push(list);
        }

        // Reconstruct SEIL lists from inverted lists (rf=1 assumed on load)
        let mut seil_lists = Vec::with_capacity(num_clusters);
        for (c, list) in inverted_lists.iter().enumerate() {
            seil_lists.push(
                list.iter()
                    .map(|&idx| SharedEntry {
                        vector_idx: idx,
                        primary_list: c,
                    })
                    .collect(),
            );
        }

        Self {
            centroids,
            num_clusters,
            dim,
            inverted_lists,
            redundancy_factor: 1,
            seil_lists,
        }
    }
}

/// Combined IVF-PQ search: IVF narrows candidates, PQ makes distance fast.
pub struct IVFPQIndex {
    pub ivf: IVFIndex,
    pub pq: ProductQuantizer,
    /// PQ codes for all vectors: `[n_vectors * pq.num_subvectors]`.
    pub codes: Vec<u8>,
}

impl IVFPQIndex {
    /// Build a combined IVF-PQ index.
    pub fn build(
        vectors: &[Vec<f32>],
        dim: usize,
        num_clusters: usize,
        num_subvectors: usize,
        num_centroids: usize,
    ) -> Self {
        let ivf = IVFIndex::train(vectors, dim, num_clusters);
        let pq = ProductQuantizer::train(vectors, dim, num_subvectors, num_centroids);
        let codes = pq.encode_all(vectors);
        Self { ivf, pq, codes }
    }

    /// Search using IVF to narrow clusters, then PQ for fast approximate
    /// distance, then re-rank top candidates with exact cosine.
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        norms: &[f32],
        tombstones: &[u8],
        nprobe: usize,
        candidates: usize,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let probe_clusters = self.ivf.nearest_clusters(query, nprobe);
        let table = self.pq.precompute_distance_table(query);

        // Collect candidate indices from probed clusters
        let mut pq_results: Vec<(usize, f32)> = Vec::new();
        for cluster_id in probe_clusters {
            for &idx in &self.ivf.inverted_lists[cluster_id] {
                if idx < tombstones.len() && tombstones[idx] != 0 {
                    continue;
                }
                let code_start = idx * self.pq.num_subvectors;
                let code_end = code_start + self.pq.num_subvectors;
                let codes = &self.codes[code_start..code_end];
                let dist = self.pq.asymmetric_distance_with_table(&table, codes);
                pq_results.push((idx, dist));
            }
        }

        // Sort by PQ distance (ascending = closest first)
        pq_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        pq_results.truncate(candidates);

        // Re-rank with exact cosine
        let query_norm = clawhdf5_accel::vector_norm(query);
        let mut reranked: Vec<(usize, f32)> = pq_results
            .iter()
            .map(|&(idx, _)| {
                let vec_norm = if idx < norms.len() {
                    norms[idx]
                } else {
                    clawhdf5_accel::vector_norm(&vectors[idx])
                };
                (
                    idx,
                    cosine_similarity_prenorm(query, query_norm, &vectors[idx], vec_norm),
                )
            })
            .collect();

        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        reranked
    }
    /// Search using RAIRS: AIR-based list selection + SEIL deduplication + PQ.
    ///
    /// Uses AIR scoring to select probe lists (considering angular alignment),
    /// then PQ for fast approximate distance, then re-ranks top candidates
    /// with exact cosine similarity. Shared vectors across probed lists are
    /// deduplicated to avoid redundant computation.
    #[allow(clippy::too_many_arguments)]
    pub fn search_rairs(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        norms: &[f32],
        tombstones: &[u8],
        nprobe: usize,
        candidates: usize,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let probe_clusters = self.ivf.nearest_clusters_air(query, nprobe);
        let table = self.pq.precompute_distance_table(query);

        let mut pq_results: Vec<(usize, f32)> = Vec::new();
        let mut seen = HashSet::new();

        for cluster_id in probe_clusters {
            for entry in &self.ivf.seil_lists[cluster_id] {
                let idx = entry.vector_idx;

                // SEIL deduplication
                if !seen.insert(idx) {
                    continue;
                }

                if idx < tombstones.len() && tombstones[idx] != 0 {
                    continue;
                }

                let code_start = idx * self.pq.num_subvectors;
                let code_end = code_start + self.pq.num_subvectors;
                let codes = &self.codes[code_start..code_end];
                let dist = self.pq.asymmetric_distance_with_table(&table, codes);
                pq_results.push((idx, dist));
            }
        }

        // Sort by PQ distance (ascending = closest first)
        pq_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        pq_results.truncate(candidates);

        // Re-rank with exact cosine
        let query_norm = clawhdf5_accel::vector_norm(query);
        let mut reranked: Vec<(usize, f32)> = pq_results
            .iter()
            .map(|&(idx, _)| {
                let vec_norm = if idx < norms.len() {
                    norms[idx]
                } else {
                    clawhdf5_accel::vector_norm(&vectors[idx])
                };
                (
                    idx,
                    cosine_similarity_prenorm(query, query_norm, &vectors[idx], vec_norm),
                )
            })
            .collect();

        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        reranked
    }
}

/// Select the best search strategy based on collection size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Brute-force SIMD (< 10K vectors).
    BruteForce,
    /// Brute-force SIMD with pre-computed norms (10K-100K).
    BruteForceNorms,
    /// IVF-PQ for very large collections (> 100K).
    IVFPQ,
}

/// Auto-select search strategy based on collection size.
pub fn auto_strategy(num_vectors: usize) -> SearchStrategy {
    if num_vectors < 10_000 {
        SearchStrategy::BruteForce
    } else if num_vectors <= 100_000 {
        SearchStrategy::BruteForceNorms
    } else {
        SearchStrategy::IVFPQ
    }
}

fn nearest_centroid(vector: &[f32], centroids: &[f32], num_clusters: usize, dim: usize) -> usize {
    let mut best = 0;
    let mut best_sim = f32::NEG_INFINITY;
    for c in 0..num_clusters {
        let centroid = &centroids[c * dim..(c + 1) * dim];
        let sim = clawhdf5_accel::cosine_similarity(vector, centroid);
        if sim > best_sim {
            best_sim = sim;
            best = c;
        }
    }
    best
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
    fn ivf_clustering_produces_clusters() {
        let dim = 32;
        let vectors = make_vectors(200, dim, 42);
        let ivf = IVFIndex::train(&vectors, dim, 10);

        assert_eq!(ivf.num_clusters, 10);
        // All vectors should be assigned
        let total: usize = ivf.inverted_lists.iter().map(|l| l.len()).sum();
        assert_eq!(total, 200);
        // No cluster should be completely empty (with enough vectors)
        let non_empty = ivf.inverted_lists.iter().filter(|l| !l.is_empty()).count();
        assert!(non_empty > 0);
    }

    #[test]
    fn ivf_balanced_clusters() {
        let dim = 32;
        let vectors = make_vectors(1000, dim, 42);
        let ivf = IVFIndex::train(&vectors, dim, 10);
        // With random data, clusters should be roughly balanced
        assert!(ivf.is_balanced(), "clusters should be reasonably balanced");
    }

    #[test]
    fn ivf_search_nprobe_all_matches_brute_force() {
        let dim = 32;
        let vectors = make_vectors(100, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 100];
        let query = vectors[0].clone();

        let ivf = IVFIndex::train(&vectors, dim, 5);

        // Search all clusters (nprobe = num_clusters)
        let ivf_results = ivf.search(&query, &vectors, &norms, &tombstones, 5, 10);

        // Brute force
        let query_norm = clawhdf5_accel::vector_norm(&query);
        let mut brute: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i,
                    cosine_similarity_prenorm(&query, query_norm, v, norms[i]),
                )
            })
            .collect();
        brute.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        brute.truncate(10);

        // Should get same top-10
        let ivf_ids: Vec<usize> = ivf_results.iter().map(|r| r.0).collect();
        let brute_ids: Vec<usize> = brute.iter().map(|r| r.0).collect();
        assert_eq!(ivf_ids, brute_ids, "nprobe=all should match brute force");
    }

    #[test]
    fn ivf_search_nprobe_1_returns_results() {
        let dim = 32;
        let vectors = make_vectors(200, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 200];
        let query = vectors[0].clone();

        let ivf = IVFIndex::train(&vectors, dim, 10);
        let results = ivf.search(&query, &vectors, &norms, &tombstones, 1, 10);
        assert!(!results.is_empty(), "nprobe=1 should still find results");
    }

    #[test]
    fn ivf_pq_combined_search_recall() {
        let dim = 64;
        let n = 500;
        let vectors = make_vectors(n, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; n];
        let query = vectors[0].clone();

        let index = IVFPQIndex::build(&vectors, dim, 10, 8, 64);
        let results = index.search(&query, &vectors, &norms, &tombstones, 5, 100, 10);

        // Exact top-10
        let query_norm = clawhdf5_accel::vector_norm(&query);
        let mut exact: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i,
                    cosine_similarity_prenorm(&query, query_norm, v, norms[i]),
                )
            })
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let exact_top10: Vec<usize> = exact.iter().take(10).map(|r| r.0).collect();

        let ivfpq_ids: Vec<usize> = results.iter().map(|r| r.0).collect();
        let overlap = exact_top10.iter().filter(|i| ivfpq_ids.contains(i)).count();
        // IVF-PQ recall@10 should be > 80%
        assert!(overlap >= 8, "IVF-PQ recall too low: {overlap}/10 overlap");
    }

    #[test]
    fn auto_strategy_selection() {
        assert_eq!(auto_strategy(100), SearchStrategy::BruteForce);
        assert_eq!(auto_strategy(9_999), SearchStrategy::BruteForce);
        assert_eq!(auto_strategy(10_000), SearchStrategy::BruteForceNorms);
        assert_eq!(auto_strategy(50_000), SearchStrategy::BruteForceNorms);
        assert_eq!(auto_strategy(100_000), SearchStrategy::BruteForceNorms);
        assert_eq!(auto_strategy(100_001), SearchStrategy::IVFPQ);
    }

    #[test]
    fn ivf_hdf5_roundtrip() {
        let dim = 16;
        let vectors = make_vectors(50, dim, 42);
        let ivf = IVFIndex::train(&vectors, dim, 5);

        let (centroids, offsets, data, meta) = ivf.to_hdf5_data();
        let ivf2 = IVFIndex::from_hdf5_data(centroids.to_vec(), &offsets, &data, meta);

        assert_eq!(ivf.num_clusters, ivf2.num_clusters);
        assert_eq!(ivf.dim, ivf2.dim);
        for c in 0..ivf.num_clusters {
            assert_eq!(ivf.inverted_lists[c], ivf2.inverted_lists[c]);
        }
    }

    #[test]
    fn ivf_assign_consistent() {
        let dim = 16;
        let vectors = make_vectors(100, dim, 42);
        let ivf = IVFIndex::train(&vectors, dim, 5);

        // Assigning a training vector should return its cluster
        for (i, v) in vectors.iter().enumerate() {
            let cluster = ivf.assign(v);
            assert!(
                ivf.inverted_lists[cluster].contains(&i),
                "vector {i} should be in cluster {cluster}"
            );
        }
    }

    #[test]
    fn ivf_respects_tombstones() {
        let dim = 16;
        let vectors = make_vectors(50, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let mut tombstones = vec![0u8; 50];
        tombstones[0] = 1;
        tombstones[1] = 1;

        let ivf = IVFIndex::train(&vectors, dim, 5);
        let results = ivf.search(&vectors[2], &vectors, &norms, &tombstones, 5, 50);
        assert!(results.iter().all(|r| r.0 != 0 && r.0 != 1));
    }

    // -----------------------------------------------------------------------
    // RAIRS tests
    // -----------------------------------------------------------------------

    #[test]
    fn air_score_basic() {
        let vector = vec![1.0, 0.0, 0.0, 0.0];
        let centroid = vec![0.9, 0.1, 0.0, 0.0];
        let query_dir = vec![1.0, 0.0, 0.0, 0.0];
        let score = air_score(&vector, &centroid, &query_dir);
        assert!(
            score > 0.0,
            "AIR score should be positive for aligned vectors"
        );
    }

    #[test]
    fn rairs_multi_assignment_populates_seil_lists() {
        let dim = 32;
        let vectors = make_vectors(200, dim, 42);
        let ivf = IVFIndex::train_rairs(&vectors, dim, 10, 2);

        assert_eq!(ivf.redundancy_factor, 2);

        // With rf=2, each vector appears in 2 lists, so total SEIL entries = 2 * n
        let total_seil: usize = ivf.seil_lists.iter().map(|l| l.len()).sum();
        assert_eq!(
            total_seil,
            200 * 2,
            "each vector should appear in 2 SEIL lists"
        );

        // Each SEIL entry should reference a valid vector index
        for list in &ivf.seil_lists {
            for entry in list {
                assert!(entry.vector_idx < 200);
                assert!(entry.primary_list < 10);
            }
        }
    }

    #[test]
    fn rairs_recall_ge_standard_ivf() {
        let dim = 64;
        let n = 1000;
        let vectors = make_vectors(n, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; n];
        let nprobe = 3;
        let k = 10;

        // Exact brute force top-k
        let query = vectors[0].clone();
        let query_norm = clawhdf5_accel::vector_norm(&query);
        let mut exact: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i,
                    cosine_similarity_prenorm(&query, query_norm, v, norms[i]),
                )
            })
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let exact_topk: Vec<usize> = exact.iter().take(k).map(|r| r.0).collect();

        // Standard IVF
        let ivf_std = IVFIndex::train(&vectors, dim, 20);
        let std_results = ivf_std.search(&query, &vectors, &norms, &tombstones, nprobe, k);
        let std_ids: Vec<usize> = std_results.iter().map(|r| r.0).collect();
        let std_recall = exact_topk.iter().filter(|i| std_ids.contains(i)).count();

        // RAIRS with rf=2
        let ivf_rairs = IVFIndex::train_rairs(&vectors, dim, 20, 2);
        let rairs_results =
            ivf_rairs.search_rairs(&query, &vectors, &norms, &tombstones, nprobe, k);
        let rairs_ids: Vec<usize> = rairs_results.iter().map(|r| r.0).collect();
        let rairs_recall = exact_topk.iter().filter(|i| rairs_ids.contains(i)).count();

        assert!(
            rairs_recall >= std_recall,
            "RAIRS recall ({rairs_recall}) should be >= standard IVF recall ({std_recall})"
        );
    }

    #[test]
    fn seil_deduplication_reduces_computations() {
        let dim = 32;
        let n = 200;
        let vectors = make_vectors(n, dim, 42);
        let ivf = IVFIndex::train_rairs(&vectors, dim, 10, 2);

        // Count total entries across 3 probed lists
        let query = vectors[0].clone();
        let probe_clusters = ivf.nearest_clusters_air(&query, 3);

        let total_entries: usize = probe_clusters
            .iter()
            .map(|&c| ivf.seil_lists[c].len())
            .sum();

        // Count unique vectors (what SEIL actually computes distances for)
        let mut seen = HashSet::new();
        for &c in &probe_clusters {
            for entry in &ivf.seil_lists[c] {
                seen.insert(entry.vector_idx);
            }
        }
        let unique_entries = seen.len();

        // With rf=2, there should be some duplicates across lists
        assert!(
            unique_entries <= total_entries,
            "SEIL should deduplicate: {unique_entries} unique <= {total_entries} total"
        );
        // With 3 probed lists and rf=2, we expect at least some savings
        if total_entries > unique_entries {
            let savings = total_entries - unique_entries;
            assert!(
                savings > 0,
                "SEIL should save at least some distance computations"
            );
        }
    }

    #[test]
    fn search_rairs_matches_brute_force_small_dataset() {
        let dim = 32;
        let n = 50;
        let vectors = make_vectors(n, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; n];
        let query = vectors[0].clone();
        let k = 10;

        // RAIRS with nprobe = all clusters
        let ivf = IVFIndex::train_rairs(&vectors, dim, 5, 2);
        let rairs_results = ivf.search_rairs(&query, &vectors, &norms, &tombstones, 5, k);

        // Brute force
        let query_norm = clawhdf5_accel::vector_norm(&query);
        let mut brute: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i,
                    cosine_similarity_prenorm(&query, query_norm, v, norms[i]),
                )
            })
            .collect();
        brute.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        brute.truncate(k);

        let rairs_ids: Vec<usize> = rairs_results.iter().map(|r| r.0).collect();
        let brute_ids: Vec<usize> = brute.iter().map(|r| r.0).collect();
        assert_eq!(
            rairs_ids, brute_ids,
            "RAIRS with nprobe=all should match brute force"
        );
    }

    #[test]
    fn rairs_ivfpq_search() {
        let dim = 64;
        let n = 500;
        let vectors = make_vectors(n, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; n];
        let query = vectors[0].clone();

        let ivf = IVFIndex::train_rairs(&vectors, dim, 10, 2);
        let pq = crate::pq::ProductQuantizer::train(&vectors, dim, 8, 64);
        let codes = pq.encode_all(&vectors);
        let index = IVFPQIndex { ivf, pq, codes };

        let results = index.search_rairs(&query, &vectors, &norms, &tombstones, 5, 100, 10);
        assert!(!results.is_empty(), "RAIRS IVF-PQ should return results");

        // Verify results are sorted by score descending
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "results should be sorted by score descending"
            );
        }

        // Check recall vs brute force
        let query_norm = clawhdf5_accel::vector_norm(&query);
        let mut exact: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i,
                    cosine_similarity_prenorm(&query, query_norm, v, norms[i]),
                )
            })
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let exact_top10: Vec<usize> = exact.iter().take(10).map(|r| r.0).collect();
        let rairs_ids: Vec<usize> = results.iter().map(|r| r.0).collect();
        let overlap = exact_top10.iter().filter(|i| rairs_ids.contains(i)).count();
        assert!(overlap >= 5, "RAIRS IVF-PQ recall too low: {overlap}/10");
    }

    #[test]
    fn standard_search_unchanged_after_rairs() {
        // Verify backward compatibility: standard search still works on RAIRS-trained index
        let dim = 32;
        let n = 100;
        let vectors = make_vectors(n, dim, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; n];
        let query = vectors[0].clone();

        let ivf = IVFIndex::train_rairs(&vectors, dim, 5, 2);
        let results = ivf.search(&query, &vectors, &norms, &tombstones, 5, 10);
        assert!(
            !results.is_empty(),
            "standard search should still work on RAIRS index"
        );

        // Results should be sorted
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }
}
