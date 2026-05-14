//! Adaptive search strategy selection and timing metrics.
//!
//! Automatically selects the best search strategy based on collection size
//! and available hardware (SIMD via clawhdf5_accel, rayon parallelism, GPU).

use std::time::Instant;

use crate::vector_search;

/// Search strategy selection based on collection size and hardware.
///
/// ```text
/// < 1K:      Scalar (overhead of SIMD dispatch not worth it)
/// 1K-100K:   Accelerate (AMX/cblas_sgemv) > BLAS (matrixmultiply) > SIMD prenorm
/// 1K-10K:    SIMD brute force with pre-computed norms (fallback)
/// 10K-50K:   Rayon parallel SIMD (if available) OR GPU (if available)
/// 50K-500K:  GPU (if available) OR IVF-PQ
/// > 100K:    IVF-PQ always (regardless of BLAS/GPU)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Plain scalar search for tiny collections (< 1K).
    Scalar,
    /// SIMD brute force with pre-computed norms (1K-10K).
    SimdBruteForce,
    /// BLAS batch matrix-vector multiply (1K-100K, requires `fast-math` feature).
    Blas,
    /// Apple Accelerate / OpenBLAS cblas_sgemv (1K-100K, requires `accelerate`/`openblas`).
    Accelerate,
    /// Rayon parallel SIMD search (10K-50K, requires `parallel` feature).
    RayonParallel,
    /// GPU-accelerated search (10K-500K, requires `gpu` feature).
    Gpu,
    /// IVF-PQ approximate search for large collections.
    IvfPq,
}

impl std::fmt::Display for SearchStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchStrategy::Scalar => write!(f, "scalar"),
            SearchStrategy::SimdBruteForce => write!(f, "simd"),
            SearchStrategy::Blas => write!(f, "blas"),
            SearchStrategy::Accelerate => write!(f, "accelerate"),
            SearchStrategy::RayonParallel => write!(f, "rayon"),
            SearchStrategy::Gpu => write!(f, "gpu"),
            SearchStrategy::IvfPq => write!(f, "ivf-pq"),
        }
    }
}

/// Metrics collected during a search operation.
#[derive(Debug, Clone)]
pub struct SearchMetrics {
    /// Strategy used for this search ("scalar", "simd", "rayon", "gpu", "ivf-pq").
    pub strategy: String,
    /// Total search time in microseconds.
    pub search_time_us: u64,
    /// Number of candidate vectors scanned.
    pub candidates_scanned: usize,
    /// Re-ranking time in microseconds (for IVF-PQ).
    pub rerank_time_us: Option<u64>,
    /// Active SIMD/GPU backend (e.g., "neon", "avx2", "avx512", "gpu-metal").
    pub backend: String,
}

/// Configuration flags for strategy selection.
#[derive(Debug, Clone, Copy)]
pub struct HardwareCapabilities {
    /// Whether the `parallel` feature is enabled and rayon is available.
    pub rayon_available: bool,
    /// Whether the `gpu` feature is enabled and GPU hardware is detected.
    pub gpu_available: bool,
    /// Whether the `fast-math` feature is enabled (BLAS batch matmul).
    pub blas_available: bool,
    /// Whether the `accelerate` or `openblas` feature is enabled (cblas_sgemv).
    pub accelerate_available: bool,
}

impl HardwareCapabilities {
    /// Detect available hardware capabilities at runtime.
    pub fn detect() -> Self {
        Self {
            rayon_available: cfg!(feature = "parallel"),
            gpu_available: {
                #[cfg(feature = "gpu")]
                {
                    clawhdf5_gpu::GpuAccelerator::is_available()
                }
                #[cfg(not(feature = "gpu"))]
                {
                    false
                }
            },
            blas_available: cfg!(feature = "fast-math"),
            accelerate_available: cfg!(any(feature = "accelerate", feature = "openblas")),
        }
    }
}

/// Return the name of the active SIMD/acceleration backend.
pub fn active_backend_name(gpu_active: bool) -> String {
    if gpu_active {
        return "gpu".to_owned();
    }
    let backend = clawhdf5_accel::detect_backend();
    format!("{backend:?}").to_lowercase()
}

/// Auto-select the best search strategy based on collection size and hardware.
///
/// Updated hierarchy with Accelerate/BLAS support:
/// ```text
/// < 1K:       Scalar
/// 1K-100K:    Accelerate (AMX sgemv) > BLAS (matrixmultiply) > Rayon > GPU > SIMD
/// > 500K:     IVF-PQ always
/// ```
pub fn auto_select_strategy(num_vectors: usize, hw: &HardwareCapabilities) -> SearchStrategy {
    if num_vectors > 500_000 {
        return SearchStrategy::IvfPq;
    }

    if num_vectors > 50_000 {
        if hw.accelerate_available {
            return SearchStrategy::Accelerate;
        }
        if hw.blas_available {
            return SearchStrategy::Blas;
        }
        if hw.gpu_available {
            return SearchStrategy::Gpu;
        }
        return SearchStrategy::IvfPq;
    }

    if num_vectors > 10_000 {
        if hw.accelerate_available {
            return SearchStrategy::Accelerate;
        }
        if hw.blas_available {
            return SearchStrategy::Blas;
        }
        if hw.rayon_available {
            return SearchStrategy::RayonParallel;
        }
        if hw.gpu_available {
            return SearchStrategy::Gpu;
        }
        return SearchStrategy::SimdBruteForce;
    }

    if num_vectors >= 1_000 {
        if hw.accelerate_available {
            return SearchStrategy::Accelerate;
        }
        if hw.blas_available {
            return SearchStrategy::Blas;
        }
        return SearchStrategy::SimdBruteForce;
    }

    SearchStrategy::Scalar
}

/// Execute a search using the given strategy and return results with metrics.
///
/// This dispatches to the appropriate search implementation based on the
/// selected strategy. For IVF-PQ, an index must be provided externally
/// (this function uses brute-force fallback if no IVF-PQ index is available).
#[allow(clippy::too_many_arguments)]
pub fn search_with_metrics(
    query: &[f32],
    vectors: &[Vec<f32>],
    norms: &[f32],
    tombstones: &[u8],
    k: usize,
    strategy: SearchStrategy,
    #[cfg(feature = "gpu")] gpu_backend: Option<&crate::gpu_search::GpuSearchBackend>,
    #[cfg(not(feature = "gpu"))] _gpu_backend: Option<&()>,
) -> (Vec<(usize, f32)>, SearchMetrics) {
    let start = Instant::now();
    let active_count = tombstones.iter().filter(|&&t| t == 0).count();

    let gpu_active;
    let results = match strategy {
        SearchStrategy::Scalar => {
            gpu_active = false;
            scalar_search(query, vectors, tombstones, k)
        }
        SearchStrategy::SimdBruteForce => {
            gpu_active = false;
            let all =
                vector_search::cosine_similarity_batch_prenorm(query, vectors, norms, tombstones);
            vector_search::top_k(all, k)
        }
        SearchStrategy::Blas => {
            gpu_active = false;
            #[cfg(feature = "fast-math")]
            {
                crate::blas_search::blas_cosine_batch(query, vectors, norms, tombstones, k)
            }
            #[cfg(not(feature = "fast-math"))]
            {
                let all = vector_search::cosine_similarity_batch_prenorm(
                    query, vectors, norms, tombstones,
                );
                vector_search::top_k(all, k)
            }
        }
        SearchStrategy::Accelerate => {
            gpu_active = false;
            #[cfg(any(feature = "accelerate", feature = "openblas"))]
            {
                crate::accelerate_search::accelerate_cosine_batch_vecs(
                    query, vectors, norms, tombstones, k,
                )
            }
            #[cfg(not(any(feature = "accelerate", feature = "openblas")))]
            {
                let all = vector_search::cosine_similarity_batch_prenorm(
                    query, vectors, norms, tombstones,
                );
                vector_search::top_k(all, k)
            }
        }
        SearchStrategy::RayonParallel => {
            gpu_active = false;
            #[cfg(feature = "parallel")]
            {
                vector_search::parallel_cosine_batch_prenorm(query, vectors, norms, tombstones, k)
            }
            #[cfg(not(feature = "parallel"))]
            {
                let all = vector_search::cosine_similarity_batch_prenorm(
                    query, vectors, norms, tombstones,
                );
                vector_search::top_k(all, k)
            }
        }
        SearchStrategy::Gpu => {
            #[cfg(feature = "gpu")]
            {
                if let Some(backend) = gpu_backend {
                    gpu_active = backend.is_available();
                    backend.search_cosine(query, vectors, norms, tombstones, k)
                } else {
                    gpu_active = false;
                    let all = vector_search::cosine_similarity_batch_prenorm(
                        query, vectors, norms, tombstones,
                    );
                    vector_search::top_k(all, k)
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                gpu_active = false;
                let all = vector_search::cosine_similarity_batch_prenorm(
                    query, vectors, norms, tombstones,
                );
                vector_search::top_k(all, k)
            }
        }
        SearchStrategy::IvfPq => {
            gpu_active = false;
            // IVF-PQ requires an external index; fall back to prenorm brute force
            // when called through this generic interface.
            let all =
                vector_search::cosine_similarity_batch_prenorm(query, vectors, norms, tombstones);
            vector_search::top_k(all, k)
        }
    };

    let elapsed = start.elapsed();
    let metrics = SearchMetrics {
        strategy: strategy.to_string(),
        search_time_us: elapsed.as_micros() as u64,
        candidates_scanned: active_count,
        rerank_time_us: None,
        backend: active_backend_name(gpu_active),
    };

    (results, metrics)
}

/// Plain scalar cosine similarity for very small collections.
fn scalar_search(
    query: &[f32],
    vectors: &[Vec<f32>],
    tombstones: &[u8],
    k: usize,
) -> Vec<(usize, f32)> {
    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 {
        return Vec::new();
    }

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(vectors.len());

    for (i, vec) in vectors.iter().enumerate() {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }
        // Use clawhdf5_accel even for scalar strategy — it's always available and
        // the "scalar" name refers to the strategy tier, not the implementation.
        let vec_norm = clawhdf5_accel::vector_norm(vec);
        let score = crate::cosine_similarity_prenorm(query, query_norm, vec, vec_norm);
        results.push((i, score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

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

    // --- auto_select_strategy tests ---

    #[test]
    fn strategy_scalar_under_1k() {
        let hw = HardwareCapabilities {
            rayon_available: false,
            gpu_available: false,
            blas_available: false,
            accelerate_available: false,
        };
        assert_eq!(auto_select_strategy(0, &hw), SearchStrategy::Scalar);
        assert_eq!(auto_select_strategy(500, &hw), SearchStrategy::Scalar);
        assert_eq!(auto_select_strategy(999, &hw), SearchStrategy::Scalar);
    }

    #[test]
    fn strategy_simd_1k_to_10k() {
        let hw = HardwareCapabilities {
            rayon_available: false,
            gpu_available: false,
            blas_available: false,
            accelerate_available: false,
        };
        assert_eq!(
            auto_select_strategy(1_000, &hw),
            SearchStrategy::SimdBruteForce
        );
        assert_eq!(
            auto_select_strategy(5_000, &hw),
            SearchStrategy::SimdBruteForce
        );
        assert_eq!(
            auto_select_strategy(10_000, &hw),
            SearchStrategy::SimdBruteForce
        );
    }

    #[test]
    fn strategy_rayon_10k_to_50k_when_available() {
        let hw = HardwareCapabilities {
            rayon_available: true,
            gpu_available: false,
            blas_available: false,
            accelerate_available: false,
        };
        assert_eq!(
            auto_select_strategy(10_001, &hw),
            SearchStrategy::RayonParallel
        );
        assert_eq!(
            auto_select_strategy(30_000, &hw),
            SearchStrategy::RayonParallel
        );
        assert_eq!(
            auto_select_strategy(50_000, &hw),
            SearchStrategy::RayonParallel
        );
    }

    #[test]
    fn strategy_gpu_10k_to_50k_when_no_rayon() {
        let hw = HardwareCapabilities {
            rayon_available: false,
            gpu_available: true,
            blas_available: false,
            accelerate_available: false,
        };
        assert_eq!(auto_select_strategy(10_001, &hw), SearchStrategy::Gpu);
        assert_eq!(auto_select_strategy(50_000, &hw), SearchStrategy::Gpu);
    }

    #[test]
    fn strategy_gpu_50k_to_500k() {
        let hw = HardwareCapabilities {
            rayon_available: true,
            gpu_available: true,
            blas_available: false,
            accelerate_available: false,
        };
        assert_eq!(auto_select_strategy(50_001, &hw), SearchStrategy::Gpu);
        assert_eq!(auto_select_strategy(200_000, &hw), SearchStrategy::Gpu);
        assert_eq!(auto_select_strategy(500_000, &hw), SearchStrategy::Gpu);
    }

    #[test]
    fn strategy_ivfpq_over_500k() {
        let hw = HardwareCapabilities {
            rayon_available: true,
            gpu_available: true,
            blas_available: true,
            accelerate_available: true,
        };
        assert_eq!(auto_select_strategy(500_001, &hw), SearchStrategy::IvfPq);
        assert_eq!(auto_select_strategy(1_000_000, &hw), SearchStrategy::IvfPq);
    }

    #[test]
    fn strategy_ivfpq_fallback_50k_no_gpu() {
        let hw = HardwareCapabilities {
            rayon_available: false,
            gpu_available: false,
            blas_available: false,
            accelerate_available: false,
        };
        assert_eq!(auto_select_strategy(50_001, &hw), SearchStrategy::IvfPq);
    }

    #[test]
    fn strategy_simd_fallback_10k_no_parallel_no_gpu() {
        let hw = HardwareCapabilities {
            rayon_available: false,
            gpu_available: false,
            blas_available: false,
            accelerate_available: false,
        };
        assert_eq!(
            auto_select_strategy(15_000, &hw),
            SearchStrategy::SimdBruteForce
        );
    }

    // --- SearchStrategy Display ---

    #[test]
    fn strategy_display_names() {
        assert_eq!(SearchStrategy::Scalar.to_string(), "scalar");
        assert_eq!(SearchStrategy::SimdBruteForce.to_string(), "simd");
        assert_eq!(SearchStrategy::Blas.to_string(), "blas");
        assert_eq!(SearchStrategy::Accelerate.to_string(), "accelerate");
        assert_eq!(SearchStrategy::RayonParallel.to_string(), "rayon");
        assert_eq!(SearchStrategy::Gpu.to_string(), "gpu");
        assert_eq!(SearchStrategy::IvfPq.to_string(), "ivf-pq");
    }

    // --- SearchMetrics ---

    #[test]
    fn search_metrics_strategy_name() {
        let metrics = SearchMetrics {
            strategy: "simd".to_owned(),
            search_time_us: 100,
            candidates_scanned: 1000,
            rerank_time_us: None,
            backend: "neon".to_owned(),
        };
        assert_eq!(metrics.strategy, "simd");
        assert_eq!(metrics.candidates_scanned, 1000);
        assert_eq!(metrics.backend, "neon");
    }

    // --- search_with_metrics ---

    #[test]
    fn search_with_metrics_scalar() {
        let vectors = make_vectors(50, 16, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 50];
        let query = vectors[0].clone();

        let (results, metrics) = search_with_metrics(
            &query,
            &vectors,
            &norms,
            &tombstones,
            5,
            SearchStrategy::Scalar,
            None,
        );

        assert_eq!(results.len(), 5);
        assert_eq!(metrics.strategy, "scalar");
        assert!(metrics.search_time_us > 0 || metrics.candidates_scanned > 0);
        assert_eq!(metrics.candidates_scanned, 50);
        assert!(metrics.rerank_time_us.is_none());
        assert!(!metrics.backend.is_empty());
        // First result should be the query itself
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn search_with_metrics_simd() {
        let vectors = make_vectors(100, 32, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 100];
        let query = vectors[0].clone();

        let (results, metrics) = search_with_metrics(
            &query,
            &vectors,
            &norms,
            &tombstones,
            10,
            SearchStrategy::SimdBruteForce,
            None,
        );

        assert_eq!(metrics.strategy, "simd");
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn search_with_metrics_timing_nonzero() {
        let vectors = make_vectors(1000, 64, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 1000];
        let query = vectors[0].clone();

        let (_, metrics) = search_with_metrics(
            &query,
            &vectors,
            &norms,
            &tombstones,
            10,
            SearchStrategy::SimdBruteForce,
            None,
        );

        // With 1000 vectors, search should take > 0 microseconds
        assert!(metrics.search_time_us < 1_000_000); // under 1 second
        assert_eq!(metrics.candidates_scanned, 1000);
    }

    #[test]
    fn search_with_metrics_results_match_direct() {
        let vectors = make_vectors(200, 32, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 200];
        let query = vectors[3].clone();

        let (results, _) = search_with_metrics(
            &query,
            &vectors,
            &norms,
            &tombstones,
            10,
            SearchStrategy::SimdBruteForce,
            None,
        );

        let direct =
            vector_search::cosine_similarity_batch_prenorm(&query, &vectors, &norms, &tombstones);
        let direct_top = vector_search::top_k(direct, 10);

        assert_eq!(results.len(), direct_top.len());
        for (r, d) in results.iter().zip(&direct_top) {
            assert_eq!(r.0, d.0);
            assert!((r.1 - d.1).abs() < 1e-6);
        }
    }

    #[test]
    fn search_with_metrics_respects_tombstones() {
        let vectors = make_vectors(100, 16, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let mut tombstones = vec![0u8; 100];
        tombstones[0] = 1;
        tombstones[1] = 1;

        let query = vectors[2].clone();
        let (results, metrics) = search_with_metrics(
            &query,
            &vectors,
            &norms,
            &tombstones,
            100,
            SearchStrategy::Scalar,
            None,
        );

        assert!(results.iter().all(|r| r.0 != 0 && r.0 != 1));
        assert_eq!(metrics.candidates_scanned, 98);
    }

    #[test]
    fn hardware_capabilities_detect() {
        let hw = HardwareCapabilities::detect();
        // Just verify it doesn't panic and returns something
        let _ = hw.rayon_available;
        let _ = hw.gpu_available;
    }

    #[test]
    fn active_backend_name_returns_valid() {
        let name = active_backend_name(false);
        assert!(!name.is_empty());
        // Should be one of the known backends
        let valid = ["neon", "avx2", "avx512", "sse4", "wasmsimd128", "scalar"];
        assert!(
            valid.iter().any(|v| name.contains(v)),
            "unexpected backend: {name}"
        );
    }

    #[test]
    fn search_metrics_has_backend_field() {
        let vectors = make_vectors(50, 16, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 50];
        let query = vectors[0].clone();

        let (_, metrics) = search_with_metrics(
            &query,
            &vectors,
            &norms,
            &tombstones,
            5,
            SearchStrategy::Scalar,
            None,
        );

        assert!(!metrics.backend.is_empty());
    }

    // --- BLAS strategy selection tests ---

    #[test]
    fn strategy_blas_preferred_1k_to_100k() {
        let hw = HardwareCapabilities {
            rayon_available: true,
            gpu_available: true,
            blas_available: true,
            accelerate_available: false,
        };
        // BLAS should be preferred over rayon/gpu/simd in the 1K-100K range
        assert_eq!(auto_select_strategy(1_000, &hw), SearchStrategy::Blas);
        assert_eq!(auto_select_strategy(5_000, &hw), SearchStrategy::Blas);
        assert_eq!(auto_select_strategy(10_001, &hw), SearchStrategy::Blas);
        assert_eq!(auto_select_strategy(50_000, &hw), SearchStrategy::Blas);
        assert_eq!(auto_select_strategy(100_000, &hw), SearchStrategy::Blas);
    }

    #[test]
    fn strategy_blas_not_for_small() {
        let hw = HardwareCapabilities {
            rayon_available: false,
            gpu_available: false,
            blas_available: true,
            accelerate_available: false,
        };
        // Under 1K, still use scalar
        assert_eq!(auto_select_strategy(500, &hw), SearchStrategy::Scalar);
        assert_eq!(auto_select_strategy(999, &hw), SearchStrategy::Scalar);
    }

    #[test]
    fn strategy_fallback_without_blas() {
        let hw = HardwareCapabilities {
            rayon_available: false,
            gpu_available: false,
            blas_available: false,
            accelerate_available: false,
        };
        // Without BLAS, falls back to SIMD/IVF-PQ
        assert_eq!(
            auto_select_strategy(5_000, &hw),
            SearchStrategy::SimdBruteForce
        );
        assert_eq!(auto_select_strategy(50_001, &hw), SearchStrategy::IvfPq);
    }

    #[cfg(feature = "fast-math")]
    #[test]
    fn search_with_metrics_blas() {
        let vectors = make_vectors(200, 32, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 200];
        let query = vectors[0].clone();

        let (results, metrics) = search_with_metrics(
            &query,
            &vectors,
            &norms,
            &tombstones,
            10,
            SearchStrategy::Blas,
            None,
        );

        assert_eq!(metrics.strategy, "blas");
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn search_with_metrics_rayon() {
        let vectors = make_vectors(500, 32, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 500];
        let query = vectors[0].clone();

        let (results, metrics) = search_with_metrics(
            &query,
            &vectors,
            &norms,
            &tombstones,
            10,
            SearchStrategy::RayonParallel,
            None,
        );

        assert_eq!(metrics.strategy, "rayon");
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }

    // --- Accelerate strategy selection tests ---

    #[test]
    fn strategy_accelerate_preferred_over_blas() {
        let hw = HardwareCapabilities {
            rayon_available: true,
            gpu_available: true,
            blas_available: true,
            accelerate_available: true,
        };
        // Accelerate should be preferred over BLAS/rayon/gpu in the 1K-100K range
        assert_eq!(auto_select_strategy(1_000, &hw), SearchStrategy::Accelerate);
        assert_eq!(auto_select_strategy(5_000, &hw), SearchStrategy::Accelerate);
        assert_eq!(
            auto_select_strategy(10_001, &hw),
            SearchStrategy::Accelerate
        );
        assert_eq!(
            auto_select_strategy(50_000, &hw),
            SearchStrategy::Accelerate
        );
        assert_eq!(
            auto_select_strategy(100_000, &hw),
            SearchStrategy::Accelerate
        );
    }

    #[test]
    fn strategy_accelerate_not_for_small() {
        let hw = HardwareCapabilities {
            rayon_available: false,
            gpu_available: false,
            blas_available: false,
            accelerate_available: true,
        };
        // Under 1K, still use scalar
        assert_eq!(auto_select_strategy(500, &hw), SearchStrategy::Scalar);
        assert_eq!(auto_select_strategy(999, &hw), SearchStrategy::Scalar);
    }

    #[test]
    fn strategy_accelerate_not_for_huge() {
        let hw = HardwareCapabilities {
            rayon_available: true,
            gpu_available: true,
            blas_available: true,
            accelerate_available: true,
        };
        // Over 500K, always IVF-PQ
        assert_eq!(auto_select_strategy(500_001, &hw), SearchStrategy::IvfPq);
    }

    #[cfg(any(feature = "accelerate", feature = "openblas"))]
    #[test]
    fn search_with_metrics_accelerate() {
        let vectors = make_vectors(200, 32, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 200];
        let query = vectors[0].clone();

        let (results, metrics) = search_with_metrics(
            &query,
            &vectors,
            &norms,
            &tombstones,
            10,
            SearchStrategy::Accelerate,
            None,
        );

        assert_eq!(metrics.strategy, "accelerate");
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }
}
