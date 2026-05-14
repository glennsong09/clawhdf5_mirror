//! Apple Accelerate (AMX) / OpenBLAS-backed vector search via cblas_sgemv.
//!
//! On macOS, this links to Apple's Accelerate.framework which dispatches to the
//! AMX coprocessor for matrix/vector operations — matching numpy's performance.
//! On Linux, it links to OpenBLAS as a fallback.
//!
//! The key insight: `cblas_sgemv` computes `y = alpha * A * x + beta * y` in a
//! single call, where A is our N×D matrix of vectors and x is the query. This
//! gives us all N dot products at once, leveraging hardware-accelerated BLAS.

#[cfg(target_os = "macos")]
extern crate accelerate_src;

#[cfg(not(target_os = "macos"))]
extern crate openblas_src;

use cblas_sys::*;

/// Batch cosine similarity using cblas_sgemv (Accelerate AMX on macOS).
///
/// Computes all dot products in a single sgemv call, then divides by norms.
/// Returns top-k `(index, score)` pairs sorted by score descending.
///
/// `vectors_flat` is a contiguous `[N × dim]` f32 buffer in row-major order.
/// `norms` contains pre-computed L2 norms for each vector.
pub fn accelerate_cosine_batch(
    query: &[f32],
    vectors_flat: &[f32],
    norms: &[f32],
    tombstones: &[u8],
    dim: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 || vectors_flat.is_empty() {
        return Vec::new();
    }

    let n = vectors_flat.len() / dim;
    let all_active = tombstones.iter().all(|&t| t == 0);

    if all_active {
        return accelerate_cosine_all_active(query, vectors_flat, norms, dim, n, query_norm, k);
    }

    // With tombstones: pack active rows into a contiguous buffer
    let mut active_indices: Vec<usize> = Vec::with_capacity(n);
    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);

    for i in 0..n {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }
        active_indices.push(i);
        let offset = i * dim;
        flat.extend_from_slice(&vectors_flat[offset..offset + dim]);
    }

    let active_n = active_indices.len();
    if active_n == 0 {
        return Vec::new();
    }

    let mut dot_products = vec![0.0f32; active_n];

    // Single sgemv: dot_products = flat_matrix (active_n × dim) × query (dim × 1)
    // SAFETY: cblas_sgemv requires valid pointers to f32 slices with the
    // dimensions declared. Slices are contiguous and have the correct lengths.
    unsafe {
        cblas_sgemv(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            active_n as i32,
            dim as i32,
            1.0,
            flat.as_ptr(),
            dim as i32,
            query.as_ptr(),
            1,
            0.0,
            dot_products.as_mut_ptr(),
            1,
        );
    }

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(active_n);
    for (j, &orig_idx) in active_indices.iter().enumerate() {
        let denom = query_norm * norms[orig_idx];
        let score = if denom == 0.0 {
            0.0
        } else {
            dot_products[j] / denom
        };
        results.push((orig_idx, score));
    }

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Fast path when no tombstones — avoids the pack step entirely.
fn accelerate_cosine_all_active(
    query: &[f32],
    vectors_flat: &[f32],
    norms: &[f32],
    dim: usize,
    n: usize,
    query_norm: f32,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut dot_products = vec![0.0f32; n];

    // SAFETY: cblas_sgemv requires valid pointers to f32 slices with the
    // dimensions declared. Slices are contiguous and have the correct lengths.
    unsafe {
        cblas_sgemv(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            n as i32,
            dim as i32,
            1.0,
            vectors_flat.as_ptr(),
            dim as i32,
            query.as_ptr(),
            1,
            0.0,
            dot_products.as_mut_ptr(),
            1,
        );
    }

    let mut results: Vec<(usize, f32)> = dot_products
        .iter()
        .enumerate()
        .map(|(i, &dot)| {
            let denom = query_norm * norms[i];
            let score = if denom == 0.0 { 0.0 } else { dot / denom };
            (i, score)
        })
        .collect();

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Batch cosine similarity from Vec<Vec<f32>> (convenience wrapper).
///
/// Flattens vectors and delegates to `accelerate_cosine_batch`.
pub fn accelerate_cosine_batch_vecs(
    query: &[f32],
    vectors: &[Vec<f32>],
    norms: &[f32],
    tombstones: &[u8],
    k: usize,
) -> Vec<(usize, f32)> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = query.len();

    // Build flat buffer, filtering tombstones
    let mut active_indices: Vec<usize> = Vec::with_capacity(vectors.len());
    let mut flat: Vec<f32> = Vec::with_capacity(vectors.len() * dim);

    for (i, v) in vectors.iter().enumerate() {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }
        active_indices.push(i);
        flat.extend_from_slice(v);
    }

    let active_n = active_indices.len();
    if active_n == 0 {
        return Vec::new();
    }

    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 {
        return Vec::new();
    }

    let mut dot_products = vec![0.0f32; active_n];

    // SAFETY: cblas_sgemv requires valid pointers to f32 slices with the
    // dimensions declared. Slices are contiguous and have the correct lengths.
    unsafe {
        cblas_sgemv(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            active_n as i32,
            dim as i32,
            1.0,
            flat.as_ptr(),
            dim as i32,
            query.as_ptr(),
            1,
            0.0,
            dot_products.as_mut_ptr(),
            1,
        );
    }

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(active_n);
    for (j, &orig_idx) in active_indices.iter().enumerate() {
        let denom = query_norm * norms[orig_idx];
        let score = if denom == 0.0 {
            0.0
        } else {
            dot_products[j] / denom
        };
        results.push((orig_idx, score));
    }

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Compute L2 norms for all vectors in a flat buffer using cblas_snrm2.
///
/// Returns a Vec of norms, one per vector.
pub fn accelerate_batch_norms(vectors_flat: &[f32], dim: usize) -> Vec<f32> {
    if dim == 0 || vectors_flat.is_empty() {
        return Vec::new();
    }
    let n = vectors_flat.len() / dim;
    let mut norms = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * dim;
        // SAFETY: cblas_snrm2 requires a valid pointer to f32 slice of length dim.
        let norm = unsafe { cblas_snrm2(dim as i32, vectors_flat[offset..].as_ptr(), 1) };
        norms.push(norm);
    }

    norms
}

// ---------------------------------------------------------------------------
// vDSP-based dot product (macOS only) — may be faster than sgemv for small N
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
extern "C" {
    fn vDSP_dotpr(__A: *const f32, __IA: i32, __B: *const f32, __IB: i32, __C: *mut f32, __N: u32);
}

/// Batch cosine similarity using vDSP_dotpr (macOS only).
///
/// Calls vDSP_dotpr per vector — each call is hardware-accelerated but
/// there's per-call overhead. May beat sgemv for small N where the matrix
/// setup cost of sgemv dominates.
#[cfg(target_os = "macos")]
pub fn vdsp_cosine_batch(
    query: &[f32],
    vectors_flat: &[f32],
    norms: &[f32],
    tombstones: &[u8],
    dim: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 || vectors_flat.is_empty() {
        return Vec::new();
    }

    let n = vectors_flat.len() / dim;
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(n);

    for i in 0..n {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }

        let offset = i * dim;
        let mut dot: f32 = 0.0;
        // SAFETY: vDSP_dotpr requires valid f32 pointers with length dim. Offsets are
        // within the vectors_flat slice bounds.
        unsafe {
            vDSP_dotpr(
                vectors_flat[offset..].as_ptr(),
                1,
                query.as_ptr(),
                1,
                &mut dot,
                dim as u32,
            );
        }

        let denom = query_norm * norms[i];
        let score = if denom == 0.0 { 0.0 } else { dot / denom };
        results.push((i, score));
    }

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors_flat(n: usize, dim: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as f32) / 65536.0 - 0.5
        };
        (0..n * dim).map(|_| next()).collect()
    }

    fn make_vectors_vecs(n: usize, dim: usize, seed: u32) -> Vec<Vec<f32>> {
        let flat = make_vectors_flat(n, dim, seed);
        flat.chunks(dim).map(|c| c.to_vec()).collect()
    }

    fn compute_norms_flat(flat: &[f32], dim: usize) -> Vec<f32> {
        flat.chunks(dim)
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect()
    }

    // --- Test 1: Accelerate results match SIMD within f32 epsilon ---
    #[test]
    fn accelerate_matches_simd_scores() {
        let dim = 384;
        let n = 500;
        let vecs = make_vectors_vecs(n, dim, 42);
        let flat: Vec<f32> = vecs.iter().flat_map(|v| v.iter().copied()).collect();
        let norms = compute_norms_flat(&flat, dim);
        let tombstones = vec![0u8; n];
        let query = vecs[0].clone();

        let accel_results = accelerate_cosine_batch(&query, &flat, &norms, &tombstones, dim, n);
        let simd_results = crate::vector_search::cosine_similarity_batch_prenorm(
            &query,
            &vecs,
            &norms,
            &tombstones,
        );

        assert_eq!(accel_results.len(), simd_results.len());
        for (a, s) in accel_results.iter().zip(&simd_results) {
            assert_eq!(a.0, s.0, "index mismatch");
            assert!(
                (a.1 - s.1).abs() < 1e-4,
                "score mismatch at idx {}: accel={} vs simd={}",
                a.0,
                a.1,
                s.1,
            );
        }
    }

    // --- Test 2: Accelerate ranking matches SIMD ranking ---
    #[test]
    fn accelerate_ranking_matches_simd() {
        let dim = 128;
        let n = 200;
        let vecs = make_vectors_vecs(n, dim, 77);
        let flat: Vec<f32> = vecs.iter().flat_map(|v| v.iter().copied()).collect();
        let norms = compute_norms_flat(&flat, dim);
        let tombstones = vec![0u8; n];
        let query = vecs[5].clone();

        let accel_top10 = accelerate_cosine_batch(&query, &flat, &norms, &tombstones, dim, 10);
        let simd_all = crate::vector_search::cosine_similarity_batch_prenorm(
            &query,
            &vecs,
            &norms,
            &tombstones,
        );
        let simd_top10 = crate::vector_search::top_k(simd_all, 10);

        let accel_ids: Vec<usize> = accel_top10.iter().map(|r| r.0).collect();
        let simd_ids: Vec<usize> = simd_top10.iter().map(|r| r.0).collect();
        assert_eq!(accel_ids, simd_ids, "top-10 ranking should match");
    }

    // --- Test 3: Vec<Vec> wrapper matches flat variant ---
    #[test]
    fn accelerate_vecs_matches_flat() {
        let dim = 64;
        let n = 200;
        let vecs = make_vectors_vecs(n, dim, 42);
        let flat: Vec<f32> = vecs.iter().flat_map(|v| v.iter().copied()).collect();
        let norms = compute_norms_flat(&flat, dim);
        let tombstones = vec![0u8; n];
        let query = vecs[3].clone();

        let flat_results = accelerate_cosine_batch(&query, &flat, &norms, &tombstones, dim, 10);
        let vec_results = accelerate_cosine_batch_vecs(&query, &vecs, &norms, &tombstones, 10);

        assert_eq!(flat_results.len(), vec_results.len());
        for (f, v) in flat_results.iter().zip(&vec_results) {
            assert_eq!(f.0, v.0);
            assert!((f.1 - v.1).abs() < 1e-5);
        }
    }

    // --- Test 4: Tombstones properly excluded ---
    #[test]
    fn accelerate_excludes_tombstones() {
        let dim = 3;
        let flat = vec![
            1.0, 0.0, 0.0, // idx 0: identical to query
            0.0, 1.0, 0.0, // idx 1: tombstoned
            0.5, 0.5, 0.0, // idx 2: partial match
        ];
        let norms = compute_norms_flat(&flat, dim);
        let tombstones = vec![0, 1, 0];
        let query = vec![1.0, 0.0, 0.0];

        let results = accelerate_cosine_batch(&query, &flat, &norms, &tombstones, dim, 10);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(idx, _)| *idx != 1));
        assert_eq!(results[0].0, 0);
    }

    // --- Test 5: Pre-computed norms via accelerate_batch_norms ---
    #[test]
    fn accelerate_batch_norms_match_individual() {
        let dim = 384;
        let n = 100;
        let flat = make_vectors_flat(n, dim, 42);

        let batch_norms = accelerate_batch_norms(&flat, dim);
        let individual_norms = compute_norms_flat(&flat, dim);

        assert_eq!(batch_norms.len(), individual_norms.len());
        for (b, i) in batch_norms.iter().zip(&individual_norms) {
            assert!(
                (b - i).abs() < 1e-5,
                "norm mismatch: batch={b} vs individual={i}"
            );
        }
    }

    // --- Test 6: Empty vectors returns empty ---
    #[test]
    fn accelerate_empty_vectors() {
        let query = vec![1.0, 0.0, 0.0];
        let results = accelerate_cosine_batch(&query, &[], &[], &[], 3, 10);
        assert!(results.is_empty());
    }

    // --- Test 7: Zero query returns empty ---
    #[test]
    fn accelerate_zero_query() {
        let flat = vec![1.0, 0.0, 0.0];
        let norms = compute_norms_flat(&flat, 3);
        let query = vec![0.0, 0.0, 0.0];
        let results = accelerate_cosine_batch(&query, &flat, &norms, &[0], 3, 10);
        assert!(results.is_empty());
    }

    // --- Test 8: Identical vector has score ~1.0 ---
    #[test]
    fn accelerate_identical_score_one() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let flat = query.clone();
        let norms = compute_norms_flat(&flat, 4);
        let results = accelerate_cosine_batch(&query, &flat, &norms, &[0], 4, 1);
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].1 - 1.0).abs() < 1e-5,
            "expected ~1.0, got {}",
            results[0].1
        );
    }

    // --- Test 9: Orthogonal vectors have score ~0 ---
    #[test]
    fn accelerate_orthogonal_score_zero() {
        let query = vec![1.0, 0.0, 0.0];
        let flat = vec![0.0, 1.0, 0.0];
        let norms = compute_norms_flat(&flat, 3);
        let results = accelerate_cosine_batch(&query, &flat, &norms, &[0], 3, 1);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].1.abs() < 1e-5,
            "expected ~0, got {}",
            results[0].1
        );
    }

    // --- Test 10: Negative correlation detected ---
    #[test]
    fn accelerate_negative_correlation() {
        let query = vec![1.0, 0.0];
        let flat = vec![-1.0, 0.0];
        let norms = compute_norms_flat(&flat, 2);
        let results = accelerate_cosine_batch(&query, &flat, &norms, &[0], 2, 1);
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].1 - (-1.0)).abs() < 1e-5,
            "expected ~-1.0, got {}",
            results[0].1
        );
    }

    // --- Test 11: All tombstoned returns empty ---
    #[test]
    fn accelerate_all_tombstoned() {
        let flat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let norms = compute_norms_flat(&flat, 3);
        let query = vec![1.0, 0.0, 0.0];
        let results = accelerate_cosine_batch(&query, &flat, &norms, &[1, 1], 3, 10);
        assert!(results.is_empty());
    }

    // --- Test 12: Top-k truncation works ---
    #[test]
    fn accelerate_top_k_truncation() {
        let dim = 32;
        let n = 100;
        let flat = make_vectors_flat(n, dim, 42);
        let norms = compute_norms_flat(&flat, dim);
        let tombstones = vec![0u8; n];
        let query: Vec<f32> = flat[..dim].to_vec();

        let results = accelerate_cosine_batch(&query, &flat, &norms, &tombstones, dim, 5);
        assert_eq!(results.len(), 5);
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    // --- Test 13: Large-scale accelerate matches SIMD (1000 vectors, 384 dims) ---
    #[test]
    fn accelerate_large_scale_matches_simd() {
        let dim = 384;
        let n = 1000;
        let vecs = make_vectors_vecs(n, dim, 42);
        let flat: Vec<f32> = vecs.iter().flat_map(|v| v.iter().copied()).collect();
        let norms = compute_norms_flat(&flat, dim);
        let mut tombstones = vec![0u8; n];
        for i in (0..n).step_by(7) {
            tombstones[i] = 1;
        }
        let query = vecs[1].clone();

        let accel_top20 = accelerate_cosine_batch(&query, &flat, &norms, &tombstones, dim, 20);
        let simd_all = crate::vector_search::cosine_similarity_batch_prenorm(
            &query,
            &vecs,
            &norms,
            &tombstones,
        );
        let simd_top20 = crate::vector_search::top_k(simd_all, 20);

        assert_eq!(accel_top20.len(), simd_top20.len());
        for (a, s) in accel_top20.iter().zip(&simd_top20) {
            assert_eq!(a.0, s.0, "index mismatch in top-20");
            assert!(
                (a.1 - s.1).abs() < 1e-4,
                "score mismatch: accel={} vs simd={}",
                a.1,
                s.1,
            );
        }
    }

    // --- Test 14: Batch norms empty ---
    #[test]
    fn accelerate_batch_norms_empty() {
        let norms = accelerate_batch_norms(&[], 4);
        assert!(norms.is_empty());
    }

    // --- Test 15: vDSP cosine matches sgemv (macOS only) ---
    #[cfg(target_os = "macos")]
    #[test]
    fn vdsp_matches_sgemv() {
        let dim = 384;
        let n = 500;
        let flat = make_vectors_flat(n, dim, 42);
        let norms = compute_norms_flat(&flat, dim);
        let tombstones = vec![0u8; n];
        let query: Vec<f32> = flat[..dim].to_vec();

        let sgemv_results = accelerate_cosine_batch(&query, &flat, &norms, &tombstones, dim, 10);
        let vdsp_results = vdsp_cosine_batch(&query, &flat, &norms, &tombstones, dim, 10);

        assert_eq!(sgemv_results.len(), vdsp_results.len());
        for (s, v) in sgemv_results.iter().zip(&vdsp_results) {
            assert_eq!(s.0, v.0, "index mismatch");
            assert!(
                (s.1 - v.1).abs() < 1e-5,
                "score mismatch: sgemv={} vs vdsp={}",
                s.1,
                v.1,
            );
        }
    }

    // --- Test 16: Performance - 10K should complete quickly ---
    #[test]
    fn accelerate_performance_10k() {
        let dim = 384;
        let n = 10_000;
        let flat = make_vectors_flat(n, dim, 42);
        let norms = compute_norms_flat(&flat, dim);
        let tombstones = vec![0u8; n];
        let query: Vec<f32> = flat[..dim].to_vec();

        let start = std::time::Instant::now();
        let results = accelerate_cosine_batch(&query, &flat, &norms, &tombstones, dim, 10);
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 10);
        assert!(
            elapsed.as_millis() < 500,
            "Accelerate 10K took {}ms, expected < 500ms",
            elapsed.as_millis()
        );
    }
}
