//! SIMD-accelerated checksum implementations.

/// Compute Fletcher-32 checksum, auto-dispatching to the appropriate backend.
///
/// Note: the fletcher32 implementations in all backends are scalar (no SIMD
/// intrinsics), so no unsafe dispatch is needed.
pub fn checksum_fletcher32(data: &[u8]) -> u32 {
    match crate::detect_backend() {
        #[cfg(target_arch = "aarch64")]
        crate::Backend::Neon => crate::neon::checksum_fletcher32(data),

        #[cfg(target_arch = "x86_64")]
        crate::Backend::Avx2 | crate::Backend::Avx512 => crate::avx2::checksum_fletcher32(data),

        _ => crate::scalar::checksum_fletcher32(data),
    }
}
