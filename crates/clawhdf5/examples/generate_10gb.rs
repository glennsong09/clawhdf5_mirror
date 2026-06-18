//! Thin wrapper so `cargo run --example generate_10gb` keeps working.
//!
//! The canonical, spec-mandated copy of this generator lives at
//! `test-vectors/gen_synthetic.rs` (P1.1 artifact requirement: commit the
//! generator script there, not the HDF5 file it produces).
include!("../../../test-vectors/gen_synthetic.rs");
