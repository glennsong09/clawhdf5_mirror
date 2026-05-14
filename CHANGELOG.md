# Changelog

## v2.1.0 (2026-04-12)

### New Features
- Expose `max_dimensions()` API on Dataset, MmapDataset, and LazyDataset
- NetCDF-4 unlimited dimension detection now works correctly
- Python bindings (`clawhdf5-py`) build and link on macOS with system Python

### Bug Fixes
- Fix GPU L2 distance test (squared vs actual L2 mismatch in test helper)
- Mark Android JNI functions as `unsafe` for Rust 2024 edition compliance
- Add `# Safety` documentation to all public unsafe extern functions
- Fix all clippy warnings: needless_range_loop, manual_strip, ptr_arg, etc.
- Rename `RelationType::from_str` to `from_label` to avoid trait confusion
- Isolate h5py interop tests with `#[ignore]` when h5py unavailable

### Code Quality
- Full rustfmt pass across workspace (61 files)
- Refine inner unsafe blocks for Rust 2024 edition style
- Zero clippy warnings, zero clippy errors across entire workspace
- 1,546 tests passing, 0 failures

## v2.0.0 (2026-03-19)

- Unified rustyhdf5 (11 crates) and edgehdf5 (4 crates) into a single workspace
- All crates renamed to clawhdf5-* prefix
- Version bumped to 2.0.0 across all crates
- Git dependencies replaced with in-workspace path dependencies
- Added `agent` feature flag to clawhdf5-agent

