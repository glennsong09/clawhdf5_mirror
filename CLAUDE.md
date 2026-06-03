# clawhdf5

## Purpose
Pure-Rust HDF5 format implementation with HNSW vector search, WAL-backed persistence, agent memory storage, and GPU-accelerated I/O. Used by ZeroClaw as its persistent memory and knowledge graph backend.

## Architecture

Cargo workspace with 17 crates under `crates/`:

| Crate | Role |
|-------|------|
| `clawhdf5-types` | Shared type definitions and physical constants |
| `clawhdf5-format` | HDF5 binary spec parser (superblock, B-tree, heap) |
| `clawhdf5-io` | Read/write implementation |
| `clawhdf5-filters` | Compression filters (gzip, LZ4, Zstd, Blosc) |
| `clawhdf5-derive` | Proc-macro derive for HDF5-serializable structs |
| `clawhdf5` | Main facade crate |
| `clawhdf5-netcdf4` | NetCDF-4 compatibility layer |
| `clawhdf5-ann` | HNSW approximate nearest-neighbor vector index |
| `clawhdf5-agent` | Agent memory, session history, knowledge graph storage |
| `clawhdf5-gpu` | GPU-accelerated I/O via CubeCL |
| `clawhdf5-accel` | CPU SIMD acceleration path |
| `clawhdf5-migrate` | Schema migration engine |
| `clawhdf5-android` | Android JNI bindings |
| `clawhdf5-cli` | Command-line interface |
| `clawhdf5-napi` | Node.js native addon bindings |
| `clawhdf5-py` | PyO3 Python bindings |
| `clawhdf5-bench` | Benchmark suite |

## Key Features
- Zero-dependency HDF5 read/write (no libhdf5 C library required)
- HNSW vector index for semantic similarity search over agent memories — the
  `clawhdf5-agent` `hnsw` feature is **on by default**, so `hybrid_search` uses
  the approximate `clawhdf5-ann` index for the vector stage (the index mirrors
  the cache and self-heals on drift). Build the agent with
  `--no-default-features --features float16` to force the exact linear cosine scan.
- WAL (write-ahead log) for crash-safe persistence
- GPU-accelerated batch I/O for large dataset processing
- Python and Node.js bindings for cross-language use
- NetCDF-4 compatibility for scientific data interop

## Workflows

### Build
```bash
cargo build --release
```

### Test
```bash
cargo test --workspace
```

### CLI
```bash
cargo run -p clawhdf5-cli -- --help
# inspect, dump, index, search subcommands
```

### Python bindings
```bash
cd crates/clawhdf5-py
maturin develop
python -c "import clawhdf5; print(clawhdf5.__version__)"
```

## Integration
ZeroClaw imports this as a Cargo feature (`clawhdf5` feature flag) to persist agent memory with HNSW vector search for context retrieval.
