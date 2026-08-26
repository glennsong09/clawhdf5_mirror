# clawhdf5 vs libhdf5: independent benchmark check

## Setup

Ran on this machine (Ryzen 9 9950X3D, no prior libhdf5 install):

- clawhdf5 side: `clawhdf5-bench`'s existing Criterion suite, run natively
  (`cargo bench -p clawhdf5-bench`).
- libhdf5 side: the Rust `hdf5-metno` binding pinned in `Cargo.toml` couldn't
  parse HDF5 2.x version strings, so no in-repo Rust comparison was possible
  against a `develop` libhdf5 build at the time. Wrote a standalone C
  harness replicating the exact shapes / chunk layouts / deflate level /
  element counts used in `h5bench_write.rs`, `h5bench_read.rs`, and
  `h5bench_meta.rs`, timed the same way (100 samples, warmup, median),
  built against a properly optimized (`-O3 -DNDEBUG`) libhdf5.
- That binding gap is now fixed (vendored `hdf5-metno-sys` patch, see
  `Cargo.toml`'s `[patch.crates-io]`), so `cargo bench -p clawhdf5-bench
  --features libhdf5-compare` runs natively against HDF5 2.x too. A
  cross-check run through the patched binding (`read_sequential/libhdf5`)
  reproduced the same throughput as the standalone C harness, confirming
  the two measurement paths agree.

## Results

| Workload | Result |
|---|---|
| Sequential read, 1K elements | clawhdf5 **~48×** faster |
| Sequential read, 10K elements | clawhdf5 **~16×** faster |
| Sequential read, 100K elements | clawhdf5 **~1.9×** faster |
| Attribute write, 4/16/64/128 attrs | clawhdf5 **~4.6–7.8×** faster, gap shrinks with scale |
| Group create, 4/16/32/64 groups | clawhdf5 **~3.3–6.8×** faster, gap shrinks with scale |
| Contiguous write, 1K/10K elements | clawhdf5 **~3–7×** faster |
| Contiguous write, 100K elements | **libhdf5 faster, ~1.27×** |
| Chunked write + deflate, 32×32 | **libhdf5 faster, ~1.1×** |
| Chunked write + deflate, 128×128 | **libhdf5 faster, ~2.4×** |
| Chunked write + deflate, 512×512 | **libhdf5 faster, ~3.6×** |

## Key finding

clawhdf5 holds a real, structural advantage on reads and small metadata
operations: it parses from an in-memory buffer with no per-open cost,
while libhdf5 pays file-open/lock/cache-init overhead on every call. That
advantage shrinks as payload size grows and I/O starts to dominate.

Chunked, deflate-compressed writes are the closest-run workload of the
suite and the least favorable to clawhdf5 — a CPU-bound path where libhdf5
is consistently faster, and the margin grows with matrix size.
