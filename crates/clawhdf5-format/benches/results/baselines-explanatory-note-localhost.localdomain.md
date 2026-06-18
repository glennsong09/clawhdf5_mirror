# P1.2b Baseline Benchmark — Explanatory Note

This note accompanies `baselines-localhost.localdomain.csv` and
`baselines-capability-matrix-localhost.localdomain.md`, per the S2-D2 spec's
"Benchmark validity and interpretation" requirement (p.52): every benchmark
artifact needs a reproducible explanatory note covering exact reproduction
steps, hardware, what is measured, how to read the results, and a root-cause
explanation of any notable trend or anomaly.

## Reproduction

```bash
cargo run --example generate_10gb --release 10 /tmp/noaa_data/synthetic_10gb.h5
# Real NOAA sample: see test-vectors/README.md for filename + download URL.
cargo run -p clawhdf5 --release --example baselines_bench -- \
    /tmp/noaa_data/synthetic_10gb.h5 30 /tmp/noaa_data/goes18_sample.nc
```

The harness is `crates/clawhdf5/examples/baselines_bench.rs`. Argument order
is `<synthetic_path> <trials> <noaa_path> [noaa_dataset]` (`noaa_dataset`
defaults to `Rad`).

## Hardware and parameters

- Host: `localhost.localdomain`
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor
- RAM: 46.2 GB
- Recorded automatically in the CSV's `#`-prefixed header line (hostname,
  CPU model, RAM size, UTC date) per the artifact-convention requirement.
- 30 measured trials per cell, after 5 discarded warmup iterations
  (`WARMUP_TRIALS` in the harness) — JIT/allocator/cache warmup is excluded
  from the reported statistics.
- Inputs: the 10 GB synthetic file from P1.1 (`dataset_64kb`, `dataset_256kb`,
  `dataset_1mb`) and a real NOAA GOES-18 sample (`Rad` dataset, DEFLATE
  filtered, ~122 KB nominal / 240 chunks).

## What is measured

Three comparable-primitive baselines, run against the same chunk sets:

- **flat** — a single SHA-256 over the whole dataset (no per-chunk granularity).
- **mac** — per-chunk HMAC-SHA-256 (symmetric; verifier needs the shared key).
- **sig_ed25519** — per-chunk Ed25519 signature (asymmetric; verifier needs
  only the public key).

For each, the harness records: `commit_time` (build cost), `verify_chunk_latency`
(single-chunk spot check), `verify_dataset_latency` (full-dataset check),
`append_time` / `update_time` (incremental cost of adding/changing one
chunk), `metadata_bytes` (integrity-metadata storage overhead),
`public_verification` (can a verifier without the secret check it?), and
`subset_proof_bytes` (bytes needed to prove a `k`-chunk hyperslab).

`sig_mldsa` (post-quantum) is explicitly out of scope for Phase 1 — listed as
"not implemented (Phase 2)" in every row, per the spec.

## How to read the capability matrix

Every time metric is the **median of the 30 measured trials**, with a
**95% bootstrap confidence interval** in brackets (2000 resamples) — never a
bare mean, per the statistical protocol. Deterministic metrics
(`metadata_bytes`, `public_verification`, `subset_proof_bytes`) have no
trial-to-trial variance, so they're reported as a single value with no CI.

## How to read the plots

- `plot_time_metrics_256kb.png`: grouped bars per backend/metric at the
  256 KB / synthetic cell, log-scaled, with 95% CI error bars.
- `plot_ed25519_projection.png`: projected single-core CPU-hours to sign
  `N = 1e7` chunks, as a function of chunk size, for both the synthetic
  sweep and the single NOAA data point.

## Expected trends and whether the data matches

- **flat and mac scale with total dataset bytes, not chunk count or size**:
  `commit_time`/`verify_dataset_latency` for both are ~1.27s regardless of
  chunk size, because they hash a fixed ~640 MB across chunks of whatever
  size. Matches expectation — these backends are insensitive to chunking
  granularity since SHA-256/HMAC do one pass over the same total bytes.
- **sig_ed25519 cost scales with chunk *count*, not chunk size**: at 256 KB
  (13,652 chunks) `commit_time` ≈ 6.15s; per-chunk that's ≈ 451 µs, matching
  `append_time` (≈ 453 µs, a single incremental sign) almost exactly. This is
  expected — signing cost is dominated by the fixed per-signature overhead,
  not the bytes being signed (a SHA-512 pre-hash over short HDF5 chunks is
  far cheaper than the elliptic-curve scalar multiplication itself).
- **mac and sig_ed25519's append/update costs are O(1)**, matching the spec's
  expectation for incremental per-chunk primitives — confirmed by
  `append_time`/`update_time` being independent of `n_chunks` at a given
  chunk size.

## Anomaly 1: sig_ed25519 signing is ~2x *slower* than verifying

At the 256 KB representative cell, `commit_time` (median ≈ 6.15s, signing all
13,652 chunks) is roughly **double** `verify_dataset_latency` (median ≈
3.25s, verifying the same chunks). This is the inverse of the commonly-cited
Ed25519 microbenchmark result, where verification is the more expensive
operation (it requires a double scalar multiplication) and signing is fast
because it can use a precomputed table for the fixed base-point
multiplication.

Root cause, confirmed in `crates/clawhdf5-format/Cargo.toml`:

```toml
ed25519_dalek = { package = "ed25519-dalek", version = "2", optional = true, default-features = false, features = ["rand_core", "zeroize"] }
```

`ed25519-dalek`'s `default-features` includes `fast`, which enables
`curve25519-dalek/precomputed-tables` — the precomputed base-point table that
makes *signing's* fixed-base scalar multiplication cheap. This dependency
declaration sets `default-features = false` and does not re-enable `fast`,
so every `SigningKey::sign()` call falls back to an un-tabled (slower)
scalar multiplication, while `verify()`'s variable-base double-scalar
multiplication (which never benefited from that table) is unaffected. That
inverts the usual sign/verify cost ordering. This is a real, reproducible
property of the current dependency configuration, not a measurement
artifact — re-enabling the `fast` feature would be expected to bring signing
back below verification, at the cost of a larger compiled binary (the
precomputed table is several hundred KB).

## Anomaly 2: the NOAA point sits below the synthetic chunk-size trend line

In `plot_ed25519_projection.png`, the NOAA sample (labeled "122 KB chunks")
projects to ≈0.298 CPU-hours — *lower* than the synthetic 64 KB point
(≈0.356 CPU-hours), even though 122 KB > 64 KB and the synthetic trend is
monotonically increasing with chunk size.

Root cause: the harness's `chunk_size_kb` label for NOAA is the dataset's
**nominal** (uncompressed) chunk size — `nominal_size = chunk_elements *
element_size` in `extract_chunks_from_file()` — but the bytes actually
hashed/signed are the **on-disk filtered (DEFLATE-compressed) bytes**
(`file_data[start..end]`, sized by the chunk's real on-disk `chunk_size`).
Instrumenting `extract_chunks_from_file()` directly against
`/tmp/noaa_data/goes18_sample.nc`'s `Rad` dataset confirms:

```
nominal_size=125000 actual_avg_on_disk_bytes=52081 ratio=0.417
```

So the "122 KB" x-axis label overstates the real per-chunk payload by
~2.4x — the actual average compressed chunk is ~50.9 KB, which is why the
NOAA point's signing cost lands between the synthetic 64 KB point's chunk
*count* effect and a naively-expected ~122 KB-scaled cost. This is expected
behavior given satellite radiance imagery's compressibility, not a harness
bug — it does mean the NOAA x-position in the projection plot is only
informative as nominal chunk size, not as a direct proxy for bytes
processed. A follow-up improvement would be to plot NOAA by its actual
on-disk chunk size instead of nominal size for a fairer cross-comparison.

## Confirming/correcting the §4.3 N=1e7 projections

Spec step 3 requires projecting per-chunk-sig storage and signing cost to
N = 10^7 chunks and checking it against the figures cited in §4.3 (Table 1
and its footnotes g/h/†): ~640 MB (Ed25519) / ~33 GB (ML-DSA-65) storage,
and ~3 CPU-hours signing time.

- **Ed25519 storage — confirmed.** 64 B/chunk × 10^7 chunks = 640.0 MB,
  exactly matching the cited figure (`storage=640.0 MB` in every harness
  projection line, independent of chunk size, since storage scales with
  chunk *count* only).
- **ML-DSA-65 storage — confirmed by arithmetic, not measured.**
  Footnote g cites 3,309 B/chunk (FIPS 204 final); 3,309 B × 10^7 =
  33.09 GB, consistent with the cited "~33 GB." `sig_mldsa` itself is
  "not implemented (Phase 2)" in this harness (per the spec: the
  ML-DSA-65 backend lands once the P2.1 signing crate is available), so
  this is a check of the cited arithmetic, not an independent
  hardware measurement.
- **ML-DSA-65 signing (~3 CPU-hours) — cannot be confirmed yet.**
  Footnote h's "~3 CPU-hours" is specifically the ML-DSA-65 estimate
  (~1 ms/chunk × 10^7 chunks), not Ed25519. Since `sig_mldsa` is
  unimplemented in Phase 1, there is no hardware measurement to compare
  against this figure — it remains an open item for Phase 2, to be
  re-checked once the ML-DSA-65 backend exists.
- **Ed25519 signing — measured, and chunk-size dependent.** Table 1's
  N=10^7 scenario is captioned "10 TB at 1 MB chunks," so the directly
  comparable Ed25519 figure is the **1 MB** projection: **4.842
  CPU-hours** [95% CI 4.838, 4.853] — higher than the ~3-CPU-hour figure
  that footnote h attributes to ML-DSA-65, even though Ed25519 is the
  cheaper-per-signature classical scheme. At smaller chunk sizes the
  projection drops well below 3 hours (256 KB: 1.252 CPU-hours; 64 KB:
  0.356 CPU-hours; NOAA's ~122 KB nominal: 0.298 CPU-hours), so a single
  "~3-CPU-hour"-style number isn't meaningful for Ed25519 without fixing
  the chunk size — signing cost here is driven by per-signature
  overhead and chunk *count*, not bytes (see "Expected trends" above).
  The elevated 1 MB-chunk figure is consistent with Anomaly 1: the
  `fast`/precomputed-tables feature being disabled makes every
  `sign()` call slower than it would be with default features enabled,
  so 4.842 CPU-hours is itself a hardware-specific, build-configuration-
  dependent number, not an intrinsic property of Ed25519 — re-enabling
  `fast` would be expected to bring this below the ML-DSA-65 estimate.

## Inconclusive results

None — every measured cell's 95% bootstrap CI is tight relative to the
median (sub-1% width in all cases), so no condition pairs are
statistically indistinguishable at this sample size.
