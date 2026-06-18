# P1.2b Capability Matrix

Representative cell: chunk_size=256 KB, source=synthetic. Time metrics are median over 30 measured trials (after 5 discarded warmups) with a 95% bootstrap confidence interval in brackets (2000 resamples). Deterministic, non-time metrics (metadata_bytes, public_verification, subset_proof_bytes) show the single measured value with no CI.

| metric | flat (SHA-256) | mac (HMAC-SHA-256) | sig_ed25519 | sig_mldsa (ML-DSA-65) |
|---|---|---|---|---|
| commit_time | 1274163066 ns [95% CI 1273715271, 1275964332] | 1288039906 ns [95% CI 1283609277, 1291618018] | 6154455578 ns [95% CI 6146728459, 6166237208] | not implemented (Phase 2) |
| verify_chunk_latency | N/A — no partial access | 94312 ns [95% CI 94112, 95254] | 240042 ns [95% CI 239702, 240674] | not implemented (Phase 2) |
| verify_dataset_latency | 1273973669 ns [95% CI 1273544905, 1274380215] | 1284980733 ns [95% CI 1283742709, 1290710292] | 3250039182 ns [95% CI 3247194546, 3254915246] | not implemented (Phase 2) |
| append_time | 1274006000 ns [95% CI 1273871340, 1274627191] | 116610 ns [95% CI 115568, 118424] | 453139 ns [95% CI 452733, 454558] | not implemented (Phase 2) |
| update_time | 1274432308 ns [95% CI 1273894184, 1275930944] | 93972 ns [95% CI 93281, 95094] | 449788 ns [95% CI 449247, 450464] | not implemented (Phase 2) |
| metadata_bytes | 32 bytes | 436864 bytes | 873728 bytes | not implemented (Phase 2) |
| public_verification | Yes | N/A — no public key | Yes | not implemented (Phase 2) |
| subset_proof_bytes | N/A | N/A | 87360 bytes | not implemented (Phase 2) |
