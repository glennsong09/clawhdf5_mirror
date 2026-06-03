# clawhdf5-accel

[![crates.io](https://img.shields.io/crates/v/clawhdf5-accel.svg)](https://crates.io/crates/clawhdf5-accel)
[![docs.rs](https://docs.rs/clawhdf5-accel/badge.svg)](https://docs.rs/clawhdf5-accel)

SIMD-accelerated operations for clawhdf5.

## Features

- AVX2 and NEON SIMD acceleration
- AVX-512 support (`avx512` feature)
- Float16 conversion (`float16` feature)
- CRC32 checksum acceleration

## Usage

```rust
use clawhdf5_accel::checksum::crc32_simd;

let crc = crc32_simd(&data);
```

## License

MIT
