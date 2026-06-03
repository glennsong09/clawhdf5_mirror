# clawhdf5-filters

[![crates.io](https://img.shields.io/crates/v/clawhdf5-filters.svg)](https://crates.io/crates/clawhdf5-filters)
[![docs.rs](https://docs.rs/clawhdf5-filters/badge.svg)](https://docs.rs/clawhdf5-filters)

Filter and compression pipeline for clawhdf5.

## Features

- DEFLATE compression/decompression
- Fast deflate via zlib-ng (`fast-deflate` feature)
- Apple Compression framework support (`apple-compression` feature)

## Usage

```rust
use clawhdf5_filters::{deflate_decode, deflate_encode};

let compressed = deflate_encode(&data, 6).unwrap();
let decompressed = deflate_decode(&compressed).unwrap();
```

## License

MIT
