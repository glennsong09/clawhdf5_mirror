# clawhdf5-io

[![crates.io](https://img.shields.io/crates/v/clawhdf5-io.svg)](https://crates.io/crates/clawhdf5-io)
[![docs.rs](https://docs.rs/clawhdf5-io/badge.svg)](https://docs.rs/clawhdf5-io)

I/O abstraction layer for clawhdf5.

## Features

- Memory-mapped file access (`mmap` feature)
- Async I/O via Tokio (`async` feature)
- HSDS remote access (`hsds` feature)
- Prefetching and sweep optimizations

## Usage

```rust
use clawhdf5_io::MmapReader;

let reader = MmapReader::open("data.h5").unwrap();
```

## License

MIT
