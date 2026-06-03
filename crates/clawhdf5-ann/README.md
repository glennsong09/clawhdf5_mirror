# clawhdf5-ann

[![crates.io](https://img.shields.io/crates/v/clawhdf5-ann.svg)](https://crates.io/crates/clawhdf5-ann)
[![docs.rs](https://docs.rs/clawhdf5-ann/badge.svg)](https://docs.rs/clawhdf5-ann)

HNSW approximate nearest neighbor index stored as HDF5.

## Features

- Build and query HNSW indexes persisted in HDF5 format
- Pure Rust, no C dependencies
- Efficient similarity search for high-dimensional vectors

## Usage

```rust
use clawhdf5_ann::HnswIndex;

let index = HnswIndex::from_hdf5("vectors.h5").unwrap();
let neighbors = index.search(&query, 10);
```

## License

MIT
