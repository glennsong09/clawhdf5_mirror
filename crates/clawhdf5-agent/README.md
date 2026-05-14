# edgehdf5-memory

[![crates.io](https://img.shields.io/crates/v/edgehdf5-memory.svg)](https://crates.io/crates/edgehdf5-memory)
[![docs.rs](https://img.shields.io/docsrs/edgehdf5-memory)](https://docs.rs/edgehdf5-memory)

HDF5-backed persistent memory store for on-device AI agents.

Built on [rustyhdf5](https://crates.io/crates/rustyhdf5), edgehdf5-memory provides a vector-searchable memory backend optimized for edge AI workloads. Store embeddings, text chunks, and metadata in a single HDF5 file with SIMD-accelerated similarity search.

## Features

- Persistent vector store in HDF5 format
- Cosine similarity and L2 distance search
- SIMD-accelerated via rustyhdf5-accel (AVX2, NEON)
- Optional GPU acceleration via rustyhdf5-gpu
- Memory-mapped access for large stores
- f16 storage support for compact embeddings

## Usage

```toml
[dependencies]
edgehdf5-memory = "1.93"
```

## License

MIT
