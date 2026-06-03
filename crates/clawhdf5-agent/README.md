# clawhdf5-agent

[![crates.io](https://img.shields.io/crates/v/clawhdf5-agent.svg)](https://crates.io/crates/clawhdf5-agent)
[![docs.rs](https://img.shields.io/docsrs/clawhdf5-agent)](https://docs.rs/clawhdf5-agent)

HDF5-backed persistent memory store for on-device AI agents.

Built on [clawhdf5](https://crates.io/crates/clawhdf5), clawhdf5-agent provides a vector-searchable memory backend optimized for edge AI workloads. Store embeddings, text chunks, and metadata in a single HDF5 file with SIMD-accelerated similarity search.

## Features

- Persistent vector store in HDF5 format
- Cosine similarity and L2 distance search
- SIMD-accelerated via clawhdf5-accel (AVX2, NEON)
- Optional GPU acceleration via clawhdf5-gpu
- Memory-mapped access for large stores
- f16 storage support for compact embeddings

## Usage

```toml
[dependencies]
clawhdf5-agent = "2.1.0"
```

## License

MIT
