# edgehdf5-migrate

[![crates.io](https://img.shields.io/crates/v/edgehdf5-migrate.svg)](https://crates.io/crates/edgehdf5-migrate)
[![docs.rs](https://img.shields.io/docsrs/edgehdf5-migrate)](https://docs.rs/edgehdf5-migrate)

CLI tool to migrate SQLite agent memory databases to HDF5 format.

Converts existing SQLite-based agent memory stores (embeddings, text chunks, metadata) into the HDF5 format used by [edgehdf5-memory](https://crates.io/crates/edgehdf5-memory).

## Installation

```bash
cargo install edgehdf5-migrate
```

## Usage

```bash
edgehdf5-migrate --input agent.db --output agent.h5
```

## License

MIT
