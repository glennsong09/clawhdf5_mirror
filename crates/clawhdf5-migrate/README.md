# clawhdf5-migrate

[![crates.io](https://img.shields.io/crates/v/clawhdf5-migrate.svg)](https://crates.io/crates/clawhdf5-migrate)
[![docs.rs](https://img.shields.io/docsrs/clawhdf5-migrate)](https://docs.rs/clawhdf5-migrate)

CLI tool to migrate SQLite agent memory databases to HDF5 format.

Converts existing SQLite-based agent memory stores (embeddings, text chunks, metadata) into the HDF5 format used by [clawhdf5-agent](https://crates.io/crates/clawhdf5-agent).

## Installation

```bash
cargo install clawhdf5-migrate
```

## Usage

```bash
clawhdf5-migrate --input agent.db --output agent.h5
```

## License

MIT
