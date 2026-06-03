# clawhdf5-derive

[![crates.io](https://img.shields.io/crates/v/clawhdf5-derive.svg)](https://crates.io/crates/clawhdf5-derive)
[![docs.rs](https://docs.rs/clawhdf5-derive/badge.svg)](https://docs.rs/clawhdf5-derive)

Derive macros for clawhdf5 HDF5 traits.

## Features

- `#[derive(HDF5Type)]` for automatic HDF5 datatype mapping
- Struct-to-compound-type derivation

## Usage

```rust
use clawhdf5_derive::HDF5Type;

#[derive(HDF5Type)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}
```

## License

MIT
