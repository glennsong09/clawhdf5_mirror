# clawhdf5-netcdf4

[![crates.io](https://img.shields.io/crates/v/clawhdf5-netcdf4.svg)](https://crates.io/crates/clawhdf5-netcdf4)
[![docs.rs](https://docs.rs/clawhdf5-netcdf4/badge.svg)](https://docs.rs/clawhdf5-netcdf4)

NetCDF-4 read support built on clawhdf5 — pure Rust, no C dependencies.

## Features

- Read NetCDF-4 / HDF5-backed `.nc` files
- Dimension, variable, and CF convention support
- Climate and scientific data access

## Usage

```rust
use clawhdf5_netcdf4::NetCDF4File;

let nc = NetCDF4File::open("climate.nc").unwrap();
let temp = nc.variable("temperature").unwrap();
```

## License

MIT
