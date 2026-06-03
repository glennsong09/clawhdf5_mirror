# clawhdf5-py

[![crates.io](https://img.shields.io/crates/v/clawhdf5-py.svg)](https://crates.io/crates/clawhdf5-py)
[![docs.rs](https://docs.rs/clawhdf5-py/badge.svg)](https://docs.rs/clawhdf5-py)

Python bindings for clawhdf5 — a pure-Rust HDF5 library.

## Features

- h5py-compatible API (`File`, `Group`, `Dataset`)
- NumPy array integration
- Read and write HDF5 files from Python with no C dependencies

## Usage

```python
import clawhdf5

with clawhdf5.File('data.h5', 'r') as f:
    data = f['/dataset'][:]
```

## License

MIT
