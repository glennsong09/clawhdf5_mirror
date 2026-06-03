# clawhdf5-gpu

[![crates.io](https://img.shields.io/crates/v/clawhdf5-gpu.svg)](https://crates.io/crates/clawhdf5-gpu)
[![docs.rs](https://docs.rs/clawhdf5-gpu/badge.svg)](https://docs.rs/clawhdf5-gpu)

GPU-accelerated vector operations for clawhdf5 using wgpu compute shaders.

## Features

- GPU-accelerated distance computations (L2, cosine)
- wgpu-based compute shaders for cross-platform GPU support
- Float16 support via `half` crate

## Usage

```rust
use clawhdf5_gpu::GpuAccelerator;

let accel = GpuAccelerator::new().unwrap();
let distances = accel.l2_distances(&query, &vectors).unwrap();
```

## License

MIT
