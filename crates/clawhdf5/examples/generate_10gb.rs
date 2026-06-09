//! Generate a synthetic HDF5 file with three datasets using different chunk sizes.
//!
//! Run with: cargo run --example generate_10gb --release [SIZE_GB] [OUTPUT_PATH]
//!
//! Examples:
//!   cargo run --example generate_10gb --release              # 3 GB file (default)
//!   cargo run --example generate_10gb --release 10           # 10 GB file
//!   cargo run --example generate_10gb --release 10 out.h5    # 10 GB to out.h5
//!
//! This creates a file with:
//! - dataset_64kb:  with 64 KB chunks  (8,192 f64 elements per chunk)
//! - dataset_256kb: with 256 KB chunks (32,768 f64 elements per chunk)
//! - dataset_1mb:   with 1 MB chunks   (131,072 f64 elements per chunk)

use clawhdf5::{AttrValue, FileBuilder};
use std::time::Instant;

fn get_elements_per_dataset(target_gb: f64) -> usize {
    // Each dataset gets 1/3 of total size
    // f64 = 8 bytes
    let bytes_per_dataset = (target_gb * 1_073_741_824.0 / 3.0) as usize;
    let elements = bytes_per_dataset / 8;
    // Round down to nearest multiple of 131,072 (1 MB chunk size) for clean division
    (elements / 131_072) * 131_072
}

// Chunk sizes in f64 elements
const CHUNK_64KB: u64 = 8_192;      // 64 KB / 8 bytes = 8,192 elements
const CHUNK_256KB: u64 = 32_768;    // 256 KB / 8 bytes = 32,768 elements
const CHUNK_1MB: u64 = 131_072;     // 1 MB / 8 bytes = 131,072 elements

fn generate_data(size: usize, seed: u64) -> Vec<f64> {
    println!("  Generating {} elements ({:.2} GB)...",
        size,
        (size * 8) as f64 / 1_073_741_824.0
    );

    // Simple deterministic data generation
    (0..size)
        .map(|i| ((i as u64).wrapping_add(seed) as f64 * 0.001).sin())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Parse size (default 3 GB for safety, use 10 for full 10 GB)
    let target_gb: f64 = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);

    let output_path = args.get(2)
        .cloned()
        .unwrap_or_else(|| format!("synthetic_{}gb.h5", target_gb as u32));

    let elements_per_dataset = get_elements_per_dataset(target_gb);

    println!("Generating {:.1} GB HDF5 file: {}", target_gb, output_path);
    println!("Each dataset: {} elements ({:.2} GB)",
        elements_per_dataset,
        (elements_per_dataset * 8) as f64 / 1_073_741_824.0
    );
    println!();

    let total_start = Instant::now();
    let mut builder = FileBuilder::new();

    // Set file-level attributes
    builder.set_attr("generator", AttrValue::String("clawhdf5 generate_10gb example".into()));
    builder.set_attr("total_size_bytes", AttrValue::I64((elements_per_dataset * 3 * 8) as i64));

    // Dataset 1: 64 KB chunks
    println!("Creating dataset_64kb (64 KB chunks = {} elements per chunk)...", CHUNK_64KB);
    let start = Instant::now();
    let data1 = generate_data(elements_per_dataset, 1);
    println!("  Data generation took {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    builder.create_dataset("dataset_64kb")
        .with_f64_data(&data1)
        .with_shape(&[elements_per_dataset as u64])
        .with_chunks(&[CHUNK_64KB])
        .set_attr("chunk_size_bytes", AttrValue::I64(64 * 1024))
        .set_attr("chunk_size_elements", AttrValue::I64(CHUNK_64KB as i64))
        .set_attr("description", AttrValue::String("Dataset with 64 KB chunks".into()));
    println!("  Dataset builder configuration took {:.2}s", start.elapsed().as_secs_f64());
    drop(data1); // Free memory before next dataset
    println!();

    // Dataset 2: 256 KB chunks
    println!("Creating dataset_256kb (256 KB chunks = {} elements per chunk)...", CHUNK_256KB);
    let start = Instant::now();
    let data2 = generate_data(elements_per_dataset, 2);
    println!("  Data generation took {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    builder.create_dataset("dataset_256kb")
        .with_f64_data(&data2)
        .with_shape(&[elements_per_dataset as u64])
        .with_chunks(&[CHUNK_256KB])
        .set_attr("chunk_size_bytes", AttrValue::I64(256 * 1024))
        .set_attr("chunk_size_elements", AttrValue::I64(CHUNK_256KB as i64))
        .set_attr("description", AttrValue::String("Dataset with 256 KB chunks".into()));
    println!("  Dataset builder configuration took {:.2}s", start.elapsed().as_secs_f64());
    drop(data2);
    println!();

    // Dataset 3: 1 MB chunks
    println!("Creating dataset_1mb (1 MB chunks = {} elements per chunk)...", CHUNK_1MB);
    let start = Instant::now();
    let data3 = generate_data(elements_per_dataset, 3);
    println!("  Data generation took {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    builder.create_dataset("dataset_1mb")
        .with_f64_data(&data3)
        .with_shape(&[elements_per_dataset as u64])
        .with_chunks(&[CHUNK_1MB])
        .set_attr("chunk_size_bytes", AttrValue::I64(1024 * 1024))
        .set_attr("chunk_size_elements", AttrValue::I64(CHUNK_1MB as i64))
        .set_attr("description", AttrValue::String("Dataset with 1 MB chunks".into()));
    println!("  Dataset builder configuration took {:.2}s", start.elapsed().as_secs_f64());
    drop(data3);
    println!();

    // Write to file
    println!("Writing HDF5 file to disk...");
    let start = Instant::now();
    builder.write(&output_path)?;
    println!("  File write took {:.2}s", start.elapsed().as_secs_f64());

    // Print summary
    let metadata = std::fs::metadata(&output_path)?;
    println!();
    println!("=== Summary ===");
    println!("Output file: {}", output_path);
    println!("File size: {:.2} GB ({} bytes)",
        metadata.len() as f64 / 1_073_741_824.0,
        metadata.len()
    );
    println!("Total time: {:.2}s", total_start.elapsed().as_secs_f64());
    println!();
    println!("Datasets:");
    println!("  - dataset_64kb:  {} elements, 64 KB chunks ({} chunks)",
        elements_per_dataset,
        elements_per_dataset / CHUNK_64KB as usize
    );
    println!("  - dataset_256kb: {} elements, 256 KB chunks ({} chunks)",
        elements_per_dataset,
        elements_per_dataset / CHUNK_256KB as usize
    );
    println!("  - dataset_1mb:   {} elements, 1 MB chunks ({} chunks)",
        elements_per_dataset,
        elements_per_dataset / CHUNK_1MB as usize
    );

    Ok(())
}
