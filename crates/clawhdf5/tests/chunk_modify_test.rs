//! Test for reading, modifying, and writing back individual chunks.
//!
//! This test demonstrates:
//! 1. Opening an HDF5 file and iterating over every chunk of a dataset
//! 2. Reading raw chunk bytes
//! 3. Modifying one chunk in memory
//! 4. Writing it back to the file
//! 5. Re-opening and verifying the modification without corruption

use clawhdf5::{AttrValue, File, FileBuilder};
use clawhdf5_format::chunked_read::ChunkInfo;
use clawhdf5_format::data_layout::DataLayout;
use clawhdf5_format::dataspace::Dataspace;
use clawhdf5_format::datatype::Datatype;
use clawhdf5_format::filter_pipeline::FilterPipeline;
use clawhdf5_format::fixed_array::{read_fixed_array_chunks, FixedArrayHeader};
use clawhdf5_format::group_v2;
use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature;
use clawhdf5_format::superblock::Superblock;

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

/// Parse a dataset's layout information and extract chunk info
fn parse_dataset_and_chunks(
    file_data: &[u8],
    superblock: &Superblock,
    dataset_path: &str,
) -> Result<
    (
        Vec<ChunkInfo>,
        Dataspace,
        Datatype,
        Option<FilterPipeline>,
        usize, // element size
        usize, // chunk elements
    ),
    Box<dyn std::error::Error>,
> {
    // Resolve the dataset address
    let addr = group_v2::resolve_path_any(file_data, superblock, dataset_path)?;

    // Parse the object header
    let header = ObjectHeader::parse(
        file_data,
        addr as usize,
        superblock.offset_size,
        superblock.length_size,
    )?;

    // Extract layout message
    let layout_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DataLayout)
        .ok_or("No DataLayout message")?;
    let layout = DataLayout::parse(
        &layout_msg.data,
        superblock.offset_size,
        superblock.length_size,
    )?;

    // Extract dataspace message
    let dataspace_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::Dataspace)
        .ok_or("No Dataspace message")?;
    let dataspace = Dataspace::parse(&dataspace_msg.data, superblock.length_size)?;

    // Extract datatype message
    let datatype_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::Datatype)
        .ok_or("No Datatype message")?;
    let (datatype, _) = Datatype::parse(&datatype_msg.data)?;

    // Extract filter pipeline if present
    let pipeline = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .and_then(|msg| FilterPipeline::parse(&msg.data).ok());

    // Get element size
    let element_size = match &datatype {
        Datatype::FixedPoint { size, .. } => *size as usize,
        Datatype::FloatingPoint { size, .. } => *size as usize,
        _ => 8,
    };

    // Get chunk info based on layout type
    let rank = dataspace.dimensions.len();
    let chunks = match &layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            ..
        } => {
            let addr = btree_address.ok_or("No chunk address")?;
            // Spatial chunk dims exclude the element size dimension
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();

            match (*version, *chunk_index_type) {
                (4, Some(3)) => {
                    // Fixed Array index
                    let fa_header = FixedArrayHeader::parse(
                        file_data,
                        addr as usize,
                        superblock.offset_size,
                        superblock.length_size,
                    )?;
                    read_fixed_array_chunks(
                        file_data,
                        &fa_header,
                        &dataspace.dimensions,
                        &spatial_chunk_dims,
                        element_size as u32,
                        superblock.offset_size,
                        superblock.length_size,
                    )?
                }
                _ => {
                    return Err(format!(
                        "Unsupported chunk index: version={}, type={:?}",
                        version, chunk_index_type
                    )
                    .into())
                }
            }
        }
        _ => return Err("Dataset is not chunked".into()),
    };

    let chunk_elements: usize = chunks
        .first()
        .map(|c| {
            // Calculate from chunk offsets and dataset dims
            // For 1D, chunk size = chunk_dims[0]
            if let DataLayout::Chunked {
                chunk_dimensions, ..
            } = &layout
            {
                chunk_dimensions[..rank]
                    .iter()
                    .map(|&d| d as usize)
                    .product()
            } else {
                1024
            }
        })
        .unwrap_or(1024);

    Ok((
        chunks,
        dataspace,
        datatype,
        pipeline,
        element_size,
        chunk_elements,
    ))
}

#[test]
fn test_chunk_read_modify_write() -> Result<(), Box<dyn std::error::Error>> {
    let test_file = std::env::var("CLAWHDF5_TEST_FILE")
        .unwrap_or_else(|_| "/tmp/test_chunk_modify.h5".to_string());

    // Create a test file with known data
    create_test_file(&test_file)?;

    // Step 1: Open the file and read all chunk metadata
    println!("Step 1: Opening file and collecting chunk info...");
    let file_data = std::fs::read(&test_file)?;

    // Parse superblock
    let sig_offset = signature::find_signature(&file_data)?;
    let superblock = Superblock::parse(&file_data, sig_offset)?;

    // Parse dataset and get chunks
    let (chunks, dataspace, datatype, pipeline, element_size, chunk_elements) =
        parse_dataset_and_chunks(&file_data, &superblock, "test_data")?;

    println!("  Dataset shape: {:?}", dataspace.dimensions);
    println!("  Datatype: {:?}", datatype);
    println!("  Number of chunks: {}", chunks.len());
    println!(
        "  Chunk size: {} elements ({} bytes)",
        chunk_elements,
        chunk_elements * element_size
    );

    if pipeline.is_some() {
        println!("  Warning: Dataset has filters. Direct modification may corrupt data.");
        // Skip test for compressed data
        println!("  Skipping direct chunk modification test for compressed data.");
        std::fs::remove_file(&test_file)?;
        return Ok(());
    }

    // Print chunk addresses
    println!("\nChunk locations:");
    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "  Chunk {}: address=0x{:x}, size={} bytes, offsets={:?}",
            i, chunk.address, chunk.chunk_size, chunk.offsets
        );
    }

    // Read original data for later verification
    let file = File::open(&test_file)?;
    let dataset = file.dataset("test_data")?;
    let original_data = dataset.read_f64()?;
    drop(file);

    // Step 2: Iterate over every chunk and read raw bytes, compute checksums
    println!("\nStep 2: Reading all chunks and computing checksums...");
    let chunk_bytes = chunk_elements * element_size;
    let mut chunk_checksums: Vec<u64> = Vec::with_capacity(chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        let start = chunk.address as usize;
        let end = start + chunk.chunk_size as usize;

        if end > file_data.len() {
            println!("  Chunk {} would exceed file bounds", i);
            continue;
        }

        let chunk_data = &file_data[start..end];
        let checksum: u64 = chunk_data.iter().map(|&b| b as u64).sum();
        chunk_checksums.push(checksum);

        if i < 3 || i == chunks.len() - 1 {
            println!("  Chunk {}: checksum={}", i, checksum);
        } else if i == 3 {
            println!("  ... ({} more chunks) ...", chunks.len() - 4);
        }
    }

    // Step 3: Modify one chunk in memory (chunk 1, the second chunk)
    let target_chunk_idx = 1;
    if target_chunk_idx >= chunks.len() {
        println!("Not enough chunks to modify, skipping modification test");
        std::fs::remove_file(&test_file)?;
        return Ok(());
    }

    println!("\nStep 3: Modifying chunk {} in memory...", target_chunk_idx);
    let target_chunk = &chunks[target_chunk_idx];
    let chunk_start = target_chunk.address as usize;
    let chunk_end = chunk_start + target_chunk.chunk_size as usize;

    let mut modified_chunk: Vec<u8> = file_data[chunk_start..chunk_end].to_vec();

    // Read original first value (f64 little-endian)
    let original_first_value = f64::from_le_bytes(modified_chunk[0..8].try_into()?);
    println!("  Original first value in chunk: {}", original_first_value);

    // Modify: set all values in this chunk to 999.0
    let new_value: f64 = 999.0;
    let new_bytes = new_value.to_le_bytes();
    let values_in_chunk = modified_chunk.len() / 8;
    for i in 0..values_in_chunk {
        modified_chunk[i * 8..(i + 1) * 8].copy_from_slice(&new_bytes);
    }
    println!(
        "  Modified all {} values in chunk to {}",
        values_in_chunk, new_value
    );

    // Step 4: Write the modified chunk back to the file
    println!("\nStep 4: Writing modified chunk back to file...");
    {
        let mut file_handle = OpenOptions::new().write(true).open(&test_file)?;
        file_handle.seek(SeekFrom::Start(target_chunk.address))?;
        file_handle.write_all(&modified_chunk)?;
        file_handle.sync_all()?;
    }
    println!(
        "  Wrote {} bytes at offset 0x{:x}",
        modified_chunk.len(),
        target_chunk.address
    );

    // Step 5: Re-open the file and verify
    println!("\nStep 5: Re-opening file and verifying...");
    let file_data_after = std::fs::read(&test_file)?;
    let file_after = File::open(&test_file)?;
    let dataset_after = file_after.dataset("test_data")?;
    let data_after = dataset_after.read_f64()?;

    // Verify the modified chunk
    // Calculate which elements belong to the modified chunk based on its offset
    let chunk_offset = target_chunk.offsets[0] as usize; // 1D dataset, first offset is element index
    let chunk_start_element = chunk_offset;
    let chunk_end_element =
        std::cmp::min(chunk_start_element + chunk_elements, data_after.len());

    println!(
        "  Checking modified chunk (elements {}..{})...",
        chunk_start_element, chunk_end_element
    );

    let mut modified_count = 0;
    for i in chunk_start_element..chunk_end_element {
        if (data_after[i] - new_value).abs() < 1e-10 {
            modified_count += 1;
        }
    }
    println!(
        "  {} of {} elements correctly set to {}",
        modified_count,
        chunk_end_element - chunk_start_element,
        new_value
    );
    assert_eq!(
        modified_count,
        chunk_end_element - chunk_start_element,
        "Not all elements in chunk were modified"
    );

    // Step 6: Verify adjacent chunks are NOT corrupted
    println!("\nStep 6: Verifying adjacent chunks are not corrupted...");

    // Check chunk 0 (before modified chunk)
    let chunk0_offset = chunks[0].offsets[0] as usize;
    let chunk0_end = std::cmp::min(chunk0_offset + chunk_elements, data_after.len());
    let mut chunk0_matches = 0;
    for i in chunk0_offset..chunk0_end {
        if (data_after[i] - original_data[i]).abs() < 1e-10 {
            chunk0_matches += 1;
        }
    }
    println!(
        "  Chunk 0: {} of {} elements unchanged",
        chunk0_matches,
        chunk0_end - chunk0_offset
    );
    assert_eq!(
        chunk0_matches,
        chunk0_end - chunk0_offset,
        "Chunk 0 was corrupted!"
    );

    // Check chunk 2 (after modified chunk) if it exists
    if chunks.len() > 2 {
        let chunk2_offset = chunks[2].offsets[0] as usize;
        let chunk2_end = std::cmp::min(chunk2_offset + chunk_elements, data_after.len());
        let mut chunk2_matches = 0;
        for i in chunk2_offset..chunk2_end {
            if (data_after[i] - original_data[i]).abs() < 1e-10 {
                chunk2_matches += 1;
            }
        }
        println!(
            "  Chunk 2: {} of {} elements unchanged",
            chunk2_matches,
            chunk2_end - chunk2_offset
        );
        assert_eq!(
            chunk2_matches,
            chunk2_end - chunk2_offset,
            "Chunk 2 was corrupted!"
        );
    }

    // Step 7: Verify all other chunks via raw byte checksum
    println!("\nStep 7: Verifying all chunk checksums...");
    let mut checksum_errors = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        if i == target_chunk_idx {
            continue; // Skip the modified chunk
        }

        let start = chunk.address as usize;
        let end = start + chunk.chunk_size as usize;

        if end > file_data_after.len() {
            continue;
        }

        let chunk_data = &file_data_after[start..end];
        let checksum: u64 = chunk_data.iter().map(|&b| b as u64).sum();

        if checksum != chunk_checksums[i] {
            println!(
                "  ERROR: Chunk {} checksum mismatch: expected {}, got {}",
                i, chunk_checksums[i], checksum
            );
            checksum_errors += 1;
        }
    }

    if checksum_errors == 0 {
        println!(
            "  All {} unmodified chunks have correct checksums",
            chunks.len() - 1
        );
    }
    assert_eq!(checksum_errors, 0, "Some chunks were corrupted!");

    println!("\n=== TEST PASSED ===");
    println!(
        "Successfully modified chunk {} without corrupting adjacent chunks",
        target_chunk_idx
    );

    // Cleanup
    std::fs::remove_file(&test_file)?;

    Ok(())
}

/// Create a small test file with known data
fn create_test_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create a dataset with 10 chunks of 1024 elements each (80 KB total)
    let elements_per_chunk = 1024;
    let num_chunks = 10;
    let total_elements = elements_per_chunk * num_chunks;

    // Generate known data: each element is its index as f64
    let data: Vec<f64> = (0..total_elements).map(|i| i as f64).collect();

    let mut builder = FileBuilder::new();
    builder.set_attr("test", AttrValue::String("chunk_modify_test".into()));

    builder
        .create_dataset("test_data")
        .with_f64_data(&data)
        .with_shape(&[total_elements as u64])
        .with_chunks(&[elements_per_chunk as u64])
        .set_attr(
            "description",
            AttrValue::String("Test dataset for chunk modification".into()),
        );

    builder.write(path)?;

    println!("Created test file: {}", path);
    println!(
        "  {} elements in {} chunks of {} elements each",
        total_elements, num_chunks, elements_per_chunk
    );

    Ok(())
}

#[test]
fn test_chunk_iteration_large_file() -> Result<(), Box<dyn std::error::Error>> {
    // This test can be run against the 10GB file if available
    let large_file = std::env::var("CLAWHDF5_LARGE_TEST_FILE").ok();

    if let Some(path) = large_file {
        println!("Testing with large file: {}", path);

        let file_data = std::fs::read(&path)?;
        let sig_offset = signature::find_signature(&file_data)?;
        let superblock = Superblock::parse(&file_data, sig_offset)?;

        let (chunks, dataspace, datatype, _, element_size, chunk_elements) =
            parse_dataset_and_chunks(&file_data, &superblock, "dataset_1mb")?;

        println!("Dataset shape: {:?}", dataspace.dimensions);
        println!("Number of chunks: {}", chunks.len());
        println!(
            "Chunk size: {} elements ({} MB)",
            chunk_elements,
            chunk_elements * element_size / 1024 / 1024
        );

        assert!(chunks.len() > 0);
        println!(
            "Successfully iterated chunk metadata for {} chunks",
            chunks.len()
        );
    } else {
        println!("Skipping large file test (set CLAWHDF5_LARGE_TEST_FILE to enable)");
    }

    Ok(())
}
