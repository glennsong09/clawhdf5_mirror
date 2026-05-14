//! Disk I/O operations for HDF5 memory files.
//!
//! Uses memory-mapped I/O via `clawhdf5_io::MmapReader` for efficient
//! file reading with OS-managed paging.

use std::path::Path;

use crate::MemoryConfig;
use crate::MemoryError;
use crate::cache::MemoryCache;
use crate::knowledge::KnowledgeCache;
use crate::schema;
use crate::session::SessionCache;

/// Write all in-memory state to an HDF5 file on disk.
pub fn write_to_disk(
    path: &Path,
    config: &MemoryConfig,
    cache: &MemoryCache,
    sessions: &SessionCache,
    knowledge: &KnowledgeCache,
) -> Result<(), MemoryError> {
    let bytes = schema::build_hdf5_file(config, cache, sessions, knowledge)?;

    if bytes.is_empty() {
        return Err(MemoryError::Hdf5("build_hdf5_file produced 0 bytes".into()));
    }

    // Write to a temp file first, then rename for atomicity
    let tmp_path = path.with_extension("h5.tmp");
    std::fs::write(&tmp_path, &bytes).map_err(MemoryError::Io)?;
    std::fs::rename(&tmp_path, path).map_err(MemoryError::Io)?;

    Ok(())
}

/// Read an HDF5 file and return all state.
///
/// Uses memory-mapped I/O via `clawhdf5_io::MmapReader` for efficient
/// file access. The OS pages in data on demand rather than reading the
/// entire file into a contiguous buffer upfront.
pub fn read_from_disk(
    path: &Path,
) -> Result<(MemoryConfig, MemoryCache, SessionCache, KnowledgeCache), MemoryError> {
    let mmap = clawhdf5_io::MmapReader::open(path).map_err(MemoryError::Io)?;

    // Advise the OS we'll need the whole file for parsing
    mmap.advise_willneed(0, mmap.len());

    // Parse the HDF5 file from the mmap'd bytes
    let file = clawhdf5::File::from_bytes(mmap.as_bytes().to_vec())
        .map_err(|e| MemoryError::Hdf5(format!("cannot open {}: {e}", path.display())))?;

    let (mut config, cache, sessions, knowledge) = schema::validate_and_load(&file)?;
    config.path = path.to_path_buf();

    Ok((config, cache, sessions, knowledge))
}

/// Copy an HDF5 file atomically to a destination.
pub fn snapshot_file(src: &Path, dest: &Path) -> Result<std::path::PathBuf, MemoryError> {
    let dest_file = if dest.is_dir() {
        let filename = src.file_name().ok_or_else(|| {
            MemoryError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "source has no filename",
            ))
        })?;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        dest.join(format!("snapshot_{ts}_{}", filename.to_string_lossy()))
    } else {
        dest.to_path_buf()
    };

    // Atomic copy: write to temp, then rename
    let tmp_path = dest_file.with_extension("h5.tmp");
    std::fs::copy(src, &tmp_path).map_err(MemoryError::Io)?;
    std::fs::rename(&tmp_path, &dest_file).map_err(MemoryError::Io)?;

    Ok(dest_file)
}
