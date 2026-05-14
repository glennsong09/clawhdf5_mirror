//! Android JNI bridge for clawhdf5-agent HDF5 backend.
//!
//! Exposes `extern "C"` functions for use via JNI from Kotlin.
//! Each HDF5Memory instance is managed via an opaque handle (pointer).
//!
//! Thread safety: the caller (Kotlin side) must synchronize access
//! to a single handle. Multiple handles are independent.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;

use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};

// ---------------------------------------------------------------------------
// Handle management
// ---------------------------------------------------------------------------

/// Opaque handle to an HDF5Memory instance.
type Handle = *mut HDF5Memory;

/// Create a new HDF5 memory file.
///
/// Returns a handle on success, null on failure.
///
/// # Safety
///
/// `path` and `agent_id` must be valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_create(
    path: *const c_char,
    agent_id: *const c_char,
    embedding_dim: u32,
) -> Handle {
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let path = match unsafe { cstr_to_string(path) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let agent_id = match unsafe { cstr_to_string(agent_id) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    let config = MemoryConfig::new(PathBuf::from(path), &agent_id, embedding_dim as usize);
    match HDF5Memory::create(config) {
        Ok(mem) => Box::into_raw(Box::new(mem)),
        Err(_) => ptr::null_mut(),
    }
}

/// Open an existing HDF5 memory file.
///
/// Returns a handle on success, null on failure.
///
/// # Safety
///
/// `path` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_open(path: *const c_char) -> Handle {
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let path = match unsafe { cstr_to_string(path) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    match HDF5Memory::open(std::path::Path::new(&path)) {
        Ok(mem) => Box::into_raw(Box::new(mem)),
        Err(_) => ptr::null_mut(),
    }
}

/// Close and free an HDF5Memory handle.
///
/// # Safety
///
/// `handle` must be a handle previously returned by [`edgehdf5_create`] or
/// [`edgehdf5_open`], and must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_close(handle: Handle) {
    if !handle.is_null() {
        // SAFETY: handle was created by Box::into_raw in edgehdf5_create; this is the final use.
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ---------------------------------------------------------------------------
// Memory operations
// ---------------------------------------------------------------------------

/// Save a memory entry. Returns the entry index, or -1 on failure.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - All `*const c_char` arguments must be valid, null-terminated C strings.
/// - `embedding_ptr` must point to at least `embedding_len` contiguous `f32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_save(
    handle: Handle,
    chunk: *const c_char,
    embedding_ptr: *const f32,
    embedding_len: u32,
    source_channel: *const c_char,
    timestamp: f64,
    session_id: *const c_char,
    tags: *const c_char,
) -> i64 {
    // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
    let mem = match unsafe { handle.as_mut() } {
        Some(m) => m,
        None => return -1,
    };

    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let chunk = match unsafe { cstr_to_string(chunk) } {
        Some(s) => s,
        None => return -1,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let source_channel = match unsafe { cstr_to_string(source_channel) } {
        Some(s) => s,
        None => return -1,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let session_id = match unsafe { cstr_to_string(session_id) } {
        Some(s) => s,
        None => return -1,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let tags = match unsafe { cstr_to_string(tags) } {
        Some(s) => s,
        None => return -1,
    };

    let embedding =
        // SAFETY: JNI caller guarantees embedding_ptr points to embedding_len valid f32 values.
        unsafe { std::slice::from_raw_parts(embedding_ptr, embedding_len as usize) }.to_vec();

    let entry = MemoryEntry {
        chunk,
        embedding,
        source_channel,
        timestamp,
        session_id,
        tags,
    };

    match mem.save(entry) {
        Ok(idx) => idx as i64,
        Err(_) => -1,
    }
}

/// Get the number of active (non-deleted) entries.
///
/// # Safety
///
/// `handle` must be a valid handle or null (returns 0 if null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_count_active(handle: Handle) -> u64 {
    // SAFETY: handle is a valid non-null Handle from edgehdf5_create.
    match unsafe { handle.as_ref() } {
        Some(mem) => mem.count_active() as u64,
        None => 0,
    }
}

/// Get the total number of entries (including tombstoned).
///
/// # Safety
///
/// `handle` must be a valid handle or null (returns 0 if null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_count(handle: Handle) -> u64 {
    // SAFETY: handle is a valid non-null Handle from edgehdf5_create.
    match unsafe { handle.as_ref() } {
        Some(mem) => mem.count() as u64,
        None => 0,
    }
}

/// Delete a memory entry by index. Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// `handle` must be a valid, non-null handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_delete(handle: Handle, index: u64) -> i32 {
    // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
    let mem = match unsafe { handle.as_mut() } {
        Some(m) => m,
        None => return -1,
    };

    match mem.delete(index as usize) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ---------------------------------------------------------------------------
// Hybrid search
// ---------------------------------------------------------------------------

/// Result buffer for hybrid search. Caller allocates arrays.
///
/// Performs hybrid search and writes up to `max_results` entries into the
/// provided output arrays. Returns the number of results written.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - `query_text` must be a valid, null-terminated C string.
/// - `query_embedding_ptr` must point to at least `query_embedding_len` `f32` values.
/// - `out_indices` and `out_scores` must point to arrays of at least `max_results` elements.
/// - `out_chunks` must be null or point to an array of at least `max_results` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_hybrid_search(
    handle: Handle,
    query_embedding_ptr: *const f32,
    query_embedding_len: u32,
    query_text: *const c_char,
    vector_weight: f32,
    keyword_weight: f32,
    max_results: u32,
    out_indices: *mut u64,
    out_scores: *mut f32,
    out_chunks: *mut *mut c_char,
) -> u32 {
    // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
    let mem = match unsafe { handle.as_mut() } {
        Some(m) => m,
        None => return 0,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let query_text = match unsafe { cstr_to_string(query_text) } {
        Some(s) => s,
        None => return 0,
    };
    let query_embedding =
        // SAFETY: JNI caller guarantees query_embedding_ptr points to query_embedding_len valid f32 values.
        unsafe { std::slice::from_raw_parts(query_embedding_ptr, query_embedding_len as usize) };

    let results = mem.hybrid_search(
        query_embedding,
        &query_text,
        vector_weight,
        keyword_weight,
        max_results as usize,
    );

    let count = results.len().min(max_results as usize);
    for (i, result) in results.iter().take(count).enumerate() {
        // SAFETY: The CString result pointers are valid Rust-owned allocations from CString::into_raw.
        unsafe {
            *out_indices.add(i) = result.index as u64;
            *out_scores.add(i) = result.score;
            if !out_chunks.is_null() {
                match CString::new(result.chunk.as_str()) {
                    Ok(cs) => *out_chunks.add(i) = cs.into_raw(),
                    Err(_) => *out_chunks.add(i) = ptr::null_mut(),
                }
            }
        }
    }

    count as u32
}

/// Free a chunk string returned by hybrid search.
///
/// # Safety
///
/// `s` must be a pointer previously returned by [`edgehdf5_hybrid_search`]
/// via `out_chunks`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_free_string(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: s was created by CString::into_raw in this module; this is the final use.
        unsafe { drop(CString::from_raw(s)) };
    }
}

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

/// Add a session entry. Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - All `*const c_char` arguments must be valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_add_session(
    handle: Handle,
    id: *const c_char,
    start_idx: u64,
    end_idx: u64,
    channel: *const c_char,
    summary: *const c_char,
) -> i32 {
    // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
    let mem = match unsafe { handle.as_mut() } {
        Some(m) => m,
        None => return -1,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let id = match unsafe { cstr_to_string(id) } {
        Some(s) => s,
        None => return -1,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let channel = match unsafe { cstr_to_string(channel) } {
        Some(s) => s,
        None => return -1,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let summary = match unsafe { cstr_to_string(summary) } {
        Some(s) => s,
        None => return -1,
    };

    match mem.add_session(
        &id,
        start_idx as usize,
        end_idx as usize,
        &channel,
        &summary,
    ) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Get a session summary by ID. Returns a C string (caller must free with
/// `edgehdf5_free_string`), or null if not found.
///
/// # Safety
///
/// - `handle` must be a valid handle or null.
/// - `session_id` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_get_session_summary(
    handle: Handle,
    session_id: *const c_char,
) -> *mut c_char {
    // SAFETY: handle is a valid non-null Handle from edgehdf5_create.
    let mem = match unsafe { handle.as_ref() } {
        Some(m) => m,
        None => return ptr::null_mut(),
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let session_id = match unsafe { cstr_to_string(session_id) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    match mem.get_session_summary(&session_id) {
        Ok(Some(summary)) => match CString::new(summary) {
            Ok(cs) => cs.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        _ => ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Knowledge graph
// ---------------------------------------------------------------------------

/// Add a knowledge graph entity. Returns entity ID, or -1 on failure.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - `name` and `entity_type` must be valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_add_entity(
    handle: Handle,
    name: *const c_char,
    entity_type: *const c_char,
    embedding_idx: i64,
) -> i64 {
    // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
    let mem = match unsafe { handle.as_mut() } {
        Some(m) => m,
        None => return -1,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let name = match unsafe { cstr_to_string(name) } {
        Some(s) => s,
        None => return -1,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let entity_type = match unsafe { cstr_to_string(entity_type) } {
        Some(s) => s,
        None => return -1,
    };

    match mem.add_entity(&name, &entity_type, embedding_idx) {
        Ok(id) => id as i64,
        Err(_) => -1,
    }
}

/// Add a knowledge graph relation. Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - `relation` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_add_relation(
    handle: Handle,
    src: u64,
    tgt: u64,
    relation: *const c_char,
    weight: f32,
) -> i32 {
    // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
    let mem = match unsafe { handle.as_mut() } {
        Some(m) => m,
        None => return -1,
    };
    // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
    let relation = match unsafe { cstr_to_string(relation) } {
        Some(s) => s,
        None => return -1,
    };

    match mem.add_relation(src, tgt, &relation, weight) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a C string pointer to an owned Rust String.
///
/// # Safety
/// The pointer must be a valid null-terminated C string.
unsafe fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: ptr is non-null (checked above) and is a valid null-terminated C string per caller.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(String::from)
}
