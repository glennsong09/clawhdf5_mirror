use clawhdf5::reader::File;
use clawhdf5_format::type_builders::AttrValue;

/// Summary of a migration validation.
#[derive(Debug)]
pub struct ValidationSummary {
    pub chunks: u64,
    pub sessions: u64,
    pub entities: u64,
    pub relations: u64,
    pub embedding_dim: u64,
}

/// Validate an HDF5 file written by the migration tool.
///
/// Checks that row counts and embedding dimensions match expectations.
pub fn validate_hdf5(
    path: &str,
    expected_chunks: usize,
    expected_sessions: usize,
    expected_entities: usize,
    expected_relations: usize,
    expected_dim: usize,
) -> Result<ValidationSummary, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let root = file.root();

    // Read root attributes
    let attrs = root.attrs()?;
    let stored_dim = match attrs.get("embedding_dim") {
        Some(AttrValue::I64(d)) => *d as u64,
        _ => 0,
    };

    // Validate chunks group
    let chunks_group = file.group("chunks")?;
    let chunk_attrs = chunks_group.attrs()?;
    let chunk_count = match chunk_attrs.get("count") {
        Some(AttrValue::I64(n)) => *n as u64,
        _ => 0,
    };

    if chunk_count != expected_chunks as u64 {
        return Err(format!(
            "Chunk count mismatch: HDF5 has {}, expected {}",
            chunk_count, expected_chunks
        )
        .into());
    }

    // Validate embedding dimensions if chunks exist
    if chunk_count > 0 && expected_dim > 0 {
        let emb_ds = chunks_group.dataset("embeddings")?;
        let shape = emb_ds.shape()?;
        if shape.len() == 2 && shape[1] != expected_dim as u64 {
            return Err(format!(
                "Embedding dim mismatch: HDF5 has {}, expected {}",
                shape[1], expected_dim
            )
            .into());
        }

        if stored_dim != expected_dim as u64 {
            return Err(format!(
                "Embedding dim attr mismatch: HDF5 attr={}, expected {}",
                stored_dim, expected_dim
            )
            .into());
        }
    }

    // Validate sessions group
    let sessions_group = file.group("sessions")?;
    let sess_attrs = sessions_group.attrs()?;
    let session_count = match sess_attrs.get("count") {
        Some(AttrValue::I64(n)) => *n as u64,
        _ => 0,
    };

    if session_count != expected_sessions as u64 {
        return Err(format!(
            "Session count mismatch: HDF5 has {}, expected {}",
            session_count, expected_sessions
        )
        .into());
    }

    // Validate entities group
    let entities_group = file.group("entities")?;
    let ent_attrs = entities_group.attrs()?;
    let entity_count = match ent_attrs.get("count") {
        Some(AttrValue::I64(n)) => *n as u64,
        _ => 0,
    };

    if entity_count != expected_entities as u64 {
        return Err(format!(
            "Entity count mismatch: HDF5 has {}, expected {}",
            entity_count, expected_entities
        )
        .into());
    }

    // Validate relations group
    let relations_group = file.group("relations")?;
    let rel_attrs = relations_group.attrs()?;
    let relation_count = match rel_attrs.get("count") {
        Some(AttrValue::I64(n)) => *n as u64,
        _ => 0,
    };

    if relation_count != expected_relations as u64 {
        return Err(format!(
            "Relation count mismatch: HDF5 has {}, expected {}",
            relation_count, expected_relations
        )
        .into());
    }

    Ok(ValidationSummary {
        chunks: chunk_count,
        sessions: session_count,
        entities: entity_count,
        relations: relation_count,
        embedding_dim: stored_dim,
    })
}
