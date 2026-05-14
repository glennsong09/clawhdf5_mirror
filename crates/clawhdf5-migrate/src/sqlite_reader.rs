use rusqlite::{Connection, Result as SqlResult};

/// A memory chunk read from SQLite.
#[derive(Debug, Clone)]
pub struct MemoryChunk {
    pub id: i64,
    pub chunk: String,
    pub embedding: Vec<f32>,
    pub source_channel: String,
    pub timestamp: f64,
    pub session_id: String,
    pub tags: String,
    pub deleted: i32,
}

/// A session read from SQLite.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub start_idx: i64,
    pub end_idx: i64,
    pub channel: String,
    pub timestamp: f64,
    pub summary: String,
}

/// An entity read from SQLite.
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: i64,
    pub name: String,
    pub entity_type: String,
    pub embedding_idx: i64,
}

/// A relation read from SQLite.
#[derive(Debug, Clone)]
pub struct Relation {
    pub src: i64,
    pub tgt: i64,
    pub relation: String,
    pub weight: f64,
    pub timestamp: f64,
}

/// All data read from a ZeroClaw SQLite database.
#[derive(Debug)]
pub struct SqliteData {
    pub chunks: Vec<MemoryChunk>,
    pub sessions: Vec<Session>,
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub embedding_dim: usize,
}

/// Auto-detect embedding dimension from the first chunk's BLOB size.
fn detect_embedding_dim(conn: &Connection) -> SqlResult<Option<usize>> {
    let mut stmt = conn.prepare("SELECT embedding FROM memory_chunks LIMIT 1")?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        let blob: Vec<u8> = row.get(0)?;
        Ok(Some(blob.len() / 4)) // f32 = 4 bytes
    } else {
        Ok(None)
    }
}

/// Parse a raw byte BLOB into a Vec<f32>.
fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Read all data from a ZeroClaw SQLite database.
///
/// If `skip_deleted` is true, rows with `deleted=1` are excluded from chunks.
/// If `embedding_dim` is `None`, auto-detect from the first row.
pub fn read_sqlite(
    path: &str,
    skip_deleted: bool,
    embedding_dim: Option<usize>,
) -> Result<SqliteData, Box<dyn std::error::Error>> {
    let conn = Connection::open(path)?;

    let dim = match embedding_dim {
        Some(d) => d,
        None => detect_embedding_dim(&conn)?.unwrap_or(0),
    };

    let chunks = read_chunks(&conn, skip_deleted, dim)?;
    let sessions = read_sessions(&conn)?;
    let entities = read_entities(&conn)?;
    let relations = read_relations(&conn)?;

    Ok(SqliteData {
        chunks,
        sessions,
        entities,
        relations,
        embedding_dim: dim,
    })
}

fn read_chunks(
    conn: &Connection,
    skip_deleted: bool,
    expected_dim: usize,
) -> SqlResult<Vec<MemoryChunk>> {
    let sql = if skip_deleted {
        "SELECT id, chunk, embedding, source_channel, timestamp, session_id, tags, deleted \
         FROM memory_chunks WHERE deleted = 0"
    } else {
        "SELECT id, chunk, embedding, source_channel, timestamp, session_id, tags, deleted \
         FROM memory_chunks"
    };

    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
        let blob: Vec<u8> = row.get(2)?;
        let mut embedding = blob_to_f32(&blob);

        // Validate/truncate to expected dimension
        if expected_dim > 0 {
            embedding.truncate(expected_dim);
        }

        Ok(MemoryChunk {
            id: row.get(0)?,
            chunk: row.get(1)?,
            embedding,
            source_channel: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            timestamp: row.get(4)?,
            session_id: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
            tags: row.get::<_, Option<String>>(6)?.unwrap_or_default(),
            deleted: row.get(7)?,
        })
    })?;

    rows.collect()
}

fn read_sessions(conn: &Connection) -> SqlResult<Vec<Session>> {
    let mut stmt =
        conn.prepare("SELECT id, start_idx, end_idx, channel, timestamp, summary FROM sessions")?;
    let rows = stmt.query_map([], |row| {
        Ok(Session {
            id: row.get(0)?,
            start_idx: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
            end_idx: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
            channel: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            timestamp: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
            summary: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
        })
    })?;
    rows.collect()
}

fn read_entities(conn: &Connection) -> SqlResult<Vec<Entity>> {
    let mut stmt = conn.prepare("SELECT id, name, type, embedding_idx FROM entities")?;
    let rows = stmt.query_map([], |row| {
        Ok(Entity {
            id: row.get(0)?,
            name: row.get(1)?,
            entity_type: row.get(2)?,
            embedding_idx: row.get::<_, Option<i64>>(3)?.unwrap_or(-1),
        })
    })?;
    rows.collect()
}

fn read_relations(conn: &Connection) -> SqlResult<Vec<Relation>> {
    let mut stmt = conn.prepare("SELECT src, tgt, relation, weight, timestamp FROM relations")?;
    let rows = stmt.query_map([], |row| {
        Ok(Relation {
            src: row.get(0)?,
            tgt: row.get(1)?,
            relation: row.get(2)?,
            weight: row.get::<_, Option<f64>>(3)?.unwrap_or(1.0),
            timestamp: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
        })
    })?;
    rows.collect()
}
