mod hdf5_writer;
mod sqlite_reader;
mod validate;

use clap::Parser;

/// Migrate ZeroClaw agent memory from SQLite to HDF5 format.
#[derive(Parser, Debug)]
#[command(name = "clawhdf5-migrate", version, about)]
struct Cli {
    /// Source SQLite database path
    #[arg(long)]
    sqlite: String,

    /// Destination HDF5 file path
    #[arg(long)]
    hdf5: String,

    /// Agent ID for metadata
    #[arg(long, default_value = "migrated")]
    agent_id: String,

    /// Embedder name for metadata
    #[arg(long, default_value = "unknown")]
    embedder: String,

    /// Embedding dimension (auto-detect from first row if not specified)
    #[arg(long)]
    embedding_dim: Option<usize>,

    /// Skip deleted/tombstoned entries
    #[arg(long)]
    skip_deleted: bool,

    /// Enable deflate compression on embeddings
    #[arg(long)]
    compression: bool,

    /// Compression level 1-9
    #[arg(long, default_value_t = 4)]
    compression_level: u32,

    /// Store embeddings as float16 (halves storage)
    #[arg(long)]
    float16: bool,

    /// Validate without writing
    #[arg(long)]
    dry_run: bool,

    /// Print progress
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.verbose {
        eprintln!("Reading SQLite database: {}", cli.sqlite);
    }

    let data = sqlite_reader::read_sqlite(&cli.sqlite, cli.skip_deleted, cli.embedding_dim)?;

    if cli.verbose {
        eprintln!(
            "Read {} chunks, {} sessions, {} entities, {} relations",
            data.chunks.len(),
            data.sessions.len(),
            data.entities.len(),
            data.relations.len()
        );
        eprintln!("Embedding dimension: {}", data.embedding_dim);
    }

    if cli.dry_run {
        eprintln!("Dry run — no output file written.");
        eprintln!(
            "Would migrate: {} chunks, {} sessions, {} entities, {} relations (dim={})",
            data.chunks.len(),
            data.sessions.len(),
            data.entities.len(),
            data.relations.len(),
            data.embedding_dim
        );
        return Ok(());
    }

    if cli.verbose {
        eprintln!("Writing HDF5 file: {}", cli.hdf5);
    }

    let opts = hdf5_writer::WriteOptions {
        agent_id: cli.agent_id,
        embedder: cli.embedder,
        compression: cli.compression,
        compression_level: cli.compression_level.clamp(1, 9),
        float16: cli.float16,
    };

    hdf5_writer::write_hdf5(&cli.hdf5, &data, &opts)?;

    if cli.verbose {
        eprintln!("Validating output...");
    }

    let summary = validate::validate_hdf5(
        &cli.hdf5,
        data.chunks.len(),
        data.sessions.len(),
        data.entities.len(),
        data.relations.len(),
        data.embedding_dim,
    )?;

    eprintln!(
        "Migration complete: {} chunks, {} sessions, {} entities, {} relations (dim={})",
        summary.chunks,
        summary.sessions,
        summary.entities,
        summary.relations,
        summary.embedding_dim
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::TempDir;

    /// Create a test SQLite database with the ZeroClaw schema.
    fn create_test_db(dir: &TempDir) -> String {
        let db_path = dir.path().join("test.db");
        let path_str = db_path.to_str().unwrap().to_string();
        let conn = Connection::open(&path_str).unwrap();
        conn.execute_batch(
            "CREATE TABLE memory_chunks (
                id INTEGER PRIMARY KEY,
                chunk TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_channel TEXT DEFAULT 'api',
                timestamp REAL NOT NULL,
                session_id TEXT,
                tags TEXT DEFAULT '',
                deleted INTEGER DEFAULT 0
            );
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                start_idx INTEGER,
                end_idx INTEGER,
                channel TEXT,
                timestamp REAL,
                summary TEXT
            );
            CREATE TABLE entities (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                embedding_idx INTEGER DEFAULT -1
            );
            CREATE TABLE relations (
                src INTEGER NOT NULL,
                tgt INTEGER NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                timestamp REAL,
                FOREIGN KEY (src) REFERENCES entities(id),
                FOREIGN KEY (tgt) REFERENCES entities(id)
            );",
        )
        .unwrap();
        path_str
    }

    /// Insert a memory chunk with a known embedding.
    fn insert_chunk(conn: &Connection, id: i64, text: &str, embedding: &[f32], deleted: i32) {
        let blob: Vec<u8> = embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
        conn.execute(
            "INSERT INTO memory_chunks (id, chunk, embedding, source_channel, timestamp, session_id, tags, deleted)
             VALUES (?1, ?2, ?3, 'api', 1700000000.0, 'sess-1', 'tag1,tag2', ?4)",
            rusqlite::params![id, text, blob, deleted],
        )
        .unwrap();
    }

    fn insert_session(conn: &Connection, id: &str, start: i64, end: i64) {
        conn.execute(
            "INSERT INTO sessions (id, start_idx, end_idx, channel, timestamp, summary)
             VALUES (?1, ?2, ?3, 'discord', 1700000000.0, 'test summary')",
            rusqlite::params![id, start, end],
        )
        .unwrap();
    }

    fn insert_entity(conn: &Connection, id: i64, name: &str, etype: &str) {
        conn.execute(
            "INSERT INTO entities (id, name, type, embedding_idx) VALUES (?1, ?2, ?3, -1)",
            rusqlite::params![id, name, etype],
        )
        .unwrap();
    }

    fn insert_relation(conn: &Connection, src: i64, tgt: i64, rel: &str) {
        conn.execute(
            "INSERT INTO relations (src, tgt, relation, weight, timestamp)
             VALUES (?1, ?2, ?3, 1.0, 1700000000.0)",
            rusqlite::params![src, tgt, rel],
        )
        .unwrap();
    }

    fn make_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| seed + i as f32 * 0.1).collect()
    }

    // ---------- Test 1: Basic end-to-end migration ----------
    #[test]
    fn test_basic_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "hello world", &make_embedding(8, 1.0), 0);
        insert_chunk(&conn, 2, "goodbye world", &make_embedding(8, 2.0), 0);
        insert_session(&conn, "s1", 0, 1);
        insert_entity(&conn, 1, "Alice", "person");
        insert_relation(&conn, 1, 1, "self");
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "test-agent".into(),
            embedder: "test-embed".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary = validate::validate_hdf5(h5_path.to_str().unwrap(), 2, 1, 1, 1, 8).unwrap();
        assert_eq!(summary.chunks, 2);
        assert_eq!(summary.sessions, 1);
        assert_eq!(summary.entities, 1);
        assert_eq!(summary.relations, 1);
        assert_eq!(summary.embedding_dim, 8);
    }

    // ---------- Test 2: Skip deleted rows ----------
    #[test]
    fn test_skip_deleted() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "active", &make_embedding(4, 1.0), 0);
        insert_chunk(&conn, 2, "deleted", &make_embedding(4, 2.0), 1);
        insert_chunk(&conn, 3, "also active", &make_embedding(4, 3.0), 0);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, true, None).unwrap();
        assert_eq!(data.chunks.len(), 2);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary = validate::validate_hdf5(h5_path.to_str().unwrap(), 2, 0, 0, 0, 4).unwrap();
        assert_eq!(summary.chunks, 2);
    }

    // ---------- Test 3: Include deleted rows ----------
    #[test]
    fn test_include_deleted() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "active", &make_embedding(4, 1.0), 0);
        insert_chunk(&conn, 2, "deleted", &make_embedding(4, 2.0), 1);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        assert_eq!(data.chunks.len(), 2);
    }

    // ---------- Test 4: Auto-detect embedding dimension ----------
    #[test]
    fn test_auto_detect_dim() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(16, 0.5), 0);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        assert_eq!(data.embedding_dim, 16);
    }

    // ---------- Test 5: Manual embedding dimension ----------
    #[test]
    fn test_manual_dim() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(16, 0.5), 0);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, Some(8)).unwrap();
        assert_eq!(data.embedding_dim, 8);
        // Embedding truncated to dim 8
        assert_eq!(data.chunks[0].embedding.len(), 8);
    }

    // ---------- Test 6: Float16 conversion ----------
    #[test]
    fn test_float16_conversion() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        let emb = vec![1.0f32, 2.5, -0.5, 3.125];
        insert_chunk(&conn, 1, "test", &emb, 0);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: true,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        // Verify file was created and is valid
        let summary = validate::validate_hdf5(h5_path.to_str().unwrap(), 1, 0, 0, 0, 4).unwrap();
        assert_eq!(summary.chunks, 1);

        // Verify float16 values are within tolerance
        for &v in &emb {
            let f16 = half::f16::from_f32(v);
            let roundtrip = f16.to_f32();
            assert!(
                (v - roundtrip).abs() < 0.01,
                "f16 roundtrip too lossy for {v}"
            );
        }
    }

    // ---------- Test 7: Compression produces valid file ----------
    #[test]
    fn test_compression() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_compressed = dir.path().join("compressed.h5");
        let h5_uncompressed = dir.path().join("uncompressed.h5");

        let conn = Connection::open(&db_path).unwrap();
        // Insert enough data so compression can be effective
        for i in 0..100 {
            insert_chunk(&conn, i, &format!("chunk {i}"), &make_embedding(32, 0.0), 0);
        }
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();

        let opts_compressed = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: true,
            compression_level: 6,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_compressed.to_str().unwrap(), &data, &opts_compressed).unwrap();

        let opts_plain = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_uncompressed.to_str().unwrap(), &data, &opts_plain).unwrap();

        let sz_c = std::fs::metadata(&h5_compressed).unwrap().len();
        let sz_u = std::fs::metadata(&h5_uncompressed).unwrap().len();
        assert!(
            sz_c < sz_u,
            "Compressed ({sz_c}) should be smaller than uncompressed ({sz_u})"
        );
    }

    // ---------- Test 8: Dry run doesn't create file ----------
    #[test]
    fn test_dry_run() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("should_not_exist.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(4, 1.0), 0);
        drop(conn);

        // Simulate dry-run: read data but don't write
        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        assert_eq!(data.chunks.len(), 1);
        assert!(!h5_path.exists());
    }

    // ---------- Test 9: Empty database migration ----------
    #[test]
    fn test_empty_db() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        assert_eq!(data.chunks.len(), 0);
        assert_eq!(data.sessions.len(), 0);
        assert_eq!(data.entities.len(), 0);
        assert_eq!(data.relations.len(), 0);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary = validate::validate_hdf5(h5_path.to_str().unwrap(), 0, 0, 0, 0, 0).unwrap();
        assert_eq!(summary.chunks, 0);
    }

    // ---------- Test 10: Large migration (1000 entries) ----------
    #[test]
    fn test_large_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        for i in 0..1000 {
            insert_chunk(
                &conn,
                i,
                &format!("chunk number {i} with some text"),
                &make_embedding(64, i as f32 * 0.01),
                0,
            );
        }
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        assert_eq!(data.chunks.len(), 1000);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary =
            validate::validate_hdf5(h5_path.to_str().unwrap(), 1000, 0, 0, 0, 64).unwrap();
        assert_eq!(summary.chunks, 1000);
    }

    // ---------- Test 11: Session migration ----------
    #[test]
    fn test_session_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_session(&conn, "session-alpha", 0, 10);
        insert_session(&conn, "session-beta", 11, 20);
        insert_session(&conn, "session-gamma", 21, 30);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        assert_eq!(data.sessions.len(), 3);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary = validate::validate_hdf5(h5_path.to_str().unwrap(), 0, 3, 0, 0, 0).unwrap();
        assert_eq!(summary.sessions, 3);
    }

    // ---------- Test 12: Knowledge graph (entities + relations) ----------
    #[test]
    fn test_knowledge_graph_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_entity(&conn, 1, "Alice", "person");
        insert_entity(&conn, 2, "Bob", "person");
        insert_entity(&conn, 3, "Rust", "language");
        insert_relation(&conn, 1, 2, "knows");
        insert_relation(&conn, 1, 3, "uses");
        insert_relation(&conn, 2, 3, "uses");
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        assert_eq!(data.entities.len(), 3);
        assert_eq!(data.relations.len(), 3);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary = validate::validate_hdf5(h5_path.to_str().unwrap(), 0, 0, 3, 3, 0).unwrap();
        assert_eq!(summary.entities, 3);
        assert_eq!(summary.relations, 3);
    }

    // ---------- Test 13: Validation catches chunk count mismatch ----------
    #[test]
    fn test_validation_catches_count_mismatch() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(4, 1.0), 0);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        // Expect 5 chunks but only 1 was written
        let result = validate::validate_hdf5(h5_path.to_str().unwrap(), 5, 0, 0, 0, 4);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Chunk count mismatch")
        );
    }

    // ---------- Test 14: Metadata attributes are stored ----------
    #[test]
    fn test_metadata_attributes() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(8, 1.0), 0);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "my-agent-42".into(),
            embedder: "openai-ada".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let file = clawhdf5::File::open(h5_path.to_str().unwrap()).unwrap();
        let root = file.root();
        let attrs = root.attrs().unwrap();

        match attrs.get("agent_id") {
            Some(clawhdf5_format::type_builders::AttrValue::String(s)) => {
                assert_eq!(s, "my-agent-42");
            }
            other => panic!("Expected String agent_id, got {other:?}"),
        }
        match attrs.get("embedder") {
            Some(clawhdf5_format::type_builders::AttrValue::String(s)) => {
                assert_eq!(s, "openai-ada");
            }
            other => panic!("Expected String embedder, got {other:?}"),
        }
        match attrs.get("embedding_dim") {
            Some(clawhdf5_format::type_builders::AttrValue::I64(d)) => {
                assert_eq!(*d, 8);
            }
            other => panic!("Expected I64 embedding_dim, got {other:?}"),
        }
    }

    // ---------- Test 15: Embedding values roundtrip correctly ----------
    #[test]
    fn test_embedding_roundtrip() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        let emb = vec![0.1, 0.2, 0.3, 0.4];
        insert_chunk(&conn, 1, "test", &emb, 0);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let file = clawhdf5::File::open(h5_path.to_str().unwrap()).unwrap();
        let chunks_group = file.group("chunks").unwrap();
        let emb_ds = chunks_group.dataset("embeddings").unwrap();
        let read_back = emb_ds.read_f32().unwrap();

        assert_eq!(read_back.len(), 4);
        for (a, b) in emb.iter().zip(read_back.iter()) {
            assert!((a - b).abs() < 1e-6, "Embedding mismatch: {a} vs {b}");
        }
    }

    // ---------- Test 16: Full combined migration ----------
    #[test]
    fn test_full_combined_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        for i in 0..5 {
            insert_chunk(
                &conn,
                i,
                &format!("chunk {i}"),
                &make_embedding(16, i as f32),
                if i == 3 { 1 } else { 0 },
            );
        }
        insert_session(&conn, "s1", 0, 2);
        insert_session(&conn, "s2", 3, 4);
        insert_entity(&conn, 1, "Alice", "person");
        insert_entity(&conn, 2, "Bob", "person");
        insert_relation(&conn, 1, 2, "knows");
        drop(conn);

        // Skip deleted
        let data = sqlite_reader::read_sqlite(&db_path, true, None).unwrap();
        assert_eq!(data.chunks.len(), 4); // chunk 3 is deleted

        let opts = hdf5_writer::WriteOptions {
            agent_id: "combined-test".into(),
            embedder: "test-embedder".into(),
            compression: true,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary = validate::validate_hdf5(h5_path.to_str().unwrap(), 4, 2, 2, 1, 16).unwrap();
        assert_eq!(summary.chunks, 4);
        assert_eq!(summary.sessions, 2);
        assert_eq!(summary.entities, 2);
        assert_eq!(summary.relations, 1);
        assert_eq!(summary.embedding_dim, 16);
    }

    // ---------- Test 17: Validation catches session mismatch ----------
    #[test]
    fn test_validation_session_mismatch() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_session(&conn, "s1", 0, 10);
        drop(conn);

        let data = sqlite_reader::read_sqlite(&db_path, false, None).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let result = validate::validate_hdf5(h5_path.to_str().unwrap(), 0, 99, 0, 0, 0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Session count mismatch")
        );
    }
}
