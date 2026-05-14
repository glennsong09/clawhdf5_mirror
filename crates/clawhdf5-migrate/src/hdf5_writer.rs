use clawhdf5::writer::FileBuilder;
use clawhdf5_format::datatype::{CharacterSet, Datatype, StringPadding};
use clawhdf5_format::type_builders::AttrValue;

use crate::sqlite_reader::SqliteData;

/// Options controlling HDF5 output.
pub struct WriteOptions {
    pub agent_id: String,
    pub embedder: String,
    pub compression: bool,
    pub compression_level: u32,
    pub float16: bool,
}

/// Write SQLite data to an HDF5 file.
pub fn write_hdf5(
    path: &str,
    data: &SqliteData,
    opts: &WriteOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = FileBuilder::new();

    // Root-level metadata attributes
    builder.set_attr("agent_id", AttrValue::String(opts.agent_id.clone()));
    builder.set_attr("embedder", AttrValue::String(opts.embedder.clone()));
    builder.set_attr("embedding_dim", AttrValue::I64(data.embedding_dim as i64));
    builder.set_attr("source", AttrValue::String("sqlite-migration".into()));
    builder.set_attr("version", AttrValue::I64(1));

    write_chunks_group(&mut builder, data, opts);
    write_sessions_group(&mut builder, data);
    write_entities_group(&mut builder, data);
    write_relations_group(&mut builder, data);

    builder.write(path)?;
    Ok(())
}

/// Build a fixed-length string Datatype from the max byte length of the items.
fn string_dtype(max_len: usize) -> Datatype {
    Datatype::String {
        size: max_len.max(1) as u32,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Utf8,
    }
}

/// Pack a slice of strings into null-padded raw bytes of uniform width.
fn pack_strings(strings: &[String]) -> (Vec<u8>, usize) {
    let max_len = strings.iter().map(|s| s.len()).max().unwrap_or(0).max(1);
    let mut buf = vec![0u8; strings.len() * max_len];
    for (i, s) in strings.iter().enumerate() {
        let start = i * max_len;
        let bytes = s.as_bytes();
        let copy_len = bytes.len().min(max_len);
        buf[start..start + copy_len].copy_from_slice(&bytes[..copy_len]);
    }
    (buf, max_len)
}

fn apply_compression(ds: &mut clawhdf5_format::type_builders::DatasetBuilder, opts: &WriteOptions) {
    if opts.compression {
        ds.with_deflate(opts.compression_level);
        ds.with_shuffle();
    }
}

fn write_chunks_group(builder: &mut FileBuilder, data: &SqliteData, opts: &WriteOptions) {
    let mut group = builder.create_group("chunks");
    let n = data.chunks.len() as u64;

    if n == 0 {
        group.set_attr("count", AttrValue::I64(0));
        builder.add_group(group.finish());
        return;
    }

    group.set_attr("count", AttrValue::I64(n as i64));

    // ids
    let ids: Vec<i64> = data.chunks.iter().map(|c| c.id).collect();
    group.create_dataset("id").with_i64_data(&ids);

    // text
    let texts: Vec<String> = data.chunks.iter().map(|c| c.chunk.clone()).collect();
    let (text_raw, text_len) = pack_strings(&texts);
    group
        .create_dataset("text")
        .with_compound_data(string_dtype(text_len), text_raw, n);

    // embeddings - flatten to [N, dim]
    let dim = data.embedding_dim;
    if opts.float16 {
        let f16_data: Vec<u16> = data
            .chunks
            .iter()
            .flat_map(|c| {
                c.embedding
                    .iter()
                    .map(|&v| half::f16::from_f32(v).to_bits())
            })
            .collect();
        let raw: Vec<u8> = f16_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_dtype = Datatype::FloatingPoint {
            size: 2,
            byte_order: clawhdf5_format::datatype::DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 16,
            exponent_location: 10,
            exponent_size: 5,
            mantissa_location: 0,
            mantissa_size: 10,
            exponent_bias: 15,
        };
        let ds = group
            .create_dataset("embeddings")
            .with_compound_data(f16_dtype, raw, n)
            .with_shape(&[n, dim as u64]);
        apply_compression(ds, opts);
    } else {
        let flat: Vec<f32> = data
            .chunks
            .iter()
            .flat_map(|c| c.embedding.iter().copied())
            .collect();
        let ds = group
            .create_dataset("embeddings")
            .with_f32_data(&flat)
            .with_shape(&[n, dim as u64]);
        apply_compression(ds, opts);
    }

    // source_channel
    let channels: Vec<String> = data
        .chunks
        .iter()
        .map(|c| c.source_channel.clone())
        .collect();
    let (ch_raw, ch_len) = pack_strings(&channels);
    group
        .create_dataset("source_channel")
        .with_compound_data(string_dtype(ch_len), ch_raw, n);

    // timestamp
    let timestamps: Vec<f64> = data.chunks.iter().map(|c| c.timestamp).collect();
    group.create_dataset("timestamp").with_f64_data(&timestamps);

    // session_id
    let sess_ids: Vec<String> = data.chunks.iter().map(|c| c.session_id.clone()).collect();
    let (sid_raw, sid_len) = pack_strings(&sess_ids);
    group
        .create_dataset("session_id")
        .with_compound_data(string_dtype(sid_len), sid_raw, n);

    // tags
    let tags: Vec<String> = data.chunks.iter().map(|c| c.tags.clone()).collect();
    let (tag_raw, tag_len) = pack_strings(&tags);
    group
        .create_dataset("tags")
        .with_compound_data(string_dtype(tag_len), tag_raw, n);

    // deleted
    let deleted: Vec<i32> = data.chunks.iter().map(|c| c.deleted).collect();
    group.create_dataset("deleted").with_i32_data(&deleted);

    builder.add_group(group.finish());
}

fn write_sessions_group(builder: &mut FileBuilder, data: &SqliteData) {
    let mut group = builder.create_group("sessions");
    let n = data.sessions.len() as u64;
    group.set_attr("count", AttrValue::I64(n as i64));

    if n == 0 {
        builder.add_group(group.finish());
        return;
    }

    let ids: Vec<String> = data.sessions.iter().map(|s| s.id.clone()).collect();
    let (id_raw, id_len) = pack_strings(&ids);
    group
        .create_dataset("id")
        .with_compound_data(string_dtype(id_len), id_raw, n);

    let start_idxs: Vec<i64> = data.sessions.iter().map(|s| s.start_idx).collect();
    group.create_dataset("start_idx").with_i64_data(&start_idxs);

    let end_idxs: Vec<i64> = data.sessions.iter().map(|s| s.end_idx).collect();
    group.create_dataset("end_idx").with_i64_data(&end_idxs);

    let channels: Vec<String> = data.sessions.iter().map(|s| s.channel.clone()).collect();
    let (ch_raw, ch_len) = pack_strings(&channels);
    group
        .create_dataset("channel")
        .with_compound_data(string_dtype(ch_len), ch_raw, n);

    let timestamps: Vec<f64> = data.sessions.iter().map(|s| s.timestamp).collect();
    group.create_dataset("timestamp").with_f64_data(&timestamps);

    let summaries: Vec<String> = data.sessions.iter().map(|s| s.summary.clone()).collect();
    let (sum_raw, sum_len) = pack_strings(&summaries);
    group
        .create_dataset("summary")
        .with_compound_data(string_dtype(sum_len), sum_raw, n);

    builder.add_group(group.finish());
}

fn write_entities_group(builder: &mut FileBuilder, data: &SqliteData) {
    let mut group = builder.create_group("entities");
    let n = data.entities.len() as u64;
    group.set_attr("count", AttrValue::I64(n as i64));

    if n == 0 {
        builder.add_group(group.finish());
        return;
    }

    let ids: Vec<i64> = data.entities.iter().map(|e| e.id).collect();
    group.create_dataset("id").with_i64_data(&ids);

    let names: Vec<String> = data.entities.iter().map(|e| e.name.clone()).collect();
    let (name_raw, name_len) = pack_strings(&names);
    group
        .create_dataset("name")
        .with_compound_data(string_dtype(name_len), name_raw, n);

    let types: Vec<String> = data
        .entities
        .iter()
        .map(|e| e.entity_type.clone())
        .collect();
    let (type_raw, type_len) = pack_strings(&types);
    group
        .create_dataset("type")
        .with_compound_data(string_dtype(type_len), type_raw, n);

    let emb_idxs: Vec<i64> = data.entities.iter().map(|e| e.embedding_idx).collect();
    group
        .create_dataset("embedding_idx")
        .with_i64_data(&emb_idxs);

    builder.add_group(group.finish());
}

fn write_relations_group(builder: &mut FileBuilder, data: &SqliteData) {
    let mut group = builder.create_group("relations");
    let n = data.relations.len() as u64;
    group.set_attr("count", AttrValue::I64(n as i64));

    if n == 0 {
        builder.add_group(group.finish());
        return;
    }

    let srcs: Vec<i64> = data.relations.iter().map(|r| r.src).collect();
    group.create_dataset("src").with_i64_data(&srcs);

    let tgts: Vec<i64> = data.relations.iter().map(|r| r.tgt).collect();
    group.create_dataset("tgt").with_i64_data(&tgts);

    let rels: Vec<String> = data.relations.iter().map(|r| r.relation.clone()).collect();
    let (rel_raw, rel_len) = pack_strings(&rels);
    group
        .create_dataset("relation")
        .with_compound_data(string_dtype(rel_len), rel_raw, n);

    let weights: Vec<f64> = data.relations.iter().map(|r| r.weight).collect();
    group.create_dataset("weight").with_f64_data(&weights);

    let timestamps: Vec<f64> = data.relations.iter().map(|r| r.timestamp).collect();
    group.create_dataset("timestamp").with_f64_data(&timestamps);

    builder.add_group(group.finish());
}
