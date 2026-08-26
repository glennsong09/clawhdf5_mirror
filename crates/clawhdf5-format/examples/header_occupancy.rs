//! Object-header occupancy of a post-quantum signature attribute (S2-D2-Yr2,
//! Paper B §VII).
//!
//! The design stores the signed file-level root in an HDF5 attribute. The
//! question §VII needs answered is what that costs in the object header, and
//! at what signature size the header stops being able to hold it.
//!
//! This measures `ObjectHeaderWriter`, the writer in this workspace. It is a
//! measurement of the reference implementation, not of the HDF5 C library --
//! see the note printed at the end, which is part of the result.
//!
//!     cargo run -p clawhdf5-format --release --example header_occupancy

use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header_writer::ObjectHeaderWriter;

/// Messages a minimal real dataset header carries besides its attributes.
/// Sizes are representative of a 3-D chunked float dataset.
fn base_header() -> ObjectHeaderWriter {
    let mut w = ObjectHeaderWriter::new();
    w.add_message(MessageType::Dataspace, vec![0u8; 35]);
    w.add_message(MessageType::Datatype, vec![0u8; 24]);
    w.add_message(MessageType::FillValue, vec![0u8; 8]);
    w.add_message(MessageType::DataLayout, vec![0u8; 40]);
    w
}

/// An attribute message carrying `payload` bytes of value, with realistic
/// framing: v3 header (4 bytes), name "_merkle_signature\0" (18), an opaque
/// datatype (16), and a scalar-to-1D dataspace (16).
fn attribute_message(payload: usize) -> Vec<u8> {
    const FRAMING: usize = 4 + 18 + 16 + 16;
    vec![0u8; FRAMING + payload]
}

fn header_len(payload: Option<usize>) -> usize {
    let mut w = base_header();
    if let Some(p) = payload {
        w.add_message(MessageType::Attribute, attribute_message(p));
    }
    w.serialize().len()
}

fn main() {
    println!("=== Object-header occupancy of a signature attribute ===\n");

    let bare = header_len(None);
    println!("Baseline dataset header, no signature attribute: {bare} bytes\n");

    println!("{:<26}{:>10}{:>14}{:>12}", "scheme", "sig+pk B", "header B", "delta B");
    println!("{}", "-".repeat(62));
    let schemes: [(&str, usize); 4] = [
        ("(none)", 0),
        ("Ed25519", 96),
        ("Ed25519 + ML-DSA-65", 5358),
        ("Ed25519 + SLH-DSA-256s", 29_824),
    ];
    for (name, sz) in schemes {
        if sz == 0 {
            println!("{:<26}{:>10}{:>14}{:>12}", name, "-", bare, 0);
            continue;
        }
        let h = header_len(Some(sz));
        println!("{:<26}{:>10}{:>14}{:>12}", name, sz, h, h as i64 - bare as i64);
    }

    // Where the chunk0 size field widens (1 -> 2 -> 4 bytes). These are the
    // only structural transitions this writer has.
    println!("\n=== chunk0 size-field transitions ===");
    let mut prev = header_len(Some(1));
    for p in 2..1000usize {
        let h = header_len(Some(p));
        if h != prev + 1 {
            println!("  payload {p:>5} B: header jumps {prev} -> {h} (size field widened)");
        }
        prev = h;
    }

    // The u16 message-size field. A message of 65_536 bytes or more cannot be
    // described by it. Does the writer detect that, or silently wrap?
    println!("\n=== u16 message-size field: behaviour at and past 64 KiB ===");
    for payload in [65_000usize, 65_480, 65_600, 200_000] {
        let msg = attribute_message(payload);
        let declared = msg.len() as u16 as usize; // what serialize() writes
        let actual = msg.len();
        let status = if declared == actual { "ok" } else { "TRUNCATED" };
        println!(
            "  payload {payload:>7} B -> message {actual:>7} B, size field declares {declared:>7} B  [{status}]"
        );
    }

    println!("\n=== continuation blocks ===");
    let big = header_len(Some(200_000));
    let ser = {
        let mut w = base_header();
        w.add_message(MessageType::Attribute, attribute_message(200_000));
        w.serialize()
    };
    let has_cont = ser.windows(4).any(|w| w == b"OCHK");
    println!("  200 KB attribute -> {big} byte header, OCHK continuation block present: {has_cont}");
    println!(
        "  ObjectHeaderWriter::serialize() has no continuation path: every message\n  \
         is emitted into chunk 0 regardless of total size."
    );

    println!(
        "\nNOTE: these are measurements of this workspace's writer, not of the HDF5\n\
         C library. The 64 KB object-header limit and the compact-to-dense attribute\n\
         transition are library behaviours this writer does not model, so the paper\n\
         must not cite these figures as HDF5 limits."
    );
}
