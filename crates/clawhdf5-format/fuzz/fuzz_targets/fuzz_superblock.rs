#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try parsing with signature at offset 0
    let _ = clawhdf5_format::superblock::Superblock::parse(data, 0);
    // Try with a valid-ish signature search first
    if let Ok(offset) = clawhdf5_format::signature::find_signature(data) {
        let _ = clawhdf5_format::superblock::Superblock::parse(data, offset);
    }
});
