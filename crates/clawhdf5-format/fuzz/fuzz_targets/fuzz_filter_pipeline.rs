#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = clawhdf5_format::filter_pipeline::FilterPipeline::parse(data);
});
