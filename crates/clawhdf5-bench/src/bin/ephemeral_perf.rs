use clawhdf5_agent::ephemeral::{EphemeralConfig, EphemeralStore};
use std::time::Instant;

fn main() {
    println!("=================================================================");
    println!("  Ephemeral Tier Latency Microbenchmark");
    println!("=================================================================\n");

    // Warm up
    let mut store = EphemeralStore::new(EphemeralConfig {
        max_entries: 100_000,
        default_ttl_secs: 3600.0,
        track_access: true,
    });

    // --- SET latency ---
    let n = 100_000;
    let start = Instant::now();
    for i in 0..n {
        store.set_text(
            &format!("key:{i}"),
            &format!("value-{i}-padding-text-for-realistic-size"),
            None,
        );
    }
    let set_elapsed = start.elapsed();
    let set_per_op = set_elapsed.as_nanos() as f64 / n as f64;
    println!(
        "SET {n} entries:  {:.2} ms total  ({:.0} ns/op  {:.0} ops/sec)",
        set_elapsed.as_secs_f64() * 1000.0,
        set_per_op,
        1e9 / set_per_op
    );

    // --- GET latency (hits) ---
    let start = Instant::now();
    for i in 0..n {
        let _ = store.get_text(&format!("key:{i}"));
    }
    let get_elapsed = start.elapsed();
    let get_per_op = get_elapsed.as_nanos() as f64 / n as f64;
    println!(
        "GET {n} hits:     {:.2} ms total  ({:.0} ns/op  {:.0} ops/sec)",
        get_elapsed.as_secs_f64() * 1000.0,
        get_per_op,
        1e9 / get_per_op
    );

    // --- GET latency (misses) ---
    let start = Instant::now();
    for i in 0..n {
        let _ = store.get_text(&format!("miss:{i}"));
    }
    let miss_elapsed = start.elapsed();
    let miss_per_op = miss_elapsed.as_nanos() as f64 / n as f64;
    println!(
        "GET {n} misses:   {:.2} ms total  ({:.0} ns/op  {:.0} ops/sec)",
        miss_elapsed.as_secs_f64() * 1000.0,
        miss_per_op,
        1e9 / miss_per_op
    );

    // --- DELETE latency ---
    let start = Instant::now();
    for i in 0..n {
        store.delete(&format!("key:{i}"));
    }
    let del_elapsed = start.elapsed();
    let del_per_op = del_elapsed.as_nanos() as f64 / n as f64;
    println!(
        "DEL {n} entries:  {:.2} ms total  ({:.0} ns/op  {:.0} ops/sec)",
        del_elapsed.as_secs_f64() * 1000.0,
        del_per_op,
        1e9 / del_per_op
    );

    // --- SET with embeddings ---
    let dim = 384;
    let emb: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let n_emb = 10_000;
    let start = Instant::now();
    for i in 0..n_emb {
        store.set_with_embedding(
            &format!("emb:{i}"),
            &format!("embedding text {i}"),
            emb.clone(),
            None,
        );
    }
    let emb_elapsed = start.elapsed();
    let emb_per_op = emb_elapsed.as_nanos() as f64 / n_emb as f64;
    println!(
        "SET+EMB {n_emb} entries: {:.2} ms total  ({:.0} ns/op  {:.0} ops/sec)",
        emb_elapsed.as_secs_f64() * 1000.0,
        emb_per_op,
        1e9 / emb_per_op
    );

    // --- Embedding search ---
    let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 + 0.0001).collect();
    let start = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        let _ = store.search_embedding(&query, 10);
    }
    let search_elapsed = start.elapsed();
    let search_per_op = search_elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "SEARCH embedding 10K@384d: {:.2} ms/query ({:.0} µs avg)",
        search_per_op / 1e6,
        search_per_op / 1e3
    );

    let stats = store.stats();
    println!(
        "\nStats: {} entries, {} bytes, {} hits, {} misses",
        stats.total_entries, stats.total_bytes, stats.hit_count, stats.miss_count
    );

    println!("\n--- Redis comparison (typical single-node) ---");
    println!("Redis SET:  ~25,000 ns/op");
    println!("Redis GET:  ~25,000 ns/op");
    println!(
        "Ephemeral SET: {:.0} ns/op  ({:.1}x faster)",
        set_per_op,
        25000.0 / set_per_op
    );
    println!(
        "Ephemeral GET: {:.0} ns/op  ({:.1}x faster)",
        get_per_op,
        25000.0 / get_per_op
    );
}
