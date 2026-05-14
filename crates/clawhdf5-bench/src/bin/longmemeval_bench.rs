//! LongMemEval Benchmark Harness (Track 8.1)
//!
//! Evaluates BM25-based retrieval recall against the LongMemEval dataset (500 questions).
//! Since no embedding model is available at bench time, all embeddings are zero vectors
//! and `hybrid_search` operates in BM25-only mode (vector_weight=0.0, keyword_weight=1.0).
//!
//! This matches the MemX paper methodology: evaluate retrieval recall, not answer generation.
//!
//! # Usage
//! ```
//! cargo run --release --bin longmemeval_bench [path/to/longmemeval_oracle.json]
//! ```
//!
//! # WASM Note
//! `#[cfg(target_arch = "wasm32")]` is not supported here. Changes required for wasm32:
//! - `std::fs::read_to_string` → fetch-based async loader (e.g. `wasm_bindgen_futures`)
//! - `TempDir` → virtual in-memory HDF5 backend (separate effort; tracked in ROADMAP)
//! - `std::time::Instant` → `web_sys::Performance::now()`
//! - HDF5 I/O layer would need a wasm32 storage backend (out of scope for this bench)

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};
use serde::Deserialize;
use tempfile::TempDir;

const EMBEDDING_DIM: usize = 384;

// ---------------------------------------------------------------------------
// JSON data types
// ---------------------------------------------------------------------------

/// The `answer` field in LongMemEval can be a string or a number.
fn deserialize_answer<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    struct AnswerVisitor;
    impl<'de> de::Visitor<'de> for AnswerVisitor {
        type Value = String;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a string or number")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_string<E: de::Error>(self, v: String) -> Result<String, E> {
            Ok(v)
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<String, E> {
            Ok(v.to_string())
        }
    }
    deserializer.deserialize_any(AnswerVisitor)
}

#[derive(Deserialize)]
struct Turn {
    #[allow(dead_code)]
    role: String,
    content: String,
    #[serde(default)]
    has_answer: bool,
}

#[derive(Deserialize)]
struct Question {
    #[allow(dead_code)]
    question_id: String,
    question_type: String,
    question: String,
    #[allow(dead_code)]
    #[serde(deserialize_with = "deserialize_answer")]
    answer: String,
    #[allow(dead_code)]
    question_date: String,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
    answer_session_ids: Vec<String>,
}

// ---------------------------------------------------------------------------
// Per-type metrics accumulator
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Metrics {
    hit1_session: u32,
    hit5_session: u32,
    hit10_session: u32,
    rr_session: f64,
    hit1_turn: u32,
    hit5_turn: u32,
    hit10_turn: u32,
    rr_turn: f64,
    abstention_correct: u32,
    abstention_total: u32,
    latency_ns: Vec<u64>,
    count: u32,
}

impl Metrics {
    fn hit1_session_pct(&self) -> f64 {
        self.hit1_session as f64 / self.count.max(1) as f64 * 100.0
    }
    fn hit5_session_pct(&self) -> f64 {
        self.hit5_session as f64 / self.count.max(1) as f64 * 100.0
    }
    fn hit10_session_pct(&self) -> f64 {
        self.hit10_session as f64 / self.count.max(1) as f64 * 100.0
    }
    fn mrr_session(&self) -> f64 {
        self.rr_session / self.count.max(1) as f64
    }
    fn hit1_turn_pct(&self) -> f64 {
        self.hit1_turn as f64 / self.count.max(1) as f64 * 100.0
    }
    fn hit5_turn_pct(&self) -> f64 {
        self.hit5_turn as f64 / self.count.max(1) as f64 * 100.0
    }
    fn hit10_turn_pct(&self) -> f64 {
        self.hit10_turn as f64 / self.count.max(1) as f64 * 100.0
    }
    fn mrr_turn(&self) -> f64 {
        self.rr_turn / self.count.max(1) as f64
    }
    fn abstention_pct(&self) -> f64 {
        self.abstention_correct as f64 / self.abstention_total.max(1) as f64 * 100.0
    }
    fn latency_avg_us(&self) -> f64 {
        if self.latency_ns.is_empty() {
            return 0.0;
        }
        self.latency_ns.iter().sum::<u64>() as f64 / self.latency_ns.len() as f64 / 1000.0
    }
    fn latency_pct_us(&self, p: usize) -> f64 {
        if self.latency_ns.is_empty() {
            return 0.0;
        }
        let mut v = self.latency_ns.clone();
        v.sort_unstable();
        let idx = (p * v.len() / 100).min(v.len() - 1);
        v[idx] as f64 / 1000.0
    }
}

// ---------------------------------------------------------------------------
// Per-question evaluation result
// ---------------------------------------------------------------------------

struct EvalResult {
    hit1_session: bool,
    hit5_session: bool,
    hit10_session: bool,
    rr_session: Option<f64>,
    hit1_turn: bool,
    hit5_turn: bool,
    hit10_turn: bool,
    rr_turn: Option<f64>,
    latency: Duration,
}

fn evaluate_question(q: &Question, top_k: usize) -> EvalResult {
    let dir = TempDir::new().expect("failed to create temp dir");
    let mut config = MemoryConfig::new(dir.path().join("lme.h5"), "lme-bench", EMBEDDING_DIM);
    config.wal_enabled = false;
    config.compact_threshold = 0.0;

    let mut memory = HDF5Memory::create(config).expect("failed to create HDF5Memory");

    // Build MemoryEntry list from all haystack sessions
    let mut entries: Vec<MemoryEntry> = Vec::new();
    let mut turn_has_answer: Vec<bool> = Vec::new();
    let mut ts = 1_000_000.0f64;

    for (sess_idx, session) in q.haystack_sessions.iter().enumerate() {
        let sess_id = q
            .haystack_session_ids
            .get(sess_idx)
            .map(String::as_str)
            .unwrap_or("unknown");
        for turn in session {
            entries.push(MemoryEntry {
                chunk: turn.content.clone(),
                embedding: vec![0.0f32; EMBEDDING_DIM],
                source_channel: "longmemeval".to_string(),
                timestamp: ts,
                session_id: sess_id.to_string(),
                tags: if turn.has_answer {
                    "has_answer".to_string()
                } else {
                    String::new()
                },
            });
            turn_has_answer.push(turn.has_answer);
            ts += 1.0;
        }
    }

    let indices = memory.save_batch(entries).expect("failed to save entries");

    // Map memory index → has_answer
    let has_answer_indices: HashSet<usize> = indices
        .iter()
        .zip(turn_has_answer.iter())
        .filter(|(_, ha)| **ha)
        .map(|(idx, _)| *idx)
        .collect();

    // Set of session IDs that contain the answer
    let answer_sess_set: HashSet<&str> = q.answer_session_ids.iter().map(String::as_str).collect();

    // Run hybrid search (BM25-only: vector_weight=0.0, keyword_weight=1.0)
    let zero_emb = vec![0.0f32; EMBEDDING_DIM];
    let t0 = Instant::now();
    let results = memory.hybrid_search(&zero_emb, &q.question, 0.0, 1.0, top_k);
    let latency = t0.elapsed();

    // Session-level recall
    let mut hit1_session = false;
    let mut hit5_session = false;
    let mut hit10_session = false;
    let mut rr_session: Option<f64> = None;

    for (rank, result) in results.iter().enumerate() {
        let sess_id = memory.cache.session_ids[result.index].as_str();
        if answer_sess_set.contains(sess_id) {
            hit10_session = true;
            if rank < 5 {
                hit5_session = true;
            }
            if rank == 0 {
                hit1_session = true;
            }
            if rr_session.is_none() {
                rr_session = Some(1.0 / (rank + 1) as f64);
            }
            break;
        }
    }

    // Turn-level recall
    let mut hit1_turn = false;
    let mut hit5_turn = false;
    let mut hit10_turn = false;
    let mut rr_turn: Option<f64> = None;

    for (rank, result) in results.iter().enumerate() {
        if has_answer_indices.contains(&result.index) {
            hit10_turn = true;
            if rank < 5 {
                hit5_turn = true;
            }
            if rank == 0 {
                hit1_turn = true;
            }
            if rr_turn.is_none() {
                rr_turn = Some(1.0 / (rank + 1) as f64);
            }
            break;
        }
    }

    EvalResult {
        hit1_session,
        hit5_session,
        hit10_session,
        rr_session,
        hit1_turn,
        hit5_turn,
        hit10_turn,
        rr_turn,
        latency,
    }
}

// ---------------------------------------------------------------------------
// Report printing
// ---------------------------------------------------------------------------

fn print_report(overall: &Metrics, by_type: &HashMap<String, Metrics>) {
    println!("=================================================================");
    println!("  LongMemEval Benchmark  (BM25-only retrieval, zero embeddings)");
    println!("=================================================================");
    println!();
    println!("Mode: vector_weight=0.0 / keyword_weight=1.0 (pure BM25)");
    println!("Note: MemX (arxiv:2603.16171) with full system: Hit@5=51.6%, MRR=0.380");
    println!("      BM25-only numbers are expected to be lower — honest baseline.");
    println!();

    println!("## Session-Level Recall  (n={})", overall.count);
    println!(
        "  Hit@1:  {:5.1}%   Hit@5:  {:5.1}%   Hit@10: {:5.1}%   MRR: {:.4}",
        overall.hit1_session_pct(),
        overall.hit5_session_pct(),
        overall.hit10_session_pct(),
        overall.mrr_session()
    );
    println!();

    println!("## Turn-Level Recall  (n={})", overall.count);
    println!(
        "  Hit@1:  {:5.1}%   Hit@5:  {:5.1}%   Hit@10: {:5.1}%   MRR: {:.4}",
        overall.hit1_turn_pct(),
        overall.hit5_turn_pct(),
        overall.hit10_turn_pct(),
        overall.mrr_turn()
    );
    println!();

    if overall.abstention_total > 0 {
        println!("## Abstention Accuracy");
        println!(
            "  Correct: {}/{} ({:.1}%)",
            overall.abstention_correct,
            overall.abstention_total,
            overall.abstention_pct()
        );
        println!("  (abstention = system correctly returns no matching session)");
        println!();
    }

    println!(
        "## Search Latency  (n={} queries, BM25 over variable haystack sizes)",
        overall.latency_ns.len()
    );
    println!(
        "  avg={:.1} µs   p50={:.1} µs   p95={:.1} µs   p99={:.1} µs",
        overall.latency_avg_us(),
        overall.latency_pct_us(50),
        overall.latency_pct_us(95),
        overall.latency_pct_us(99)
    );
    println!();

    println!("## Per-Type Breakdown  (session-level)");
    println!();
    println!(
        "{:<32} {:>5}  {:>7}  {:>7}  {:>7}  {:>7}",
        "Question Type", "N", "Hit@1", "Hit@5", "Hit@10", "MRR"
    );
    println!("{}", "-".repeat(72));

    let mut types: Vec<(&String, &Metrics)> = by_type.iter().collect();
    types.sort_by_key(|(t, _)| t.as_str());

    for (qtype, m) in &types {
        if m.count > 0 {
            println!(
                "{:<32} {:>5}  {:>6.1}%  {:>6.1}%  {:>6.1}%  {:>7.4}",
                qtype,
                m.count,
                m.hit1_session_pct(),
                m.hit5_session_pct(),
                m.hit10_session_pct(),
                m.mrr_session()
            );
        }
        if m.abstention_total > 0 {
            println!(
                "{:<32} {:>5}  abstention accuracy: {:>5.1}%",
                format!("{qtype}_abs"),
                m.abstention_total,
                m.abstention_pct()
            );
        }
    }

    // Machine-parseable JSON summary
    println!();
    println!("## JSON Summary");
    println!("```json");
    println!("{{");
    println!("  \"benchmark\": \"longmemeval\",");
    println!("  \"mode\": \"bm25_only\",");
    println!(
        "  \"total_questions\": {},",
        overall.count + overall.abstention_total
    );
    println!("  \"session_level\": {{");
    println!(
        "    \"hit_at_1\": {:.4}, \"hit_at_5\": {:.4}, \"hit_at_10\": {:.4}, \"mrr\": {:.4}",
        overall.hit1_session_pct() / 100.0,
        overall.hit5_session_pct() / 100.0,
        overall.hit10_session_pct() / 100.0,
        overall.mrr_session()
    );
    println!("  }},");
    println!("  \"turn_level\": {{");
    println!(
        "    \"hit_at_1\": {:.4}, \"hit_at_5\": {:.4}, \"hit_at_10\": {:.4}, \"mrr\": {:.4}",
        overall.hit1_turn_pct() / 100.0,
        overall.hit5_turn_pct() / 100.0,
        overall.hit10_turn_pct() / 100.0,
        overall.mrr_turn()
    );
    println!("  }},");
    println!(
        "  \"abstention_accuracy\": {:.4},",
        overall.abstention_pct() / 100.0
    );
    println!("  \"latency_us\": {{");
    println!(
        "    \"avg\": {:.1}, \"p50\": {:.1}, \"p95\": {:.1}, \"p99\": {:.1}",
        overall.latency_avg_us(),
        overall.latency_pct_us(50),
        overall.latency_pct_us(95),
        overall.latency_pct_us(99)
    );
    println!("  }}");
    println!("}}");
    println!("```");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let json_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmarks/longmemeval/longmemeval_oracle.json".to_string());

    eprintln!("Loading: {json_path}");
    let data = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("Failed to read {json_path}: {e}"));
    let questions: Vec<Question> = serde_json::from_str(&data).expect("Failed to parse JSON");
    let total = questions.len();
    eprintln!("Loaded {total} questions");

    let mut overall = Metrics::default();
    let mut by_type: HashMap<String, Metrics> = HashMap::new();

    for (i, q) in questions.iter().enumerate() {
        if (i + 1) % 50 == 0 || i + 1 == total {
            eprint!("\r  [{}/{}] evaluating...", i + 1, total);
        }

        let result = evaluate_question(q, 10);

        let is_abs = q.question_type.ends_with("_abs");
        let base_type = if is_abs {
            q.question_type.trim_end_matches("_abs").to_string()
        } else {
            q.question_type.clone()
        };

        let entry = by_type.entry(base_type).or_default();

        if is_abs {
            entry.abstention_total += 1;
            overall.abstention_total += 1;
            // Correct abstention: no session-level hit in top 10
            if !result.hit10_session {
                entry.abstention_correct += 1;
                overall.abstention_correct += 1;
            }
        } else {
            entry.count += 1;
            overall.count += 1;

            if result.hit1_session {
                entry.hit1_session += 1;
                overall.hit1_session += 1;
            }
            if result.hit5_session {
                entry.hit5_session += 1;
                overall.hit5_session += 1;
            }
            if result.hit10_session {
                entry.hit10_session += 1;
                overall.hit10_session += 1;
            }
            if let Some(rr) = result.rr_session {
                entry.rr_session += rr;
                overall.rr_session += rr;
            }

            if result.hit1_turn {
                entry.hit1_turn += 1;
                overall.hit1_turn += 1;
            }
            if result.hit5_turn {
                entry.hit5_turn += 1;
                overall.hit5_turn += 1;
            }
            if result.hit10_turn {
                entry.hit10_turn += 1;
                overall.hit10_turn += 1;
            }
            if let Some(rr) = result.rr_turn {
                entry.rr_turn += rr;
                overall.rr_turn += rr;
            }

            let ns = result.latency.as_nanos() as u64;
            entry.latency_ns.push(ns);
            overall.latency_ns.push(ns);
        }
    }

    eprintln!("\r  [{total}/{total}] done.          ");
    eprintln!();
    print_report(&overall, &by_type);
}
