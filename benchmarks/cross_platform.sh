#!/usr/bin/env bash
# cross_platform.sh — ClawhDF5 Cross-Platform Benchmark Runner (Track 8.6)
#
# Runs the Criterion latency benchmarks and optional standalone bench binaries,
# then outputs results in machine-parseable JSON format.
#
# Usage:
#   ./benchmarks/cross_platform.sh [--full] [--output results.json]
#
# Options:
#   --full       Also run the standalone bench binaries (footprint, consolidation,
#                memory_arena). These take longer but produce richer data.
#   --output F   Write JSON summary to file F (default: stdout)
#   --quiet      Suppress progress messages
#
# Requirements:
#   - Rust toolchain (cargo) in PATH
#   - Run from the workspace root directory
#
# Platform support:
#   - Linux x86_64 / aarch64
#   - macOS x86_64 (Intel) / aarch64 (Apple Silicon)
#   - Windows (via Git Bash or WSL)
#
# WASM note:
#   wasm32-unknown-unknown is NOT supported by this script.
#   The bench binaries require std filesystem access and std::time::Instant.
#   For wasm32 targets:
#     - Use wasm-pack with a custom bench harness
#     - Replace std::time::Instant with web_sys::Performance::now()
#     - Replace TempDir/HDF5 I/O with an in-memory backend (separate effort)
#   See ROADMAP.md §WASM for the full scope.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_OUTPUT_DIR="${WORKSPACE_ROOT}/target/criterion"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_FULL=0
OUTPUT_FILE=""
QUIET=0

for arg in "$@"; do
    case "$arg" in
        --full)   RUN_FULL=1 ;;
        --quiet)  QUIET=1 ;;
        --output=*) OUTPUT_FILE="${arg#--output=}" ;;
        --output)  shift; OUTPUT_FILE="$1" ;;
    esac
done

log() {
    [[ "$QUIET" -eq 0 ]] && echo "[cross_platform] $*" >&2
}

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

detect_platform() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux)   OS_NAME="linux" ;;
        Darwin)  OS_NAME="macos" ;;
        MINGW*|MSYS*|CYGWIN*) OS_NAME="windows" ;;
        *)       OS_NAME="unknown ($os)" ;;
    esac

    case "$arch" in
        x86_64|amd64)  ARCH_NAME="x86_64" ;;
        aarch64|arm64) ARCH_NAME="aarch64" ;;
        armv7*)        ARCH_NAME="armv7" ;;
        *)             ARCH_NAME="unknown ($arch)" ;;
    esac

    # CPU brand string (best effort)
    if [[ "$os" == "Darwin" ]]; then
        CPU_BRAND="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
        RAM_GB="$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
    elif [[ "$os" == "Linux" ]]; then
        CPU_BRAND="$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs 2>/dev/null || echo unknown)"
        RAM_GB="$(( $(grep MemTotal /proc/meminfo | awk '{print $2}' 2>/dev/null || echo 0) / 1048576 ))"
    else
        CPU_BRAND="unknown"
        RAM_GB=0
    fi

    RUST_VERSION="$(rustc --version 2>/dev/null | awk '{print $2}' || echo unknown)"
    CARGO_VERSION="$(cargo --version 2>/dev/null | awk '{print $2}' || echo unknown)"
}

# ---------------------------------------------------------------------------
# Run Criterion benchmarks
# ---------------------------------------------------------------------------

run_criterion_benches() {
    log "Running Criterion latency benchmarks (cargo bench -p clawhdf5-agent)..."
    cd "$WORKSPACE_ROOT"
    cargo bench -p clawhdf5-agent --bench memory_bench -- --output-format json 2>/dev/null \
        || cargo bench -p clawhdf5-agent --bench memory_bench 2>&1 | tail -20
    log "Criterion benchmarks complete. Results in: $BENCH_OUTPUT_DIR"
}

# ---------------------------------------------------------------------------
# Parse Criterion JSON results (best-effort)
# ---------------------------------------------------------------------------

parse_criterion_results() {
    # Criterion saves JSON estimates in target/criterion/<bench_name>/estimates.json
    # We collect a subset of key results.
    local results=()
    if [[ -d "$BENCH_OUTPUT_DIR" ]]; then
        while IFS= read -r -d '' est_file; do
            bench_name="$(basename "$(dirname "$est_file")")"
            # Extract mean estimate in nanoseconds
            if command -v python3 &>/dev/null; then
                mean_ns="$(python3 -c "
import json, sys
try:
    d = json.load(open('$est_file'))
    print(d['mean']['point_estimate'])
except:
    print('null')
" 2>/dev/null)"
            else
                mean_ns="null"
            fi
            results+=("\"$bench_name\": $mean_ns")
        done < <(find "$BENCH_OUTPUT_DIR" -name "estimates.json" -print0 2>/dev/null)
    fi
    printf '%s\n' "${results[@]}"
}

# ---------------------------------------------------------------------------
# Run standalone bench binaries (--full mode)
# ---------------------------------------------------------------------------

run_standalone_benches() {
    log "Building standalone bench binaries..."
    cd "$WORKSPACE_ROOT"
    cargo build --release -p clawhdf5-bench 2>/dev/null

    log "Running footprint_bench..."
    FOOTPRINT_OUTPUT="$(cargo run --release -p clawhdf5-bench --bin footprint_bench 2>/dev/null || echo 'error')"

    log "Running memory_arena..."
    ARENA_OUTPUT="$(cargo run --release -p clawhdf5-bench --bin memory_arena 2>/dev/null || echo 'error')"

    log "Running consolidation_efficiency..."
    CONSOL_OUTPUT="$(cargo run --release -p clawhdf5-bench --bin consolidation_efficiency 2>/dev/null || echo 'error')"

    # LongMemEval (only if dataset exists)
    LME_JSON="${WORKSPACE_ROOT}/benchmarks/longmemeval/longmemeval_oracle.json"
    if [[ -f "$LME_JSON" ]]; then
        log "Running longmemeval_bench (500 questions, this may take a few minutes)..."
        LME_OUTPUT="$(cargo run --release -p clawhdf5-bench --bin longmemeval_bench -- "$LME_JSON" 2>/dev/null || echo 'error')"
    else
        LME_OUTPUT="dataset not found at $LME_JSON"
    fi
}

# ---------------------------------------------------------------------------
# Produce JSON output
# ---------------------------------------------------------------------------

emit_json() {
    cat <<JSON
{
  "benchmark_run": {
    "timestamp": "$TIMESTAMP",
    "platform": {
      "os": "$OS_NAME",
      "arch": "$ARCH_NAME",
      "cpu": "$(echo "$CPU_BRAND" | sed 's/"/\\"/g')",
      "ram_gb": $RAM_GB
    },
    "toolchain": {
      "rust": "$RUST_VERSION",
      "cargo": "$CARGO_VERSION"
    },
    "criterion_results_dir": "$BENCH_OUTPUT_DIR",
    "run_mode": "$([ "$RUN_FULL" -eq 1 ] && echo full || echo criterion_only)"
  }$(if [[ "$RUN_FULL" -eq 1 ]]; then cat <<FULLRESULTS
,
  "footprint_bench": $(echo "$FOOTPRINT_OUTPUT" | python3 -c "
import sys, json
lines = sys.stdin.read()
# Best-effort extract table rows
import re
rows = []
for m in re.finditer(r'(\d[\dKMG]+)\s+([\d.]+ [BKMG]+)\s+([\d.]+ [BKMG]+)\s+([\d.]+ [BKMG]+)\s+([\d.]+)x\s+([\d,]+ rec/s)', lines):
    rows.append({'records': m.group(1), 'file_size': m.group(2), 'raw_size': m.group(3), 'bytes_per_record': m.group(4), 'compression_ratio': float(m.group(5)), 'throughput': m.group(6)})
print(json.dumps(rows, indent=4))
" 2>/dev/null || echo '"(parsing error)"'),
  "memory_arena": $(echo "$ARENA_OUTPUT" | python3 -c "
import sys, json
lines = sys.stdin.read()
import re
rows = []
for m in re.finditer(r'([\w-]+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+) µs', lines):
    rows.append({'type': m.group(1), 'n': int(m.group(2)), 'hit_at_1': float(m.group(3)), 'hit_at_5': float(m.group(4)), 'hit_at_10': float(m.group(5)), 'mrr': float(m.group(6)), 'avg_latency_us': float(m.group(7))})
print(json.dumps(rows, indent=4))
" 2>/dev/null || echo '"(parsing error)"'),
  "longmemeval": $(echo "$LME_OUTPUT" | python3 -c "
import sys, json, re
text = sys.stdin.read()
m = re.search(r'\`\`\`json\s*(\{.*?\})\s*\`\`\`', text, re.DOTALL)
print(m.group(1) if m else '\"(not available)\"')
" 2>/dev/null || echo '"(not available)"')
FULLRESULTS
fi)
}
JSON
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "ClawhDF5 Cross-Platform Benchmark Runner"
    log "Workspace: $WORKSPACE_ROOT"

    detect_platform
    log "Platform: $OS_NAME / $ARCH_NAME / $CPU_BRAND"
    log "Toolchain: Rust $RUST_VERSION"

    run_criterion_benches

    if [[ "$RUN_FULL" -eq 1 ]]; then
        run_standalone_benches
    fi

    local json_output
    json_output="$(emit_json)"

    if [[ -n "$OUTPUT_FILE" ]]; then
        echo "$json_output" > "$OUTPUT_FILE"
        log "JSON results written to: $OUTPUT_FILE"
    else
        echo "$json_output"
    fi

    log "Done."
}

main "$@"
