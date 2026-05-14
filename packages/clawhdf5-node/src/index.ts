/**
 * @redclaw/clawhdf5 — Node.js bindings for the clawhdf5 HDF5-backed agent memory system.
 *
 * Re-exports the napi-rs native addon with full TypeScript types.
 *
 * @example
 * ```ts
 * import { ClawhdfMemory } from '@redclaw/clawhdf5';
 *
 * const mem = ClawhdfMemory.openOrCreate('./agent.brain', 768);
 *
 * // Ingest an existing MEMORY.md
 * const markdown = require('fs').readFileSync('./memory/MEMORY.md', 'utf8');
 * mem.ingestMarkdown('memory/MEMORY.md', markdown);
 *
 * // Search
 * const embedding = new Float32Array(768); // supply real embedding
 * const results = mem.search('rust programming', embedding, 5);
 * console.log(results);
 *
 * // Compact at session end
 * const stats = mem.runConsolidation(Date.now() / 1000);
 * console.log(`Working: ${stats.workingCount}, Episodic: ${stats.episodicCount}`);
 * ```
 */

// The native .node binary is built by napi-rs into this package directory.
// When running from source (before build), this import will fail — run
// `npm run build` first.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const native = require('../clawhdf5.node') as NativeModule;

// ─────────────────────────────────────────────────────────────────────────────
// Type definitions (mirroring the #[napi(object)] structs in Rust)
// ─────────────────────────────────────────────────────────────────────────────

/** A single result returned from a memory search. */
export interface MemorySearchResult {
  /** Text content of the matching memory record. */
  text: string;
  /** Relevance score — higher is more relevant. */
  score: number;
  /** Source file / section path the record originated from. */
  path: string;
  /** Optional `[start, end]` line range within the source file. */
  lineRange?: [number, number];
  /** Unix-epoch timestamp of the record, if available. */
  timestamp?: number;
  /** Human-readable source description (channel / file type). */
  source: string;
}

/** Aggregate statistics about the memory store. */
export interface BackendStats {
  /** Number of active (non-deleted) records. */
  totalRecords: number;
  /** Number of records that carry a non-empty embedding vector. */
  totalEmbeddings: number;
  /** On-disk size of the .h5 file in bytes. */
  fileSizeBytes: number;
  /** Modalities present in the store (e.g. `["text"]`). */
  modalities: string[];
  /** Unix-epoch seconds of the most recently stored record, if any. */
  lastUpdated?: number;
}

/** Statistics for the ephemeral in-memory working memory tier. */
export interface EphemeralStats {
  /** Number of entries currently stored. */
  totalEntries: number;
  /** Total bytes used by stored values. */
  totalBytes: number;
  /** Number of entries removed by TTL expiry in last cleanup. */
  expiredCount: number;
  /** Number of entries evicted for capacity in last cleanup. */
  evictedCount: number;
  /** Age in seconds of the oldest entry. */
  oldestEntryAgeSecs: number;
  /** Total successful gets. */
  hitCount: number;
  /** Total failed gets (not found or expired). */
  missCount: number;
}

/** Per-tier counts and running totals from one consolidation cycle. */
export interface ConsolidationStats {
  /** Records remaining in the Working tier. */
  workingCount: number;
  /** Records in the Episodic tier. */
  episodicCount: number;
  /** Records in the Semantic tier. */
  semanticCount: number;
  /** Cumulative records evicted (capacity overflow / decay). */
  totalEvictions: number;
  /** Cumulative records promoted between tiers. */
  totalPromotions: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal native module shape
// ─────────────────────────────────────────────────────────────────────────────

interface NativeMemorySearchResult {
  text: string;
  score: number;
  path: string;
  line_range?: number[];
  timestamp?: number;
  source: string;
}

interface NativeBackendStats {
  total_records: number;
  total_embeddings: number;
  file_size_bytes: number;
  modalities: string[];
  last_updated?: number;
}

interface NativeConsolidationStats {
  working_count: number;
  episodic_count: number;
  semantic_count: number;
  total_evictions: number;
  total_promotions: number;
}

interface NativeEphemeralStats {
  total_entries: number;
  total_bytes: number;
  expired_count: number;
  evicted_count: number;
  oldest_entry_age_secs: number;
  hit_count: number;
  miss_count: number;
}

interface NativeClawhdfMemory {
  search(queryText: string, queryEmbedding: Float32Array, k: number): NativeMemorySearchResult[];
  get(path: string, fromLine?: number, numLines?: number): string | null;
  write(path: string, content: string): void;
  ingestMarkdown(path: string, content: string): number;
  exportMarkdown(path: string): string;
  stats(): NativeBackendStats;
  compact(): number;
  tickSession(): void;
  flushWal(): void;
  runConsolidation(nowSecs: number): NativeConsolidationStats;
  walPendingCount(): number;
  enableEphemeral(maxEntries?: number, defaultTtlSecs?: number): void;
  ephemeralSet(key: string, value: string, ttlSecs?: number): void;
  ephemeralGet(key: string): string | null;
  ephemeralDelete(key: string): boolean;
  ephemeralStats(): NativeEphemeralStats | null;
  promoteEphemeral(minAccessCount?: number): number;
}

interface NativeModule {
  ClawhdfMemory: {
    create(path: string, embeddingDim: number): NativeClawhdfMemory;
    open(path: string): NativeClawhdfMemory;
    openOrCreate(path: string, embeddingDim: number): NativeClawhdfMemory;
    new(): never; // not a regular constructor
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Public wrapper class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * HDF5-backed memory store for AI agents.
 *
 * Wraps the native Rust implementation via napi-rs.  Use the static factory
 * methods (`create`, `open`, `openOrCreate`) rather than `new`.
 */
export class ClawhdfMemory {
  private readonly _inner: NativeClawhdfMemory;

  private constructor(inner: NativeClawhdfMemory) {
    this._inner = inner;
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  /**
   * Create a new HDF5 memory store at `path`.
   *
   * @param path  Filesystem path for the `.brain` / `.h5` file.
   * @param embeddingDim  Dimension of embedding vectors (must stay constant).
   */
  static create(path: string, embeddingDim: number): ClawhdfMemory {
    return new ClawhdfMemory(native.ClawhdfMemory.create(path, embeddingDim));
  }

  /**
   * Open an existing HDF5 memory store and replay the WAL if present.
   *
   * @param path  Path to an existing `.brain` / `.h5` file.
   */
  static open(path: string): ClawhdfMemory {
    return new ClawhdfMemory(native.ClawhdfMemory.open(path));
  }

  /**
   * Open the store at `path` if it exists; otherwise create it.
   *
   * This is the recommended entry point for most agent use cases.
   */
  static openOrCreate(path: string, embeddingDim: number): ClawhdfMemory {
    return new ClawhdfMemory(native.ClawhdfMemory.openOrCreate(path, embeddingDim));
  }

  // ── MemoryBackend methods ──────────────────────────────────────────────────

  /**
   * Hybrid vector + BM25 search.
   *
   * @param queryText       Natural-language query string (used for BM25).
   * @param queryEmbedding  Float32Array of length `embeddingDim` (used for
   *                        vector similarity).  Pass an empty array to skip
   *                        vector search.
   * @param k               Maximum number of results to return.
   */
  search(queryText: string, queryEmbedding: Float32Array, k: number): MemorySearchResult[] {
    return this._inner.search(queryText, queryEmbedding, k).map((r) => ({
      text: r.text,
      score: r.score,
      path: r.path,
      lineRange: r.line_range ? [r.line_range[0], r.line_range[1]] : undefined,
      timestamp: r.timestamp,
      source: r.source,
    }));
  }

  /**
   * Retrieve raw content stored at `path`.
   *
   * @param path      The path used when writing/ingesting.
   * @param fromLine  0-indexed first line to return (optional).
   * @param numLines  Maximum number of lines to return (optional).
   * @returns         Content string, or `null` if the path is not found.
   */
  get(path: string, fromLine?: number, numLines?: number): string | null {
    return this._inner.get(path, fromLine, numLines);
  }

  /**
   * Store raw `content` at `path`.
   */
  write(path: string, content: string): void {
    this._inner.write(path, content);
  }

  /**
   * Parse `content` as Markdown, split into sections, and ingest each section.
   *
   * @returns Number of sections ingested.
   */
  ingestMarkdown(path: string, content: string): number {
    return this._inner.ingestMarkdown(path, content);
  }

  /**
   * Reconstruct all stored sections for `path` as a Markdown string.
   */
  exportMarkdown(path: string): string {
    return this._inner.exportMarkdown(path);
  }

  /**
   * Return aggregate statistics about the memory store.
   */
  stats(): BackendStats {
    const s = this._inner.stats();
    return {
      totalRecords: s.total_records,
      totalEmbeddings: s.total_embeddings,
      fileSizeBytes: s.file_size_bytes,
      modalities: s.modalities,
      lastUpdated: s.last_updated,
    };
  }

  // ── Consolidation hooks ────────────────────────────────────────────────────

  /**
   * Remove tombstoned entries from the store.
   *
   * @returns Number of entries permanently removed.
   */
  compact(): number {
    return this._inner.compact();
  }

  /**
   * Apply one Hebbian decay tick to all activation weights and flush to disk.
   *
   * Call this at the end of each agent session to let infrequently-accessed
   * memories naturally decay toward eviction.
   */
  tickSession(): void {
    this._inner.tickSession();
  }

  /**
   * Force a WAL merge: flush the .h5 file and truncate the WAL log.
   *
   * Useful before taking a snapshot or before a long idle period.
   */
  flushWal(): void {
    this._inner.flushWal();
  }

  /**
   * Run a full hippocampal consolidation cycle.
   *
   * Snapshots live cache records into the three-tier engine (Working →
   * Episodic → Semantic) and returns per-tier counts plus eviction/promotion
   * totals.
   *
   * @param nowSecs  Current Unix epoch time in seconds (`Date.now() / 1000`).
   */
  runConsolidation(nowSecs: number): ConsolidationStats {
    const s = this._inner.runConsolidation(nowSecs);
    return {
      workingCount: s.working_count,
      episodicCount: s.episodic_count,
      semanticCount: s.semantic_count,
      totalEvictions: s.total_evictions,
      totalPromotions: s.total_promotions,
    };
  }

  // ── Metadata ───────────────────────────────────────────────────────────────

  /**
   * Number of pending WAL entries waiting to be merged into the .h5 file.
   * Returns 0 if WAL is disabled.
   */
  walPendingCount(): number {
    return this._inner.walPendingCount();
  }

  // ── Ephemeral tier ──────────────────────────────────────────────────────────

  /**
   * Enable the ephemeral (in-memory only) working memory tier.
   *
   * @param maxEntries     Capacity before LFU eviction (default 10 000).
   * @param defaultTtlSecs Default TTL in seconds (default 3600).
   */
  enableEphemeral(maxEntries?: number, defaultTtlSecs?: number): void {
    this._inner.enableEphemeral(maxEntries, defaultTtlSecs);
  }

  /**
   * Store a value in ephemeral memory. Never written to disk.
   *
   * @param key     Lookup key.
   * @param value   String value to store.
   * @param ttlSecs Optional TTL override in seconds.
   */
  ephemeralSet(key: string, value: string, ttlSecs?: number): void {
    this._inner.ephemeralSet(key, value, ttlSecs);
  }

  /**
   * Retrieve a value from ephemeral memory.
   *
   * @returns The stored string, or `null` if absent / expired / tier disabled.
   */
  ephemeralGet(key: string): string | null {
    return this._inner.ephemeralGet(key);
  }

  /**
   * Delete a key from ephemeral memory.
   *
   * @returns `true` if the key existed and was removed.
   */
  ephemeralDelete(key: string): boolean {
    return this._inner.ephemeralDelete(key);
  }

  /**
   * Return statistics for the ephemeral tier, or `null` if not enabled.
   */
  ephemeralStats(): EphemeralStats | null {
    const s = this._inner.ephemeralStats();
    if (!s) return null;
    return {
      totalEntries: s.total_entries,
      totalBytes: s.total_bytes,
      expiredCount: s.expired_count,
      evictedCount: s.evicted_count,
      oldestEntryAgeSecs: s.oldest_entry_age_secs,
      hitCount: s.hit_count,
      missCount: s.miss_count,
    };
  }

  /**
   * Promote frequently-accessed ephemeral entries to persistent HDF5 storage.
   *
   * @param minAccessCount Minimum access count to qualify (default 3).
   * @returns              Number of entries promoted.
   */
  promoteEphemeral(minAccessCount?: number): number {
    return this._inner.promoteEphemeral(minAccessCount);
  }
}
