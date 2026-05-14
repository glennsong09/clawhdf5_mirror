# OpenClaw × clawhdf5 Integration

clawhdf5 provides a drop-in HDF5-backed memory backend for the
[OpenClaw](https://github.com/redclawsystems/openclaw) agent gateway.
This document covers architecture, the full Node.js API reference, and code
examples for common operations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  OpenClaw (Node.js/TypeScript)                              │
│                                                             │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │  Agent runtime  │───▶│  @redclaw/clawhdf5 (Node.js) │   │
│  └─────────────────┘    │  TypeScript wrapper          │   │
│                         └──────────────┬─────────────┘    │
│                                        │ napi-rs FFI        │
└────────────────────────────────────────┼────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────┐
│  clawhdf5-napi  (Rust, cdylib)                              │
│                                                             │
│  ClawhdfMemory  ──▶  ClawhdfBackend  ──▶  HDF5Memory       │
│                       MemoryBackend        ├─ MemoryCache   │
│                       trait impl          ├─ WalFile        │
│                                           ├─ SessionCache   │
│                                           └─ KnowledgeCache │
│                                                             │
│  ConsolidationEngine (hippocampal tiers)                    │
│   Working (100) ──▶ Episodic (10k) ──▶ Semantic (∞)        │
└─────────────────────────────────────────────────────────────┘
                                         │
                        ┌────────────────▼────────────┐
                        │  agent.brain  (HDF5 file)    │
                        │  agent.brain.wal  (WAL log)  │
                        └─────────────────────────────┘
```

### Key design decisions

- **Single file**: everything lives in one `.brain` HDF5 file (+ WAL sidecar).
- **In-memory cache**: the full embedding matrix and chunk list are loaded into RAM for fast search.
- **Hybrid search**: vector similarity (70%) and BM25 full-text (30%) are blended with Reciprocal Rank Fusion (RRF), then re-ranked by Hebbian activation weight and temporal recency.
- **Hippocampal tiers**: records are classified as Working, Episodic, or Semantic based on importance and access frequency. Tier promotion and eviction happen during `runConsolidation()`.
- **WAL**: writes are journaled before hitting the .h5 file. On crash, the WAL is replayed at next open.

---

## Installation

```bash
npm install @redclaw/clawhdf5
```

See [packages/clawhdf5-node/README.md](../packages/clawhdf5-node/README.md)
for build-from-source instructions.

---

## Node.js API reference

### `ClawhdfMemory` (class)

All instance methods are synchronous.  The native Rust code is single-threaded
on the Node.js side; do **not** share a `ClawhdfMemory` instance across Worker
threads without external locking.

---

#### Static factory methods

##### `ClawhdfMemory.create(path: string, embeddingDim: number): ClawhdfMemory`

Create a new `.brain` file. Throws if the file already exists.

```typescript
const mem = ClawhdfMemory.create('./agent.brain', 768);
```

##### `ClawhdfMemory.open(path: string): ClawhdfMemory`

Open an existing file. Replays the WAL automatically.

```typescript
const mem = ClawhdfMemory.open('./agent.brain');
```

##### `ClawhdfMemory.openOrCreate(path: string, embeddingDim: number): ClawhdfMemory`

**Recommended entry point.** Opens if the file exists, otherwise creates it.

```typescript
const mem = ClawhdfMemory.openOrCreate('./agent.brain', 768);
```

---

#### `search(queryText, queryEmbedding, k): MemorySearchResult[]`

Hybrid BM25 + vector search.

```typescript
const embedding = new Float32Array(await embed(query));
const results = mem.search(query, embedding, 10);
for (const r of results) {
  console.log(r.score.toFixed(3), r.path, r.text.slice(0, 80));
}
```

Pass an empty `Float32Array` to use BM25 only (no vector similarity).

**Parameters:**
- `queryText: string` — used for BM25 term matching
- `queryEmbedding: Float32Array` — dense vector of length `embeddingDim`
- `k: number` — maximum results to return

**Returns:** `MemorySearchResult[]`

---

#### `get(path, fromLine?, numLines?): string | null`

Retrieve stored content by path.

```typescript
const md = mem.get('memory/user.md');        // all content
const lines = mem.get('memory/user.md', 5, 10);  // lines 5–14
const section = mem.get('memory/user.md::Goals'); // specific section
```

Section sub-paths use the `::heading` suffix produced by `ingestMarkdown`.

---

#### `write(path, content): void`

Store raw content at `path`.

```typescript
mem.write('memory/session.md', '# Session\n\nWorking on task X.');
```

---

#### `ingestMarkdown(path, content): number`

Parse `content` as Markdown, split on ATX headings, and store each section
separately.  Returns the number of sections ingested.

```typescript
import { readFileSync } from 'fs';
const md = readFileSync('./memory/MEMORY.md', 'utf8');
const count = mem.ingestMarkdown('memory/MEMORY.md', md);
console.log(`Ingested ${count} sections`);
```

Sections are addressable as `memory/MEMORY.md::HeadingName`.

---

#### `exportMarkdown(path): string`

Reconstruct stored sections for `path` back into a Markdown string.

```typescript
const md = mem.exportMarkdown('memory/MEMORY.md');
writeFileSync('./memory/MEMORY.md', md);
```

---

#### `stats(): BackendStats`

Return aggregate statistics.

```typescript
const s = mem.stats();
console.log(`Records: ${s.totalRecords}, Size: ${s.fileSizeBytes} bytes`);
```

---

#### `compact(): number`

Remove tombstoned records from the store. Returns count removed.

---

#### `tickSession(): void`

Apply Hebbian decay to all activation weights. Call at session end.

---

#### `flushWal(): void`

Force a WAL merge: flush `.h5` and truncate the WAL log.

---

#### `runConsolidation(nowSecs: number): ConsolidationStats`

Run one full hippocampal consolidation cycle.

```typescript
const stats = mem.runConsolidation(Date.now() / 1000);
console.log(stats);
// { workingCount: 42, episodicCount: 310, semanticCount: 5,
//   totalEvictions: 0, totalPromotions: 7 }
```

---

#### `walPendingCount(): number`

Number of pending WAL entries (0 if WAL is disabled).

---

### Type reference

```typescript
interface MemorySearchResult {
  text: string;
  score: number;            // 0–1, higher = more relevant
  path: string;             // source file path
  lineRange?: [number, number];
  timestamp?: number;       // Unix epoch seconds
  source: string;
}

interface BackendStats {
  totalRecords: number;
  totalEmbeddings: number;
  fileSizeBytes: number;
  modalities: string[];     // e.g. ["text"]
  lastUpdated?: number;     // Unix epoch seconds
}

interface ConsolidationStats {
  workingCount: number;
  episodicCount: number;
  semanticCount: number;
  totalEvictions: number;
  totalPromotions: number;
}
```

---

## Common patterns

### Session lifecycle

```typescript
import { ClawhdfMemory } from '@redclaw/clawhdf5';

const mem = ClawhdfMemory.openOrCreate('./agent.brain', 768);

// --- agent session runs ---

// On session end: decay + consolidate
mem.tickSession();
const consolidationStats = mem.runConsolidation(Date.now() / 1000);
console.log('[memory] consolidation:', consolidationStats);
```

### Ingest all memory files at startup

```typescript
import { readdirSync, readFileSync, statSync } from 'fs';
import { join, relative } from 'path';

function ingestDirectory(mem: ClawhdfMemory, dir: string): void {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      ingestDirectory(mem, full);
    } else if (entry.endsWith('.md')) {
      const content = readFileSync(full, 'utf8');
      const path = relative(process.cwd(), full);
      mem.ingestMarkdown(path, content);
    }
  }
  mem.flushWal();
}

const mem = ClawhdfMemory.openOrCreate('./agent.brain', 768);
ingestDirectory(mem, './memory');
```

### Search with real embeddings

```typescript
import OpenAI from 'openai';
import { ClawhdfMemory } from '@redclaw/clawhdf5';

const ai = new OpenAI();
const mem = ClawhdfMemory.openOrCreate('./agent.brain', 1536);

async function searchMemory(query: string, k = 5) {
  const resp = await ai.embeddings.create({
    model: 'text-embedding-3-small',
    input: query,
  });
  const embedding = new Float32Array(resp.data[0].embedding);
  return mem.search(query, embedding, k);
}
```

---

## Error handling

All methods that can fail throw a `NapiError` (a standard JS `Error` subclass)
with the Rust error message as `message`.

```typescript
try {
  const md = mem.exportMarkdown('nonexistent.md');
} catch (e) {
  console.error('Export failed:', (e as Error).message);
  // "no records found for path: nonexistent.md"
}
```

---

## See also

- [openclaw-config.md](openclaw-config.md) — Full configuration schema
- [migration-guide.md](migration-guide.md) — Migrating from sqlite-vec
- [packages/clawhdf5-node/README.md](../packages/clawhdf5-node/README.md) — Build instructions
- [BENCHMARKS.md](../BENCHMARKS.md) — Performance results
