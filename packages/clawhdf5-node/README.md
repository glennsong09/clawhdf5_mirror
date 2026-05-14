# @redclaw/clawhdf5

Node.js (TypeScript) bindings for [clawhdf5](../../README.md) — a pure-Rust
HDF5-backed agent memory system with hippocampal consolidation.

Built with [napi-rs](https://napi.rs).

## Installation

```bash
npm install @redclaw/clawhdf5
```

Pre-built binaries are published for:

| Platform | Architecture |
|----------|-------------|
| Linux (glibc) | x64, aarch64 |
| macOS | x64, aarch64 (Apple Silicon) |
| Windows | x64 |

## Quick start

```ts
import { ClawhdfMemory } from '@redclaw/clawhdf5';

// Open or create a memory store (embedding dim must match your embedder)
const mem = ClawhdfMemory.openOrCreate('./agent.brain', 768);

// Ingest your existing MEMORY.md files
import { readFileSync } from 'fs';
const md = readFileSync('./memory/MEMORY.md', 'utf8');
mem.ingestMarkdown('memory/MEMORY.md', md);

// Write raw content
mem.write('memory/session.md', '# Session\n\nStarted task X.');

// Search
const embedding = new Float32Array(768); // supply a real embedding here
const results = mem.search('task X progress', embedding, 5);
console.log(results[0].text);

// Run consolidation at session end
const stats = mem.runConsolidation(Date.now() / 1000);
console.log(`Episodic: ${stats.episodicCount}, Semantic: ${stats.semanticCount}`);
```

## Building from source

Requirements:

- Rust (latest stable, edition 2024)
- Node.js ≥ 16
- `@napi-rs/cli` (`npm install -g @napi-rs/cli`)

```bash
# From the repo root
cd packages/clawhdf5-node
npm install
npm run build       # release build
npm run build:debug # debug build (faster, no optimisations)
```

## API reference

See [`src/index.ts`](src/index.ts) for full JSDoc-annotated types.

### `ClawhdfMemory`

**Factory methods (use instead of `new`):**

| Method | Description |
|--------|-------------|
| `ClawhdfMemory.create(path, embeddingDim)` | Create a new memory store |
| `ClawhdfMemory.open(path)` | Open existing store, replay WAL |
| `ClawhdfMemory.openOrCreate(path, embeddingDim)` | Open or create |

**Instance methods:**

| Method | Description |
|--------|-------------|
| `search(queryText, queryEmbedding, k)` | Hybrid BM25 + vector search |
| `get(path, fromLine?, numLines?)` | Retrieve raw content |
| `write(path, content)` | Store raw content |
| `ingestMarkdown(path, content)` | Parse Markdown and ingest sections |
| `exportMarkdown(path)` | Reconstruct Markdown from stored sections |
| `stats()` | Aggregate store statistics |
| `compact()` | Remove tombstoned entries |
| `tickSession()` | Hebbian decay tick |
| `flushWal()` | Force WAL merge to disk |
| `runConsolidation(nowSecs)` | Full hippocampal consolidation cycle |
| `walPendingCount()` | Pending WAL entry count |

## License

MIT
