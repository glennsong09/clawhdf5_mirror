# Migration Guide: OpenClaw sqlite-vec → clawhdf5

This guide walks through migrating an OpenClaw agent from its default
sqlite-vec + Markdown file memory to the `clawhdf5` HDF5 backend.

---

## Why migrate?

| Feature | sqlite-vec + Markdown | clawhdf5 |
|---------|----------------------|----------|
| Storage format | SQLite WAL + flat .md files | Single HDF5 binary file |
| Vector search | sqlite-vec (SQLite extension) | Pure-Rust SIMD (clawhdf5-accel) |
| Full-text search | External (FTS5 or plain string match) | Built-in BM25 |
| Hybrid search | Manual combination | Automatic RRF blend |
| Memory tiers | Flat | Working → Episodic → Semantic |
| Hebbian decay | Not built-in | Automatic activation weighting |
| Portability | SQLite binary required | Zero native deps (all Rust) |
| Crash recovery | SQLite WAL | clawhdf5 WAL |
| Compaction | Manual | Auto-threshold + session-end |
| Embedding dim change | New DB required | New file required (same) |

---

## Step 1: Install `@redclaw/clawhdf5`

```bash
npm install @redclaw/clawhdf5
```

Or, if building from the monorepo source:

```bash
npm install -g @napi-rs/cli
cd packages/clawhdf5-node
npm install
npm run build
```

---

## Step 2: Update your OpenClaw config

Change `backend` from `"sqlite-vec"` (or `"markdown"`) to `"clawhdf5"`:

```json
{
  "memory": {
    "backend": "clawhdf5",
    "clawhdf5": {
      "path": "./agent.brain",
      "embeddingDim": 768
    }
  }
}
```

See [openclaw-config.md](openclaw-config.md) for the full schema.

---

## Step 3: Run the one-time migration

clawhdf5 ships a migration helper that reads your existing Markdown memory
files and ingests them via `ingestMarkdown()`.

### Automated migration script

```typescript
import { ClawhdfMemory } from '@redclaw/clawhdf5';
import { readFileSync, readdirSync, statSync } from 'fs';
import { join, relative } from 'path';

async function migrate(
  memoryDir: string,
  brainPath: string,
  embeddingDim: number = 768,
): Promise<void> {
  const mem = ClawhdfMemory.create(brainPath, embeddingDim);

  // Walk all .md files under memoryDir
  function walk(dir: string): string[] {
    return readdirSync(dir).flatMap((entry) => {
      const full = join(dir, entry);
      return statSync(full).isDirectory() ? walk(full) : [full];
    });
  }

  const files = walk(memoryDir).filter((f) => f.endsWith('.md'));
  let totalSections = 0;

  for (const file of files) {
    const content = readFileSync(file, 'utf8');
    const relPath = relative(process.cwd(), file);
    const count = mem.ingestMarkdown(relPath, content);
    console.log(`  ${relPath}: ${count} sections`);
    totalSections += count;
  }

  // Force WAL merge after bulk import
  mem.flushWal();

  console.log(`\nMigration complete: ${files.length} files, ${totalSections} sections`);
  const s = mem.stats();
  console.log(`  Total records:    ${s.totalRecords}`);
  console.log(`  File size:        ${(s.fileSizeBytes / 1024).toFixed(1)} KB`);
}

// Usage
migrate('./memory', './agent.brain', 768).catch(console.error);
```

### What the migration does

1. Walks all `.md` files under your memory directory.
2. Parses each file into sections using the same `MarkdownParser` used by
   OpenClaw (splits on ATX headings `#`, `##`, `###`, …).
3. Stores each section as a separate record in the HDF5 file with the file
   path as the `source_channel` (e.g. `memory/user.md::Goals`).
4. Flushes the WAL to merge everything into the `.brain` file.

After migration, **the original `.md` files are not modified or deleted**.
You can keep them as a backup or remove them once you have verified the
migrated data.

---

## Step 4: Verify

```typescript
import { ClawhdfMemory } from '@redclaw/clawhdf5';

const mem = ClawhdfMemory.open('./agent.brain');
const s = mem.stats();
console.log('Records after migration:', s.totalRecords);

// Spot-check: retrieve a known path
const userMd = mem.get('memory/MEMORY.md');
console.log(userMd?.slice(0, 200));

// Round-trip a file back to Markdown
const exported = mem.exportMarkdown('memory/MEMORY.md');
console.log(exported.slice(0, 500));
```

---

## Step 5: Update agent code

If your agent code reads memory files directly from disk, update it to use
the clawhdf5 API instead:

**Before (sqlite-vec + file reads):**
```typescript
const content = readFileSync('memory/user.md', 'utf8');
const sections = parseMarkdown(content);
const results = await vectorSearch(query, sections, k);
```

**After (clawhdf5):**
```typescript
import { ClawhdfMemory } from '@redclaw/clawhdf5';

const mem = ClawhdfMemory.openOrCreate('./agent.brain', 768);
const embedding = await embed(query); // your embedding function
const results = mem.search(query, new Float32Array(embedding), k);
```

---

## Step 6: Session lifecycle hooks

Add compaction at session end for best long-term memory health:

```typescript
// At the start of your agent process
const mem = ClawhdfMemory.openOrCreate('./agent.brain', 768);

// ... agent runs ...

// At the end of each session
mem.tickSession();                                 // decay activation weights
const stats = mem.runConsolidation(Date.now() / 1000);  // promote memories
console.log('[memory] consolidation:', stats);
```

---

## Rollback

If you need to roll back to sqlite-vec:

1. Change `memory.backend` back to `"sqlite-vec"` in your config.
2. The original `.md` files are unchanged (if you kept them).
3. Delete `agent.brain` (and `agent.brain.wal` if present).

---

## Troubleshooting

### `Error: no records found for path: memory/user.md`
The path passed to `get()` or `exportMarkdown()` must exactly match the
relative path used during `ingestMarkdown()`. Check for leading `./`
differences.

### Memory is empty after reopening
Make sure `flushWal()` was called after bulk writes. Without it, entries
remain in the WAL and may be lost if the process exits abnormally.

### Embedding dimension mismatch
The `embeddingDim` passed to `create()` cannot be changed after the file is
created. If you switch embedding models, create a new `.brain` file and
re-run the migration script.
