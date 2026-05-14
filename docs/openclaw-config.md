# OpenClaw × clawhdf5 Configuration Reference

This document describes the full configuration schema for integrating
`clawhdf5` as the memory backend in an OpenClaw agent gateway.

---

## Minimal example

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

---

## Full schema

```json
{
  "memory": {
    "backend": "clawhdf5",
    "clawhdf5": {
      "path": "./agent.brain",
      "embeddingDim": 768,
      "walEnabled": true,
      "walMaxEntries": 500,
      "consolidation": {
        "workingCapacity": 100,
        "episodicCapacity": 10000,
        "episodicHalfLifeDays": 7,
        "semanticHalfLifeDays": 30,
        "promotionThreshold": 0.6,
        "semanticAccessThreshold": 10
      },
      "compaction": {
        "autoCompactThreshold": 0.3,
        "tickOnSessionEnd": true,
        "consolidateOnCompaction": true
      }
    }
  }
}
```

---

## Field reference

### Top level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `memory.backend` | `string` | `"clawhdf5"` | Must be `"clawhdf5"` to activate this backend |

### `clawhdf5`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `string` | `"./agent.brain"` | Filesystem path for the `.brain` (HDF5) file. Relative to the OpenClaw working directory. |
| `embeddingDim` | `number` | `768` | Dimension of the embedding vectors. Must match the embedder model. Common values: `384` (MiniLM), `768` (nomic-embed-text, BGE-base), `1536` (OpenAI text-embedding-3-small). |
| `walEnabled` | `boolean` | `true` | Enable the Write-Ahead Log for crash recovery. Disable only on read-only stores or when crash safety is not required. |
| `walMaxEntries` | `number` | `500` | Number of WAL entries to accumulate before an automatic merge to the .h5 file. Lower values = more frequent flushes (safer, slightly slower). |

### `clawhdf5.consolidation`

Controls the hippocampal three-tier memory engine (Working → Episodic →
Semantic).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `workingCapacity` | `number` | `100` | Maximum records in the Working tier before lowest-decay entries are evicted. |
| `episodicCapacity` | `number` | `10000` | Maximum records in the Episodic tier. |
| `episodicHalfLifeDays` | `number` | `7` | Half-life (in days) for exponential decay of Episodic records. Records not accessed within roughly one half-life drop in importance. |
| `semanticHalfLifeDays` | `number` | `30` | Half-life for Semantic records. Longer than Episodic — semantic knowledge decays slowly. |
| `promotionThreshold` | `number` | `0.6` | Importance score (0–1) above which a Working record is promoted to the Episodic tier. Higher = more selective. |
| `semanticAccessThreshold` | `number` | `10` | Minimum access count for an Episodic record to be promoted to Semantic. |

### `clawhdf5.compaction`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `autoCompactThreshold` | `number` | `0.3` | Fraction of tombstoned records (0–1) that triggers automatic compaction. `0.3` = compact when 30% of records are deleted. Set to `0` to disable auto-compact. |
| `tickOnSessionEnd` | `boolean` | `true` | Run `tickSession()` (Hebbian decay) automatically when the agent session closes. |
| `consolidateOnCompaction` | `boolean` | `true` | Run the hippocampal consolidation engine after each compaction cycle. |

---

## Embedder compatibility

The `embeddingDim` must remain constant for the lifetime of a `.brain` file.
Mixing embedding models in the same file is not supported.

| Embedder | `embeddingDim` |
|----------|---------------|
| `all-MiniLM-L6-v2` | `384` |
| `nomic-embed-text` | `768` |
| `BGE-base-en-v1.5` | `768` |
| `OpenAI text-embedding-3-small` | `1536` |
| `OpenAI text-embedding-3-large` | `3072` |

---

## OpenClaw integration code

```typescript
import { ClawhdfMemory } from '@redclaw/clawhdf5';

// Load config from your OpenClaw config file
const cfg = loadConfig(); // your config loading logic

const mem = ClawhdfMemory.openOrCreate(
  cfg.memory.clawhdf5.path,
  cfg.memory.clawhdf5.embeddingDim ?? 768,
);

// On session end
if (cfg.memory.clawhdf5.compaction?.tickOnSessionEnd) {
  mem.tickSession();
}
if (cfg.memory.clawhdf5.compaction?.consolidateOnCompaction) {
  const stats = mem.runConsolidation(Date.now() / 1000);
  console.log('[clawhdf5] consolidation:', stats);
}
```

---

## Environment variables

The following environment variables override config file values when set:

| Variable | Overrides |
|----------|-----------|
| `CLAWHDF5_PATH` | `clawhdf5.path` |
| `CLAWHDF5_EMBEDDING_DIM` | `clawhdf5.embeddingDim` |
| `CLAWHDF5_WAL_ENABLED` | `clawhdf5.walEnabled` (`"true"` / `"false"`) |
