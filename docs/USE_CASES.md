# ClawhDF5 Use Cases

Real-world scenarios where ClawhDF5 solves problems that other approaches can't.

---

## 1. Personal AI Assistant

**Scenario:** You run a personal AI assistant (like OpenClaw, MemGPT, or a custom agent) that accumulates knowledge about you over weeks and months — preferences, decisions, context from past conversations.

**Problem:** Most assistants either forget everything between sessions (stateless) or dump everything into a growing context window (expensive, eventually hits token limits).

**ClawhDF5 solution:**

```
conversation → embedding → save to agent.h5
                              │
                ┌─────────────┤
                │             │
           Working        Knowledge
           Memory          Graph
         (recent)       (entities)
                │             │
           consolidate    traverse
                │             │
           Episodic       "Who is
           Memory         Alice's
         (important)      manager?"
                │
           Semantic
           Memory
        (core facts)
```

- **Daily conversations** enter Working memory (bounded, auto-evicts old/trivial stuff)
- **Important facts** promote to Episodic ("User got promoted to VP on March 5th")
- **Core preferences** solidify in Semantic ("User is vegan, lives in SF, uses dark mode")
- **Entity tracking** via knowledge graph ("Alice → manages → Bob", "User → works_at → Acme")
- **One file** — back it up, move it to a new machine, it travels with the agent

**What you'd need without ClawhDF5:** SQLite for structured data + Pinecone for vectors + a separate entity store + custom consolidation logic + Markdown files + glue code.

---

## 2. OpenClaw Memory Upgrade

**Scenario:** You run OpenClaw and the default Markdown + sqlite-vec memory works OK for simple recall but falls short on complex queries like "what did we decide about the deployment architecture last Tuesday?" or "who's responsible for the billing system?"

**Problem:** Markdown files have no semantic structure. sqlite-vec does flat vector search — no keyword fusion, no re-ranking, no temporal reasoning, no knowledge graph.

**ClawhDF5 solution:**

```bash
# Migrate existing memories
clawhdf5 --path memory.h5 create --agent-id openclaw --dim 384

# Import your MEMORY.md and daily logs
# (programmatically via ClawhdfBackend::ingest_markdown)
```

Then in your OpenClaw config (future):
```json
{
  "memory": {
    "backend": "clawhdf5",
    "path": "~/.openclaw/agents/main/memory.h5"
  }
}
```

**What changes:**
- "What did we discuss last Tuesday?" → temporal index finds the session, returns memories from that time range
- "Who owns the billing system?" → knowledge graph traversal: billing_system → owned_by → Alice
- "Preferences about deployment" → hybrid search (vector + BM25) finds relevant memories even with different wording
- Bad search results get filtered out by confidence rejection instead of confusing the agent

---

## 3. Multi-Agent System

**Scenario:** You have multiple specialized agents — a coding agent, a research agent, a scheduling agent — that need to share knowledge without sharing everything.

**Problem:** Giving agents a shared database creates security issues (coding agent shouldn't see personal data) and conflicts (agents overwrite each other's memories).

**ClawhDF5 solution:**

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Coding Agent │  │Research Agent│  │Schedule Agent│
│  coding.h5   │  │ research.h5  │  │ schedule.h5  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┘                 │
                │                          │
        ┌───────▼────────┐                 │
        │ Shared KG only │◄────────────────┘
        │  (export/import)│
        └────────────────┘
```

- Each agent has its own `.h5` file (full isolation)
- Knowledge graph entities/relations can be exported and imported between agents
- **Source isolation** in the provenance system prevents user-sourced memories from contaminating system memories within a single agent
- **Anomaly detection** catches if one agent is writing suspiciously (injection attack via tool output)

---

## 4. Edge / Embedded AI

**Scenario:** You're building an AI agent that runs on a Raspberry Pi, phone, or embedded device with limited resources. No cloud database. No internet for vector DB queries.

**Problem:** Most memory solutions require a server (Pinecone, Qdrant) or heavy dependencies (Python, CUDA).

**ClawhDF5 solution:**

- **Pure Rust** — compiles to a single static binary, no C dependencies
- **Single file** — all memory in one `.h5` file, no database server
- **Small footprint** — the agent crate adds ~2MB to your binary
- **ARM support** — runs on ARM64 (Raspberry Pi, phones) natively
- **Android bridge** — `clawhdf5-android` provides JNI bindings for Android apps
- **IVF-PQ** for ANN search keeps latency under 1.2ms even at 100K vectors on modest hardware
- **WAL** for crash safety — if the device loses power, no data corruption

```rust
// Same API whether you're on a server or a Pi
let config = MemoryConfig::new("/data/agent.h5", "edge-agent", 384);
let mut memory = HDF5Memory::create(config)?;
```

---

## 5. Scientific Data + AI Memory

**Scenario:** You work with HDF5 files (common in physics, climate science, genomics) and want to add AI-powered search over your datasets.

**Problem:** Existing HDF5 libraries (h5py, HDF5 C library) don't have vector search. You'd need a separate tool.

**ClawhDF5 solution:**

ClawhDF5 is a full HDF5 implementation that *also* has agent memory. You can:

- **Read existing HDF5 files** from CERN, NASA, NOAA — no C library needed
- **Add vector search** to your datasets by embedding them and storing in the agent memory layer
- **Query across datasets** using hybrid search (find the experiment that matches your description)
- **Track data provenance** with the built-in provenance system

```rust
use clawhdf5::File;
use clawhdf5_agent::{HDF5Memory, MemoryConfig};

// Read your scientific data
let data = File::open("experiment_results.h5")?;
let measurements = data.dataset("sensor_readings")?.read_f64()?;

// Create a searchable memory alongside it
let mut memory = HDF5Memory::create(
    MemoryConfig::new("experiment_memory.h5", "lab-assistant", 384)
)?;

// Embed and index experiment descriptions
memory.save(MemoryEntry {
    chunk: "Experiment 47: Temperature response at 350K with catalyst B".into(),
    embedding: embed("Temperature response..."),
    source_channel: "lab-notebook".into(),
    ..default()
})?;

// Later: "which experiments used catalyst B above 300K?"
let results = memory.hybrid_search(&query_emb, "catalyst B temperature", 0.6, 0.4, 10);
```

---

## 6. The `.brain` Format (ClawBrainHub)

**Scenario:** You've built an amazing AI agent with custom personality, skills, and accumulated knowledge. You want to package it and distribute it.

**Problem:** Agent identity is scattered across config files, prompt templates, skill definitions, vector stores, and various databases. There's no standard format.

**ClawhDF5 solution — the `.brain` file:**

```
agent.brain (HDF5)
├── /meta           — schema version, author, license
├── /identity       — system prompt, personality, avatar
├── /skills         — tool definitions, MCP configs
├── /memory         — vector embeddings, knowledge graph
├── /media          — voice samples, images
├── /runtime        — model preferences, resource limits
└── /provenance     — SHA-256 hashes, Ed25519 signatures
```

One file. Cryptographically signed. Publishable to [ClawBrainHub](https://clawbrainhub.com).

```bash
# Create a brain file
clawhdf5 --path agent.brain create --agent-id my-agent --dim 384

# Publish to ClawBrainHub (coming soon)
clawhub publish agent.brain

# Pull a brain
clawhub pull redclawsystems/research-assistant
```

This is the container image for intelligence.

---

## Choosing the Right Features

| Your Situation | Features to Enable | Why |
|----------------|-------------------|-----|
| **Quick prototype** | Default | Vector search works out of the box |
| **Production agent** | `agent`, `float16`, `parallel` | Half-precision saves 50% storage, parallel search for scale |
| **macOS** | + `accelerate` | Apple AMX coprocessor for matrix ops |
| **Linux server** | + `openblas` or `fast-math` | BLAS acceleration |
| **GPU available** | + `gpu` | wgpu-based search, wins at 100K+ scale |
| **Long-running agent** | + `async` | Tokio async with background flush |
| **Edge device** | Default only | Minimal dependencies, smallest binary |

```toml
# Production agent on Linux
clawhdf5-agent = { version = "2.0", features = ["agent", "float16", "parallel", "fast-math"] }

# Edge device
clawhdf5-agent = { version = "2.0", features = ["agent"] }

# macOS with GPU
clawhdf5-agent = { version = "2.0", features = ["agent", "float16", "accelerate", "gpu", "async"] }
```

---

<p align="center"><em>Built by <a href="https://github.com/redclawsystems">RedClaw Systems</a></em></p>
