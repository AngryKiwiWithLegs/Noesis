# Noesis

**Your thinking layer. Yours forever.**

Not a memory tool — a thinking tool.
Every AI conversation captures a fragment of how you reason.
Noesis collects those fragments, structures them, and makes them available
to every AI you use — automatically, locally, privately.

---

## Why this exists

You talk to Claude. You talk to GPT. You switch tools, start new chats.
Each conversation disappears when you close the tab.
Your reasoning — your positions, your questions, your judgements — evaporates.

Noesis captures it. Not as a transcript, but as structured thought:

```
position   — "I've decided sqlite-vec beats Chroma for this use case"
question   — "What happens to BM25 performance above 50k documents?"
synthesis  — "Claude and GPT both agree: graph expansion beats pure vector"
identity   — "I'm a senior ML engineer at a healthcare startup"
preference — "I prefer zero-dependency solutions"
```

Every thought lives in your Obsidian vault.
Every AI you use can read from it.
You own it.

---

## Setup (new user guide)

### 1. Prerequisites

| Requirement | Why | Check |
|---|---|---|
| **Python 3.11+** | Core runtime | `python3 --version` |
| **~400 MB disk** | Embedding model (~80 MB) + deps | — |
| **Obsidian** *(optional but recommended)* | Human-readable cold store | [obsidian.md](https://obsidian.md) |

That's it for the base install. Everything below is optional and additive —
you can start with zero API keys (fully local) and add cloud models later.

### 2. Install

```bash
git clone https://github.com/AngryKiwiWithLegs/Noesis.git
cd Noesis
bash setup.sh
```

`setup.sh` creates a `.venv`, installs dependencies, verifies imports, and runs
the latency test suite. **First run downloads the embedding model (~80 MB).**

```bash
# Activate the venv in every new terminal
source .venv/bin/activate
```

### 3. Configure

```bash
mkdir -p ~/.noesis
cp config.example.yaml ~/.noesis/config.yaml
```

Then edit `~/.noesis/config.yaml`. The three sections that matter:

```yaml
# Hot store — your fast local index (always required)
vector_store:
  config:
    db_path: ~/.noesis/hot.db

# Local embeddings — no network, no key (always required)
embedder:
  config:
    model: all-MiniLM-L6-v2

# Cold store — where thoughts live as readable Markdown (recommended)
cold_store:
  config:
    vault_path: ~/Documents/NoesisVault   # point at your Obsidian vault
```

### 4. Pick your model tier

Noesis is **model-agnostic**. You can run fully local, fully cloud, or both.
The two concerns are separate — understand the distinction up front:

> **Thought extraction** (in `config.yaml` `llm:`) classifies *what kind* of
> thought was captured (position / question / preference / …).
> **Chat routing** (model name prefix) decides *which provider* answers your
> tool's requests through the proxy. They are independent.

#### Option A — Fully local, zero keys (simplest start)

No `llm:` block in config. Thoughts are stored as raw text (no type
classification until you add a key). For chat, run a local model:

```bash
# Install Ollama: https://ollama.com  (one-time)
ollama pull gemma3:4b      # or qwen2.5:3b, llama3.2, phi3.5, …
```

Then send requests to the proxy with `model: gemma3:4b` — no auth needed.

#### Option B — Cloud LLM for thought extraction (recommended)

Add an `llm:` block so Noesis classifies thoughts as you chat:

```yaml
llm:
  provider: anthropic                              # or: openai
  model:    claude-haiku-4-5-20251001              # or: gpt-4o-mini
  # api_key: sk-ant-...                            # or set ANTHROPIC_API_KEY
```

#### Option C — Cloud chat models through the proxy

The proxy routes by **model-name prefix** — set your tool's model field and
send your provider key in the `Authorization` header (the proxy passes it
through; it never stores keys itself):

| Model prefix in request | Routes to | Auth header |
|---|---|---|
| `gpt-`, `o1`, `o3` | api.openai.com | `Bearer $OPENAI_API_KEY` |
| `claude-` | api.anthropic.com | `Bearer $ANTHROPIC_API_KEY` |
| `gemini-` | generativelanguage.googleapis.com | `Bearer $GEMINI_API_KEY` |
| `mistral-` | api.mistral.ai | `Bearer $MISTRAL_API_KEY` |
| `deepseek` | api.deepseek.com | `Bearer $DEEPSEEK_API_KEY` |
| `gemma`/`llama`/`phi`/`qwen`/`mixtral` | localhost:11434 (Ollama) | none |

### 5. Start the daemon

```bash
noesis start          # API proxy on :8080
noesis start --ws     # also start the browser-extension WebSocket on :8082
```

### 6. Connect your tools

**Any OpenAI-compatible tool** — change the API base URL:

```
from:  https://api.openai.com/v1
to:    http://localhost:8080/v1
```

Model name and API key stay as they were. Noesis injects memory into the
system prompt invisibly, then forwards to the real provider.

**Claude Desktop (MCP)** — `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "noesis": { "command": "noesis", "args": ["mcp"] }
  }
}
```

This exposes five tools: `remember`, `recall`, `wiki_query`,
`inspect_memory`, `memory_status`.

### 7. Verify it works

```bash
noesis status                 # should show your memory stats
noesis inspect <hash_id>      # read a specific node

# Test the proxy with a local model (no key needed):
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3:4b","messages":[{"role":"user","content":"hi"}]}'
```

Run the test suite to confirm the install:

```bash
pytest tests/ -v -k "not slow"
```

### 8. (Optional) Seed memory from past chats

```bash
noesis import --source chatgpt conversations.json
noesis import --source text    notes.txt
```

Imported nodes start `tentative` and promote as you keep using Noesis.

### 9. (Optional) Build your LLM Wiki

The wiki layer compiles documents (papers, notes, manuals) into cited,
searchable knowledge pages that cross-link with your thoughts:

```bash
noesis wiki ingest ~/papers/sqlite-vec.pdf     # compile → wiki pages
noesis wiki status                             # page counts + clusters
noesis wiki query "vector search in sqlite"    # search compiled knowledge
noesis wiki lint                               # find gaps, orphans, contradictions
noesis wiki answer <hash> <page_id>            # resolve an open question
```

Without an `llm:` key, the deterministic Mock extractor handles structured
markdown; with a key, the Cloud extractor compiles with citations.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Address already in use` on start | A daemon is already running. `lsof -iTCP:8080` to find it, then kill or reuse it. |
| Cloud model returns `401 Unauthorized` | You didn't send the provider key in the `Authorization` header. The proxy forwards it; it doesn't store keys. |
| Local model returns connection refused | Ollama isn't running. `ollama serve` or start the app. |
| Thoughts show as `tentative` forever | No `llm:` block configured, so nothing classifies/promotes them. Add a key (Option B). |
| First `noesis start` is slow | Downloading the embedding model (~80 MB, one-time). |

---

## Architecture

```
All AI tools
  Claude Desktop  →  MCP Server           ┐
  Any API tool    →  Local Proxy :8080     ├── Noesis Daemon (local)
  ChatGPT web     →  Browser Extension    ┘
                           │
              ┌────────────┴────────────┐
              │   Cloud LLM (optional)  │  thought extraction
              │   ConfidenceScorer      │  tentative→provisional→settled
              │   ConsolidationPipeline │  async, never blocks add()
              │   Supersession detector │  retires stale stances automatically
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │  Hot store (sqlite-vec) │  <10ms retrieval
              │  Cold store (Obsidian)  │  human-readable, human-editable
              │  LLM Wiki (wiki/*.md)   │  compiled document knowledge
              └────────────┬────────────┘
                           │
              ContextBuilder → system prompt injection
              (thoughts + wiki knowledge, ranked, budgeted)
```

---

## Confidence lifecycle

Every thought starts tentative and must earn its way up:

```
tentative    stored only, never injected, invisible in graph
    ↓  repetition + assertion strength + cross-tool consistency
provisional  injected when topic-relevant
    ↓  more evidence
settled      always injected, full graph links
```

Four signals promote nodes: **repetition** (same stance recurs across sessions),
**assertion strength** ("I've decided" > "maybe"), **cross-tool consistency**
(Claude and GPT agree), **absence of contradiction** (no conflicting stance found).

---

## Two layers, one vault

| Layer | Written by | Content |
|---|---|---|
| `thoughts/` | Noesis | Your reasoning, positions, questions |
| `wiki/` | LLM Wiki | Knowledge from documents you've read |
| `clusters/` | Both | Topic aggregation — the meeting point |

Questions you raise in Noesis drive LLM Wiki's knowledge ingestion.  
Knowledge in `wiki/` grounds the positions stored in `thoughts/`.

---

## CLI

```bash
noesis start   [--port 8080] [--ws]   # start daemon
noesis status  [--user me]            # memory stats
noesis inspect <hash_id>              # show a memory node
noesis sync                           # force sync vault edits → hot store
noesis import  --source chatgpt FILE  # batch import existing conversations
noesis eval                           # run injection accuracy benchmark
noesis mcp                            # start MCP server (stdio)

# LLM Wiki (compiled document knowledge)
noesis wiki ingest <file|url>         # compile a document into wiki pages
noesis wiki query <text> [-k N]       # search compiled wiki knowledge
noesis wiki answer <hash> <page_id>   # mark a question answered by a page
noesis wiki lint                      # audit: open questions, orphans, gaps
noesis wiki status                    # wiki page counts + recent activity
```

---

## Python API

```python
from noesis.memory.main import Memory

m = Memory.from_config_file("~/.noesis/config.yaml")

# Store a thought
m.add("我决定用 sqlite-vec 作为热库", user_id="me", type="position")

# Get context string for system prompt injection
ctx = m.build_context("存储方案选型", user_id="me")
# → "以下是关于用户的已知信息：\n- [position·settled] ..."

# Mem0-compatible search
results = m.search("向量库", user_id="me", top_k=5)

# Status
print(m.status(user_id="me"))
# → {'total': 42, 'settled': 18, 'provisional': 11, 'tentative': 13, ...}
```

---

## Evaluation

```bash
pytest tests/test_benchmark.py -v -m slow
```

30 cases across 5 categories. Target: **≥ 80% injection accuracy**.

| Category | Cases | Target |
|---|---|---|
| Status gating | 6 | ≥ 80% |
| Time awareness | 6 | ≥ 70% |
| Cross-tool | 5 | ≥ 80% |
| Core facts | 5 | ≥ 85% |
| Budget & dedup | 8 | ≥ 90% |

Zero tentative leakage is an absolute requirement — any tentative node
appearing in `build_context()` output is a critical bug.

---

## Comparison

| | Mem0 | LLM Wiki | **Noesis** |
|---|---|---|---|
| Source | AI conversations | Docs you feed it | AI conversations |
| Captures | Facts | Knowledge | **Reasoning process** |
| Confidence lifecycle | ✗ | ✗ | ✓ |
| Graph expansion | ✗ | ✓ | ✓ |
| Injection into AI | ✓ | ✗ | ✓ |
| Cross-tool | ✗ | ✗ | ✓ |
| Human-editable | ✗ | ✓ | ✓ |
| Local-first | ✗ | ✓ | ✓ |

Noesis and LLM Wiki are complementary. Run both in the same Obsidian vault
using `NOESIS_SCHEMA.md` as the integration contract.

---

## Development

```bash
# Run all fast tests (no benchmark)
pytest tests/ -v -k "not slow"

# Run specific week
pytest tests/test_latency.py   -v   # Week 1 — latency
pytest tests/test_confidence.py -v  # Week 2 — confidence scoring
pytest tests/test_two_phase.py  -v  # Week 2 — async pipeline
pytest tests/test_retrieval.py  -v  # Week 3 — hybrid retrieval
pytest tests/test_injection.py  -v  # Week 3 — context injection ★
pytest tests/test_proxy.py      -v  # Week 4 — API proxy + MCP
pytest tests/test_cross_tool.py -v  # Week 5 — cross-tool + clusters
pytest tests/test_wiki.py     -v   # LLM Wiki — ingest, lint, query, writer
pytest tests/test_benchmark.py  -v -m slow  # Week 6 — formal benchmark
```

**Roadmap:**

| Week | Focus | Status |
|---|---|---|
| 1 | Local stack, latency baseline | ✓ |
| 2 | Cloud LLM extraction, confidence lifecycle | ✓ |
| 3 | Hybrid retrieval, graph expansion, context builder | ✓ |
| 4 | API proxy, MCP server, CLI, daemon | ✓ |
| 5 | Browser extension, cluster manager, cross-tool | ✓ |
| 6 | 30-case benchmark, final packaging | ✓ |

---

## Design principles

1. **Capture threshold low, injection threshold high** — store aggressively, inject conservatively
2. **Time is the best filter** — repetition and consistency over weeks beats one-shot LLM judgement
3. **Thoughts store why/when/how; facts store what** — `fact_ref` prevents duplication
4. **Graph always readable** — cluster nodes cap at ~100; tentative invisible in graph
5. **Local sovereignty** — your data in your vault; LLM API calls are the only thing that leaves your machine

---

## License

Apache 2.0
