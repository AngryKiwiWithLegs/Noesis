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

## Quickstart

```bash
git clone https://github.com/you/noesis
cd noesis
bash setup.sh
```

```bash
# Copy and edit config
mkdir -p ~/.noesis
cp config.example.yaml ~/.noesis/config.yaml
# Edit vault_path and optionally add your LLM API key

# Start the daemon
noesis start
```

**Point any OpenAI-compatible tool at Noesis:**

```
Change your tool's API base URL from:
  https://api.openai.com/v1
To:
  http://localhost:8080/v1

Everything else (model name, API key) stays the same.
```

**Claude Desktop (MCP):**

```json
{
  "mcpServers": {
    "noesis": { "command": "noesis", "args": ["mcp"] }
  }
}
```

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
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │  Hot store (sqlite-vec) │  <10ms retrieval
              │  Cold store (Obsidian)  │  human-readable, human-editable
              └────────────┬────────────┘
                           │
              ContextBuilder → system prompt injection
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
