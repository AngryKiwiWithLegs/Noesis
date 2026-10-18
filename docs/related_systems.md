# Related memory systems: what each cited version describes

This table backs the Related Work section of the Noesis paper
(`paper/noesis_paper.tex`). Each row describes only the version the paper cites,
as read on 17 October 2026, not later releases. "Not described" means the cited
source does not mention it; it does not mean the system lacks it. The version
frozen with the paper is the one at the `v1.0-preprint` tag; update this file
in later commits as systems change.

| System | Version cited | Interfaces | Rule that keeps stored items out of the prompt | Change over time | Storage and models | Benchmarks reported |
|---|---|---|---|---|---|---|
| Mem0 | arXiv:2504.19413 (2025) | Not checked beyond the abstract | Not described in the abstract | Extracts and consolidates salient facts; a graph-based variant captures relations | Not checked | LoCoMo |
| OpenMemory MCP | Mem0 blog announcement, 13 May 2025 | MCP over a local SSE endpoint (Cursor, Claude Desktop, Windsurf, Cline); tools `add_memories`, `search_memory`, `list_memories`, `delete_all_memories` | Not described | Not described | Runs locally; setup asks for an OpenAI API key, use not stated | None |
| Zep (Graphiti) | arXiv:2501.13956v1 (Jan 2025) | Memory service | Not described; results are reranked by relevance (RRF, MMR, cross-encoder and others), not filtered by a confidence threshold | Bi-temporal fact edges; an LLM finds contradictions and invalidates the older edge instead of deleting it | Graph database (BM25 through Neo4j's Lucene index); LLM extraction | DMR, LongMemEval |
| Hindsight | ACL 2026 System Demonstrations, pp. 275–285 (details read from arXiv:2512.12818v1) | Recall interface; MCP or other APIs not described | Not described; opinion confidence records belief strength and is not described as a recall condition | Four networks (world facts, the agent's experiences, opinions, observations); opinion confidence moves with reinforcing, weakening or contradicting evidence | PostgreSQL with pgvector; LLM fact extraction; four recall channels fused with RRF, cross-encoder rerank, token budget | LongMemEval 83.6% (20B open model), 91.4% (Gemini-3 Pro); LoCoMo 83.2% |
| A-MEM | NeurIPS 2025 (arXiv:2502.12110) | Not checked | Not checked | Zettelkasten-style notes whose links and attributes are updated as memories arrive | Not checked | Not checked |
| AgeMem | arXiv:2601.01885v3 (ACL 2026) | Memory operations exposed to the agent as tool actions | Learned: the policy decides what to store, retrieve, update, summarize or discard | Trained with reinforcement learning | Not checked | Five long-horizon benchmarks |
| MRAgent | arXiv:2606.06036v1 (ICML 2026) | Not checked | Not described | Cue–Tag–Content associative graph accessed iteratively, guided by intermediate reasoning | Not checked | LoCoMo, LongMemEval |
| Noesis | This repository, `v1.0-preprint` | OpenAI-compatible proxy (injects and captures; used in the A/B runs), MCP server (model-called `remember`/`recall`; not evaluated), browser extension (capture only; not evaluated) | Status gate: nothing below provisional is injected; a hard token budget caps the rest | Confidence lifecycle (tentative → provisional → settled); rule-based supersession by replacement verb and domain | SQLite with sqlite-vec, all-MiniLM-L6-v2 on CPU, Markdown vault; optional cloud LLM for extraction | Synthetic personas only |

Sources are listed in `paper/references.bib`.
