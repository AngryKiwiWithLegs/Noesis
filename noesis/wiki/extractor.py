"""
noesis/wiki/extractor.py

Turns a chunk of source text into structured WikiCandidates.

Mirrors noesis/thoughts/extractor.py:
  - AbstractWikiExtractor            ~ AbstractExtractor
  - MockWikiExtractor (offline)      ~ MockExtractor (heading-split, no API key)
  - CloudWikiExtractor (LLM)         ~ CloudLLMExtractor (JSON output)

The Mock extractor is deterministic — it splits on Markdown headings and
infers clusters via the shared _DOMAIN_KEYWORDS map. This makes offline
ingestion viable (the same design philosophy that made MockExtractor work
for the experiment suite).

The Cloud extractor asks the LLM to compile knowledge into named pages
with citations — Karpathy's "extract → update entity/concept/summary pages".
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from .types import DocSource, WikiCandidate, WikiPageType

logger = logging.getLogger(__name__)


# ── Cluster inference (shared with the thoughts pipeline) ─────────────────────
# We import lazily to avoid a hard dependency cycle: pipeline.py imports
# nothing from wiki/, so this is safe, but keeping it lazy lets the wiki
# module stand alone in tests.

_DOMAIN_KEYWORDS = {
    "database":        ["sql", "sqlite", "postgres", "mysql", "mongo", "redis",
                        "database", "db", "faiss", "milvus", "qdrant"],
    "language":        ["python", "rust", "typescript", "javascript", "java",
                        "golang", "go ", "c++", "swift", "kotlin"],
    "cloud":           ["aws", "azure", "gcp", "cloud", "kubernetes", "docker",
                        "lambda", "s3", "ec2"],
    "frontend":        ["react", "vue", "svelte", "css", "html", "tailwind",
                        "nextjs", "frontend"],
    "api":             ["rest", "graphql", "grpc", "api", "endpoint", "openapi"],
    "editor":          ["vscode", "neovim", "emacs", "vim", "jetbrains", "idea"],
    "llm":             ["llm", "gpt", "claude", "gemini", "transformer",
                        "embedding", "rag", "fine-tune", "finetune"],
    "vector-store":    ["vector", "embedding", "sqlite-vec", "faiss", "ann",
                        "similarity", "retrieval"],
    "version-control": ["git", "github", "gitlab", "branch", "merge", "rebase"],
}


def infer_cluster(text: str) -> str:
    """Topic-cluster guess that stays aligned with the thoughts pipeline.

    The NOESIS_SCHEMA.md cross-link rule matches wiki↔thoughts by topic_cluster,
    so both systems MUST share the same cluster vocabulary. We delegate to the
    thoughts extractor's _infer_cluster first (the authoritative source), then
    fall back to the richer _DOMAIN_KEYWORDS map only when it returns "general".
    """
    # Delegate to the thoughts system's cluster inference for alignment
    try:
        from ..thoughts.extractor import _infer_cluster as _thought_cluster
        primary = _thought_cluster(text)
        if primary != "general":
            return primary
    except Exception:
        pass
    # Fallback: richer domain keywords for wiki-specific sources (papers, docs)
    # that the thoughts keyword map doesn't cover
    t = (text or "").lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(k in t for k in keywords):
            return domain
    return "general"


def _slugify(text: str) -> str:
    """Turn a heading/title into a kebab-case page slug."""
    s = re.sub(r"[^\w\s-]", "", text.lower().strip())
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:60] or "untitled"


def _classify_type(title: str, body: str) -> WikiPageType:
    """Heuristic page-type guess for the mock extractor."""
    t = (title + " " + body[:200]).lower()
    if any(w in t for w in ["summary", "abstract", "digest", "overview", "tl;dr"]):
        return "summary"
    # Named things (capitalised words, version numbers) tend to be entities
    if re.search(r"\b[A-Z][a-z]+\b", title) or re.search(r"\d+\.\d+", title):
        return "entity"
    return "concept"


# ── Abstract base ─────────────────────────────────────────────────────────────

class AbstractWikiExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        doc: DocSource,
        chunk: str = "",
    ) -> list[WikiCandidate]:
        ...


# ── Mock extractor (offline, deterministic) ──────────────────────────────────

class MockWikiExtractor(AbstractWikiExtractor):
    """
    Deterministic extractor for offline ingestion (no API key).

    Strategy: split the chunk on Markdown headings (H1/H2). Each section
    becomes one WikiCandidate. A document with no headings becomes a single
    page titled after the source.

    This mirrors Karpathy's guidance: deterministic scripts for intake,
    LLM only for judgment. Mock handles the structural intake; Cloud
    handles the semantic compilation.
    """

    def extract(self, doc: DocSource, chunk: str = "") -> list[WikiCandidate]:
        text = chunk or doc.content
        if not text.strip():
            return []

        sections = self._split_sections(text)
        candidates = []
        source_ref = doc.path or doc.title or "source"

        if not sections:
            # No headings — whole chunk is one page
            title = doc.title or _slugify(doc.path) or "document"
            body = text.strip()
            candidates.append(self._candidate(title, body, source_ref))
            return candidates

        for heading, body in sections:
            if not body.strip():
                continue
            candidates.append(self._candidate(heading, body, source_ref))
        return candidates

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Split markdown into (heading, body) sections at H1/H2 boundaries."""
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []
        cur_heading: str | None = None
        cur_body: list[str] = []

        for line in lines:
            m = re.match(r"^(#{1,2})\s+(.+)$", line)
            if m:
                # Flush previous section
                if cur_heading is not None:
                    sections.append((cur_heading, "\n".join(cur_body)))
                cur_heading = m.group(2).strip()
                cur_body = []
            else:
                cur_body.append(line)
        if cur_heading is not None:
            sections.append((cur_heading, "\n".join(cur_body)))
        return sections

    def _candidate(self, title: str, body: str, source_ref: str) -> WikiCandidate:
        slug = _slugify(title)
        return WikiCandidate(
            page_id=slug,
            title=title.strip(),
            page_type=_classify_type(title, body),
            topic_cluster=infer_cluster(title + " " + body),
            body=body.strip(),
            sources=[source_ref],
            citations=[],
        )


# ── Cloud LLM extractor ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a knowledge compiler for a personal wiki.
Given a chunk of a source document, extract reusable knowledge into
structured wiki pages.

## For each distinct concept, entity, or summary, produce a page:
- page_id: kebab-case slug ("sqlite-vec", "transformer-attention")
- title: human-readable title
- page_type: "concept" (definition/explanation), "entity" (named thing \
with attributes), or "summary" (synthesis/digest of the source)
- topic_cluster: 2-4 word kebab-case label matching the document's domain \
("vector-store", "llm-choice", "python-tooling", "database")
- body: clean Markdown. Compile facts into declarative statements. \
Preserve the source's actual claims — don't invent.
- citations: list of short quoted spans from the source that support the body

## Rules:
- Only emit pages for genuine knowledge worth keeping long-term
- Merge related facts into one page; don't split trivially
- If the chunk has no reusable knowledge, return an empty list
- Output ONLY valid JSON, no preamble, no markdown fences

Return: {"pages": [{"page_id", "title", "page_type", "topic_cluster", \
"body", "citations"}]}"""

_USER_TEMPLATE = """Source: {title}
Format: {fmt}

--- chunk ---
{chunk}
--- end chunk ---

Compile this into wiki pages."""


class CloudWikiExtractor(AbstractWikiExtractor):
    """
    LLM-powered extractor. Asks the model to compile knowledge into named
    pages with citations — Karpathy's "extract knowledge into wiki/*.md".

    Mirrors CloudLLMExtractor: Anthropic (default) or OpenAI, JSON output,
    retry-on-parse-failure, returns [] on total failure (never raises).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_retries: int = 2,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=self.api_key)
                except ImportError as e:
                    raise ImportError("pip install anthropic") from e
            elif self.provider == "openai":
                try:
                    import openai
                    self._client = openai.OpenAI(api_key=self.api_key)
                except ImportError as e:
                    raise ImportError("pip install openai") from e
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def extract(self, doc: DocSource, chunk: str = "") -> list[WikiCandidate]:
        body = chunk or doc.content
        if not body.strip():
            return []

        user_msg = _USER_TEMPLATE.format(
            title=doc.title or doc.path or "document",
            fmt=doc.fmt,
            chunk=body[:8000],  # cap to keep within context
        )

        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call_llm(user_msg)
                return self._parse(raw, doc)
            except Exception as e:
                logger.warning(f"Wiki extract attempt {attempt+1} failed: {e}")
                if attempt == self.max_retries:
                    logger.error("All wiki extraction attempts failed; returning empty")
                    return []
        return []

    def _call_llm(self, user_msg: str) -> str:
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return resp.content[0].text
        elif self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        raise ValueError(f"Unknown provider: {self.provider}")

    def _parse(self, raw: str, doc: DocSource) -> list[WikiCandidate]:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in: {raw[:200]}")
        data = json.loads(m.group())
        pages = data.get("pages", [])
        if not isinstance(pages, list):
            return []
        source_ref = doc.path or doc.title or "source"
        out = []
        for p in pages:
            title = p.get("title", "").strip()
            if not title:
                continue
            pid = p.get("page_id") or _slugify(title)
            out.append(WikiCandidate(
                page_id=pid,
                title=title,
                page_type=_valid_page_type(p.get("page_type", "concept")),
                topic_cluster=p.get("topic_cluster") or infer_cluster(title + " " + p.get("body", "")),
                body=(p.get("body") or "").strip(),
                sources=[source_ref],
                citations=[str(c) for c in p.get("citations", []) if c],
            ))
        return out


_VALID_PAGE_TYPES = {"concept", "entity", "summary"}


def _valid_page_type(t: str) -> WikiPageType:
    return t if t in _VALID_PAGE_TYPES else "concept"
