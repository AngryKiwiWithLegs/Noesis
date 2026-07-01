"""
noesis/wiki/ingestor.py

Orchestrates the Ingest operation:
  load source → chunk → extract → dedupe/merge → write pages → log → index

Mirrors the ConsolidationPipeline flow but is synchronous and on-demand
(wiki ingest is user-triggered via CLI, not a background queue).

Source loading:
  .md / .markdown  → strip frontmatter, keep body (markdown-aware)
  .txt             → raw text
  .pdf             → pypdf text extraction (lazy import)
  http(s)://       → httpx fetch, then markdown/text/html handling

Chunking is token-aware via the tokenizers lib (already installed),
splitting long documents at ~chunk_tokens boundaries on heading/sentence
edges.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .types import DocSource, WikiCandidate, WikiPage
from .writer import WikiWriter
from .extractor import AbstractWikiExtractor

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_TOKENS = 800


class WikiIngestor:
    """Ingest a document into the wiki. Idempotent."""

    def __init__(
        self,
        vault_path: str,
        extractor: AbstractWikiExtractor,
        writer: Optional[WikiWriter] = None,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    ):
        self.vault_path = vault_path
        self.extractor = extractor
        self.writer = writer or WikiWriter(vault_path)
        self.chunk_tokens = chunk_tokens

    def ingest(self, path: str) -> list[str]:
        """Ingest a file path or URL. Returns page_ids written/updated."""
        doc = self._load_source(path)
        if not doc.content.strip():
            logger.warning(f"No content extracted from {path}")
            return []

        chunks = self._chunk(doc)
        logger.info(f"Ingesting {path}: {len(chunks)} chunk(s)")

        written: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            candidates = self.extractor.extract(doc, chunk=chunk)
            for cand in candidates:
                self._store(cand)
                if cand.page_id not in seen:
                    seen.add(cand.page_id)
                    written.append(cand.page_id)

        # Refresh the routing catalog + log
        self.writer.rebuild_index()
        for pid in written:
            self.writer.append_log("ingest", pid, f"from {doc.title or path}")

        logger.info(f"Ingested {len(written)} page(s) from {path}")
        return written

    # ── Source loading ─────────────────────────────────────────────────────────

    def _load_source(self, path: str) -> DocSource:
        """Build a DocSource from a path or URL, dispatching by extension."""
        if path.startswith(("http://", "https://")):
            return self._load_url(path)

        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        suffix = p.suffix.lower()
        title = p.stem.replace("_", " ").replace("-", " ").title()

        if suffix in (".md", ".markdown"):
            content = self._strip_frontmatter(p.read_text(encoding="utf-8"))
            return DocSource(path=str(p), fmt="markdown", content=content, title=title)
        elif suffix == ".pdf":
            content = self._load_pdf(p)
            return DocSource(path=str(p), fmt="pdf", content=content, title=title)
        else:
            # .txt and anything else → raw text
            content = p.read_text(encoding="utf-8", errors="replace")
            return DocSource(path=str(p), fmt="text", content=content, title=title)

    def _load_url(self, url: str) -> DocSource:
        try:
            import httpx
        except ImportError as e:
            raise ImportError("pip install httpx") from e
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        # Crude format detection from content-type / extension
        ct = resp.headers.get("content-type", "")
        if "html" in ct or "<html" in text.lower():
            content = self._strip_html(text)
            fmt = "text"
        else:
            content = text
            fmt = "markdown" if ".md" in url else "text"
        title = url.rsplit("/", 1)[-1] or url
        return DocSource(path=url, fmt=fmt, content=content, title=title)

    def _load_pdf(self, p: Path) -> str:
        """Extract text from a PDF via pypdf (lazy import)."""
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # legacy fallback
            except ImportError as e:
                raise ImportError(
                    "PDF ingestion needs pypdf: pip install pypdf"
                ) from e
        reader = PdfReader(str(p))
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n\n".join(parts)

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Drop a leading YAML frontmatter block (--- ... ---)."""
        if text.lstrip().startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                return parts[2].lstrip("\n")
        return text

    @staticmethod
    def _strip_html(html: str) -> str:
        """Naive HTML → text (no bs4 dependency). Good enough for ingestion."""
        # Remove scripts/styles
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Block tags → newlines
        html = re.sub(r"<\s*(/)?(p|div|br|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
        # Strip remaining tags
        text = re.sub(r"<[^>]+>", "", html)
        # Decode common entities
        text = (text.replace("&amp;", "&").replace("&lt;", "<")
                    .replace("&gt;", ">").replace("&quot;", '"')
                    .replace("&#39;", "'").replace("&nbsp;", " "))
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    # ── Chunking ───────────────────────────────────────────────────────────────

    def _chunk(self, doc: DocSource) -> list[str]:
        """Token-aware chunking. Splits long docs at heading/sentence edges."""
        text = doc.content
        if not text.strip():
            return []

        approx_tokens = self._est_tokens(text)
        if approx_tokens <= self.chunk_tokens:
            return [text]

        chunks: list[str] = []
        # Prefer splitting on Markdown headings first
        sections = re.split(r"(?=\n#{1,2}\s)", text)
        cur = ""
        for sec in sections:
            if self._est_tokens(cur + sec) > self.chunk_tokens and cur:
                chunks.append(cur)
                cur = sec
            else:
                cur += sec
        if cur.strip():
            chunks.append(cur)

        # If any chunk is still too big, split on sentence boundaries
        final: list[str] = []
        for c in chunks:
            if self._est_tokens(c) <= self.chunk_tokens * 1.5:
                final.append(c)
            else:
                final.extend(self._split_sentences(c))
        return final or [text]

    def _split_sentences(self, text: str) -> list[str]:
        """Fallback chunker: split on sentence boundaries."""
        sentences = re.split(r"(?<=[.!?。])\s+", text)
        chunks, cur = [], ""
        for s in sentences:
            if self._est_tokens(cur + s) > self.chunk_tokens and cur:
                chunks.append(cur)
                cur = s
            else:
                cur += " " + s if cur else s
        if cur.strip():
            chunks.append(cur)
        return chunks

    @staticmethod
    def _est_tokens(text: str) -> int:
        """Cheap token estimate (CJK-aware), mirrors main.py:_token_est."""
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        return cjk // 2 + (len(text) - cjk) // 4 + 1

    # ── Storage ────────────────────────────────────────────────────────────────

    def _store(self, cand: WikiCandidate):
        """Convert a candidate to a WikiPage and upsert (idempotent)."""
        page = WikiPage(
            page_id=cand.page_id,
            title=cand.title,
            body=cand.body,
            page_type=cand.page_type,
            topic_cluster=cand.topic_cluster or "general",
            status="published",  # mock/cloud output is considered current
            sources=cand.sources,
            citations=cand.citations,
        )
        self.writer.upsert(page)
