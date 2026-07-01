"""
noesis/wiki/  — LLM Wiki module

A Karpathy-style LLM Wiki inside Noesis. Ingests documents into
structured wiki/*.md pages, cross-links them with thoughts/, and
injects wiki knowledge into live LLM context.

Implements the NOESIS_SCHEMA.md contract:
  wiki/       ← this module writes here (compiled from documents)
  thoughts/   ← Noesis writes here (graph_linker links the two)
  clusters/   ← shared tier (either system may write)

Three operations (mirrors Karpathy's llm-wiki pattern):
  Ingest  — DocSource → WikiExtractor → WikiWriter (pages + index + log)
  Query   — search wiki pages → inject as context signal
  Lint    — find open questions, orphans, contradictions; mark answered
"""
from .types import WikiPage, DocSource, WikiCandidate
from .writer import WikiWriter
from .extractor import (
    AbstractWikiExtractor,
    MockWikiExtractor,
    CloudWikiExtractor,
)
from .ingestor import WikiIngestor
from .lint import WikiLinter

__all__ = [
    "WikiPage",
    "DocSource",
    "WikiCandidate",
    "WikiWriter",
    "AbstractWikiExtractor",
    "MockWikiExtractor",
    "CloudWikiExtractor",
    "WikiIngestor",
    "WikiLinter",
]
