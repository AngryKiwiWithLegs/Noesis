"""
tests/test_wiki.py

LLM Wiki module tests. Covers the three Karpathy operations end-to-end
with the offline Mock extractor — no API keys needed.

Covers:
  - WikiWriter: write/read round-trip, upsert idempotency, index rebuild,
    reserved-files exclusion
  - WikiIngestor: Mock heading-split, cluster inference, idempotent re-ingest
  - WikiLinter: orphans, contradictions, coverage gaps, mark_answered
  - wiki_signal: query surfaces ranked wiki pages as thought-node-shaped dicts

Run:
    pytest tests/test_wiki.py -v
"""
from __future__ import annotations

import pytest

from noesis.wiki import (
    WikiWriter,
    WikiPage,
    WikiIngestor,
    WikiLinter,
    MockWikiExtractor,
)
from noesis.wiki.extractor import infer_cluster, _slugify


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vault(tmp_path):
    """A fresh vault root directory (no Memory/hot store needed)."""
    return str(tmp_path)


@pytest.fixture
def writer(vault):
    return WikiWriter(vault)


@pytest.fixture
def ingestor(vault):
    return WikiIngestor(vault, MockWikiExtractor())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _md_doc(path, sections: list[tuple[str, str]]):
    """Write a markdown doc of '# Heading\nbody' sections."""
    parts = []
    for h, body in sections:
        parts.append(f"# {h}\n\n{body}")
    path.write_text("\n\n".join(parts), encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# WikiWriter
# ══════════════════════════════════════════════════════════════════════════════

class TestWikiWriter:

    def test_write_then_read_round_trip(self, writer):
        """read_page must recover every frontmatter field + body written."""
        page = WikiPage(
            page_id="sqlite-vec",
            title="SQLite-Vec",
            body="SQLite-Vec is a vector search extension.",
            page_type="concept",
            topic_cluster="vector-store",
            status="published",
            sources=["notes.md"],
            citations=['"fast ANN"'],
        )
        writer.write_page(page)

        got = writer.read_page("sqlite-vec")
        assert got is not None
        assert got.page_id == "sqlite-vec"
        assert got.title == "SQLite-Vec"
        assert got.body == "SQLite-Vec is a vector search extension."
        assert got.page_type == "concept"
        assert got.topic_cluster == "vector-store"
        assert got.status == "published"
        assert "notes.md" in got.sources
        assert got.hash  # derived from page_id, 12 hex chars

    def test_hash_is_deterministic(self):
        """Same page_id always yields the same 12-char hash."""
        a = WikiPage(page_id="x", title="X")
        b = WikiPage(page_id="x", title="different title")
        assert a.hash == b.hash
        assert len(a.hash) == 12

    def test_upsert_preserves_created_and_merges(self, writer):
        """Re-ingesting the same page keeps created, refreshes updated, unions sources."""
        first = WikiPage(page_id="pg", title="Pg", body="first body",
                         sources=["a.md"])
        writer.write_page(first)
        created_before = writer.read_page("pg").created

        # Upsert with new content + a new source
        writer.upsert(WikiPage(page_id="pg", title="Pg", body="second body",
                               sources=["b.md"]))
        got = writer.read_page("pg")

        assert got.created == created_before          # preserved
        assert "first body" in got.body                # appended, not overwritten
        assert "second body" in got.body
        assert set(got.sources) == {"a.md", "b.md"}    # union

    def test_upsert_is_idempotent(self, writer):
        """Ingesting identical content twice should not duplicate the body."""
        page = WikiPage(page_id="pg", title="Pg", body="stable body")
        writer.upsert(page)
        writer.upsert(page)
        got = writer.read_page("pg")
        assert got.body.count("stable body") == 1

    def test_list_pages_excludes_reserved(self, writer, tmp_path):
        """index.md and log.md are structural and must not appear as pages."""
        writer.write_page(WikiPage(page_id="real-page", title="Real"))
        writer.rebuild_index()
        writer.append_log("ingest", "real-page")
        pages = writer.list_pages()
        assert "real-page" in pages
        assert "index" not in pages
        assert "log" not in pages
        # list_all includes them
        assert "index" in writer.list_all()

    def test_rebuild_index_groups_by_cluster(self, writer):
        """index.md should contain cluster headings and one row per page."""
        writer.write_page(WikiPage(page_id="a", title="A",
                                   topic_cluster="db", body="about databases"))
        writer.write_page(WikiPage(page_id="b", title="B",
                                   topic_cluster="db", body="more databases"))
        writer.write_page(WikiPage(page_id="c", title="C",
                                   topic_cluster="llm", body="about llms"))
        writer.rebuild_index()
        idx = writer.read_index()
        assert "[[wiki/a]]" in idx and "[[wiki/b]]" in idx and "[[wiki/c]]" in idx
        assert "## db" in idx and "## llm" in idx

    def test_update_related_writes_wikilinks(self, writer):
        """update_related should rewrite the Related: line with [[ref]] links."""
        writer.write_page(WikiPage(page_id="a", title="A"))
        writer.update_related("a", ["wiki/b", "wiki/c"])
        raw = writer.page_path("a").read_text()
        assert "[[wiki/b]]" in raw and "[[wiki/c]]" in raw


# ══════════════════════════════════════════════════════════════════════════════
# WikiIngestor (Mock extractor, offline)
# ══════════════════════════════════════════════════════════════════════════════

class TestWikiIngest:

    def test_splits_markdown_into_pages(self, ingestor, tmp_path, writer):
        """One H1 section → one wiki page."""
        doc = _md_doc(tmp_path / "notes.md", [
            ("SQLite-Vec", "SQLite-Vec is a vector search extension for SQLite."),
            ("PostgreSQL", "PostgreSQL is a relational database."),
        ])
        pages = ingestor.ingest(str(doc))
        assert set(pages) == {"sqlite-vec", "postgresql"}
        # Both pages persisted to disk
        assert set(writer.list_pages()) == {"sqlite-vec", "postgresql"}

    def test_cluster_inference(self, ingestor, tmp_path):
        """The Mock extractor tags pages with the shared domain keyword map."""
        doc = _md_doc(tmp_path / "notes.md", [
            ("SQLite-Vec", "vector search sqlite"),
        ])
        ingestor.ingest(str(doc))
        # Direct check of the inference function the extractor uses
        assert infer_cluster("sqlite vector embedding") == "vector-store"

    def test_no_heading_yields_single_page(self, ingestor, tmp_path):
        """A doc with no headings becomes one page titled after the source."""
        doc = tmp_path / "flat.txt"
        doc.write_text("Just a flat blob of text with no headings.", encoding="utf-8")
        pages = ingestor.ingest(str(doc))
        assert len(pages) == 1

    def test_reingest_is_idempotent(self, ingestor, tmp_path, writer):
        """Ingesting the same doc twice must not duplicate pages or bodies."""
        doc = _md_doc(tmp_path / "notes.md", [("SQLite-Vec", "stable body text")])
        first = ingestor.ingest(str(doc))
        second = ingestor.ingest(str(doc))
        assert first == second == ["sqlite-vec"]
        body = writer.read_page("sqlite-vec").body
        assert body.count("stable body text") == 1

    def test_index_and_log_written(self, ingestor, tmp_path):
        """Ingest must refresh index.md and append to log.md."""
        from pathlib import Path
        doc = _md_doc(tmp_path / "notes.md", [("SQLite-Vec", "vector search")])
        ingestor.ingest(str(doc))
        wiki = Path(ingestor.vault_path) / "wiki"
        assert (wiki / "index.md").exists()
        assert (wiki / "log.md").exists()
        assert "ingest" in (wiki / "log.md").read_text()


# ══════════════════════════════════════════════════════════════════════════════
# WikiLinter
# ══════════════════════════════════════════════════════════════════════════════

class TestWikiLint:

    def test_find_orphans(self, vault, writer):
        """A page with no inbound [[wiki/...]] ref from thoughts/clusters is an orphan."""
        writer.write_page(WikiPage(page_id="lonely", title="Lonely"))
        writer.write_page(WikiPage(page_id="linked", title="Linked"))
        # Reference 'linked' from a thoughts file
        from pathlib import Path
        thoughts = Path(vault) / "thoughts"
        thoughts.mkdir(parents=True, exist_ok=True)
        (thoughts / "abc123.md").write_text(
            "---\nstatus: settled\n---\n\nSee [[wiki/linked]] for detail.\n",
            encoding="utf-8",
        )
        linter = WikiLinter(vault)
        orphans = linter.find_orphans()
        assert "lonely" in orphans
        assert "linked" not in orphans

    def test_find_contradictions_same_domain(self, vault, writer):
        """Two pages in the same cluster asserting different DBs should flag."""
        writer.write_page(WikiPage(page_id="pref-mysql", title="MySQL",
                                   topic_cluster="database",
                                   body="I use MySQL as my database."))
        writer.write_page(WikiPage(page_id="pref-postgres", title="Postgres",
                                   topic_cluster="database",
                                   body="I switched to PostgreSQL as my database."))
        linter = WikiLinter(vault)
        contras = linter.find_contradictions()
        # Should flag at least one pair in the 'database' domain
        assert len(contras) >= 1
        domains = {c[2] for c in contras}
        assert "database" in domains

    def test_mark_answered_patches_thought_frontmatter(self, vault, writer):
        """mark_answered writes status:answered + answered_by onto the thought."""
        from pathlib import Path
        # Set up a wiki page to answer with
        writer.write_page(WikiPage(page_id="sqlite-vec", title="SQLite-Vec"))
        # Set up a question thought
        thoughts = Path(vault) / "thoughts"
        thoughts.mkdir(parents=True, exist_ok=True)
        qhash = "abcdef123456"
        (thoughts / f"{qhash}.md").write_text(
            "---\nstatus: open\ntype: question\n---\n\nHow does sqlite-vec work?\n",
            encoding="utf-8",
        )
        linter = WikiLinter(vault)
        ok = linter.mark_answered(qhash, "sqlite-vec")
        assert ok is True
        patched = (thoughts / f"{qhash}.md").read_text()
        assert "status: answered" in patched
        assert "[[wiki/sqlite-vec]]" in patched

    def test_mark_answered_rejects_missing_page(self, vault):
        """Answering with a non-existent wiki page must return False."""
        linter = WikiLinter(vault)
        assert linter.mark_answered("anyhash", "does-not-exist") is False

    def test_run_returns_all_four_checks(self, vault, writer):
        """run() returns a report dict with the four expected keys."""
        writer.write_page(WikiPage(page_id="x", title="X"))
        linter = WikiLinter(vault)
        report = linter.run()
        assert set(report.keys()) == {
            "open_questions", "orphan_pages", "contradictions", "coverage_gaps"
        }


# ══════════════════════════════════════════════════════════════════════════════
# wiki_signal (Phase 6 query surface, exercised here for wiki completeness)
# ══════════════════════════════════════════════════════════════════════════════

class TestWikiSignal:

    def test_returns_thought_node_shape(self, vault, ingestor, tmp_path):
        """wiki_signal must return dicts shaped like thought nodes for ContextBuilder."""
        from noesis.context.signals import wiki_signal
        doc = _md_doc(tmp_path / "notes.md", [
            ("SQLite-Vec", "SQLite-Vec is a vector search extension."),
        ])
        ingestor.ingest(str(doc))
        hits = wiki_signal("vector search", vault, top_k=3)
        assert len(hits) >= 1
        h = hits[0]
        # ContextBuilder treats these uniformly with thoughts, so the shape matters
        assert h["type"] == "wiki"
        assert h["id"] == "wiki/sqlite-vec"
        assert "[[wiki/sqlite-vec]]" in h["source"]
        assert "score" in h

    def test_empty_vault_returns_empty(self, vault):
        """No wiki dir / no pages → no hits, never raises."""
        from noesis.context.signals import wiki_signal
        assert wiki_signal("anything", vault) == []

    def test_ranks_relevant_page_first(self, vault, ingestor, tmp_path):
        """The query-relevant page should outrank an unrelated one."""
        from noesis.context.signals import wiki_signal
        doc = _md_doc(tmp_path / "notes.md", [
            ("SQLite-Vec", "vector search sqlite ANN"),
            ("Cooking", "pasta tomato basil recipe"),
        ])
        ingestor.ingest(str(doc))
        hits = wiki_signal("vector search sqlite", vault, top_k=2)
        assert hits[0]["id"] == "wiki/sqlite-vec"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_slugify(self):
        assert _slugify("SQLite-Vec") == "sqlite-vec"
        assert _slugify("PostgreSQL 16.x") == "postgresql-16x"
        assert _slugify("") == "untitled"
