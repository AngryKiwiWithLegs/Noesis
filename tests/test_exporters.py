"""
tests/test_exporters.py

Tests for the memory export feature (JSON, Markdown, fine-tuning formats).

Covers:
  - JSON export: schema completeness, round-trip through importer
  - Markdown export: frontmatter format, file structure
  - Fine-tuning export: only confirmed thoughts, OpenAI chat format
  - CLI: --dry-run, --include-wiki, --include-superseded flags
  - count_exportable preview function
"""
import json
import time
from pathlib import Path

import pytest

from noesis.exporters import (
    export,
    count_exportable,
    _thought_to_markdown,
    _epoch_to_iso,
    _split_to_dialogue,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _insert_thought(memory, text, status="settled", confidence=0.8,
                    topic="database", ttype="preference", user_id="test"):
    """Insert a thought directly into the vector store and return its hash."""
    vec = memory.embedding.embed(text)
    h = memory.vector_store._hash(text) if hasattr(memory.vector_store, '_hash') else None
    if h is None:
        import hashlib
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
    memory.vector_store.insert(h, vec, {
        "text": text, "type": ttype, "status": status,
        "confidence": confidence, "user_id": user_id,
        "source_tool": "test", "source_session": "",
        "topic_cluster": topic, "created_at": time.time(),
    })
    return h


# ═══════════════════════════════════════════════════════════════════════════════
# Helper unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_epoch_to_iso(self):
        ts = _epoch_to_iso(1700000000.0)
        assert ts is not None
        assert ts.startswith("2023-11-14T")

    def test_epoch_to_iso_none(self):
        assert _epoch_to_iso(None) is None

    def test_thought_to_markdown_has_frontmatter(self):
        t = {
            "hash_id": "abc123",
            "text": "I prefer PostgreSQL",
            "type": "preference",
            "status": "settled",
            "confidence": 0.9,
            "user_id": "u1",
            "topic_cluster": "database",
            "source_tool": "test",
            "source_session": "",
            "created_at": 1700000000.0,
        }
        md = _thought_to_markdown(t)
        assert md.startswith("---")
        assert "hash: abc123" in md
        assert "status: settled" in md
        assert "confidence: 0.90" in md
        assert "I prefer PostgreSQL" in md

    def test_thought_to_markdown_includes_optional_fields(self):
        t = {
            "hash_id": "abc",
            "text": "test",
            "type": "position",
            "status": "settled",
            "confidence": 0.8,
            "user_id": "u1",
            "topic_cluster": "",
            "source_tool": "",
            "source_session": "",
            "created_at": 1700000000.0,
            "fact_ref": "[[wiki/postgres]]",
        }
        md = _thought_to_markdown(t)
        assert "fact_ref: [[wiki/postgres]]" in md

    def test_split_dialogue_with_role_labels(self):
        text = "user: I like Python\nassistant: That's great!"
        user, asst = _split_to_dialogue(text, {})
        assert "I like Python" in user
        assert "That's great" in asst

    def test_split_dialogue_single_assertion(self):
        text = "I prefer Go for backend services"
        user, asst = _split_to_dialogue(text, {"type": "preference"})
        assert len(user) > 0
        assert "Go for backend" in asst


# ═══════════════════════════════════════════════════════════════════════════════
# count_exportable
# ═══════════════════════════════════════════════════════════════════════════════

class TestCountExportable:
    def test_counts_by_status(self, mem_hot_only):
        _insert_thought(mem_hot_only, "settled thought one", status="settled")
        _insert_thought(mem_hot_only, "settled thought two", status="settled")
        _insert_thought(mem_hot_only, "provisional thought", status="provisional")
        _insert_thought(mem_hot_only, "tentative thought", status="tentative")

        counts = count_exportable(mem_hot_only, "test")
        assert counts["total"] == 4
        assert counts["confirmed"] == 3   # 2 settled + 1 provisional
        assert counts["tentative"] == 1

    def test_empty_store(self, mem_hot_only):
        counts = count_exportable(mem_hot_only, "test")
        assert counts["total"] == 0
        assert counts["confirmed"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# JSON export
# ═══════════════════════════════════════════════════════════════════════════════

class TestJSONExport:
    def test_basic_export(self, mem_hot_only, tmp_path):
        h = _insert_thought(mem_hot_only, "I prefer PostgreSQL for data", status="settled")
        out = tmp_path / "export.json"

        result = export("json", mem_hot_only, "test", out)
        assert result["thoughts_exported"] == 1
        assert result["format"] == "json"

        data = json.loads(out.read_text())
        assert data["_meta"]["format"] == "noesis-json-v1"
        assert data["_meta"]["thought_count"] == 1
        assert len(data["conversations"]) == 1

        entry = data["conversations"][0]
        assert entry["hash_id"] == h
        assert entry["text"] == "I prefer PostgreSQL for data"
        assert entry["status"] == "settled"
        assert entry["type"] == "preference"
        assert entry["topic_cluster"] == "database"

    def test_schema_completeness(self, mem_hot_only, tmp_path):
        """Every thought field from the schema should be present."""
        _insert_thought(mem_hot_only, "test content", status="settled")
        out = tmp_path / "export.json"
        export("json", mem_hot_only, "test", out)

        data = json.loads(out.read_text())
        entry = data["conversations"][0]
        expected_fields = {
            "hash_id", "text", "type", "status", "confidence",
            "user_id", "source_tool", "source_session", "topic_cluster",
            "created_at", "fact_ref", "evolved_from", "superseded_by", "extra",
        }
        assert expected_fields.issubset(entry.keys())

    def test_timestamp_converted_to_iso(self, mem_hot_only, tmp_path):
        fixed_ts = 1700000000.0
        _insert_thought(mem_hot_only, "ts test", status="settled", )
        # Update the created_at directly
        mem_hot_only.vector_store._con.execute(
            "UPDATE items SET created_at=? WHERE user_id=?", [fixed_ts, "test"]
        )
        mem_hot_only.vector_store._con.commit()

        out = tmp_path / "export.json"
        export("json", mem_hot_only, "test", out)
        data = json.loads(out.read_text())
        entry = data["conversations"][0]
        assert "T" in entry["created_at"]  # ISO format
        assert entry["created_at"].startswith("2023-11")

    def test_multiple_thoughts(self, mem_hot_only, tmp_path):
        for i in range(5):
            _insert_thought(mem_hot_only, f"thought number {i}", status="settled")
        out = tmp_path / "export.json"
        result = export("json", mem_hot_only, "test", out)
        assert result["thoughts_exported"] == 5

    def test_excludes_superseded_by_default(self, mem_hot_only, tmp_path):
        _insert_thought(mem_hot_only, "active thought", status="settled")
        _insert_thought(mem_hot_only, "old thought", status="superseded")
        out = tmp_path / "export.json"
        result = export("json", mem_hot_only, "test", out)
        assert result["thoughts_exported"] == 1

    def test_includes_superseded_when_requested(self, mem_hot_only, tmp_path):
        _insert_thought(mem_hot_only, "active thought", status="settled")
        _insert_thought(mem_hot_only, "old thought", status="superseded")
        out = tmp_path / "export.json"
        result = export("json", mem_hot_only, "test", out, include_superseded=True)
        assert result["thoughts_exported"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown export
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarkdownExport:
    def test_basic_export(self, mem_hot_only, tmp_path):
        h = _insert_thought(mem_hot_only, "markdown test content", status="settled")
        out_dir = tmp_path / "md_export"

        result = export("markdown", mem_hot_only, "test", out_dir)
        assert result["thoughts_exported"] == 1
        assert (out_dir / "thoughts" / f"{h}.md").exists()

    def test_frontmatter_format(self, mem_hot_only, tmp_path):
        h = _insert_thought(mem_hot_only, "frontmatter test", status="settled",
                           confidence=0.85)
        out_dir = tmp_path / "md"
        export("markdown", mem_hot_only, "test", out_dir)

        md = (out_dir / "thoughts" / f"{h}.md").read_text()
        # Frontmatter block is delimited by --- at start and end
        assert md.startswith("---\n")
        assert "hash:" in md
        assert "status: settled" in md
        assert "confidence: 0.85" in md
        # Count the --- delimiters: opening and closing
        assert md.count("\n---\n") >= 1  # closing fence exists

    def test_multiple_files(self, mem_hot_only, tmp_path):
        for i in range(3):
            _insert_thought(mem_hot_only, f"md thought {i}", status="settled")
        out_dir = tmp_path / "md"
        result = export("markdown", mem_hot_only, "test", out_dir)
        assert result["thoughts_exported"] == 3
        files = list((out_dir / "thoughts").glob("*.md"))
        assert len(files) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-tuning export
# ═══════════════════════════════════════════════════════════════════════════════

class TestFinetuneExport:
    def test_only_confirmed_thoughts(self, mem_hot_only, tmp_path):
        _insert_thought(mem_hot_only, "confirmed thought", status="settled")
        _insert_thought(mem_hot_only, "tentative thought", status="tentative")
        _insert_thought(mem_hot_only, "old thought", status="superseded")

        out = tmp_path / "train.jsonl"
        result = export("finetune", mem_hot_only, "test", out)
        assert result["thoughts_exported"] == 1  # only the settled one

    def test_jsonl_format(self, mem_hot_only, tmp_path):
        _insert_thought(mem_hot_only, "jsonl test content", status="settled")
        out = tmp_path / "train.jsonl"
        export("finetune", mem_hot_only, "test", out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "messages" in entry
        msgs = entry["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_empty_if_all_tentative(self, mem_hot_only, tmp_path):
        _insert_thought(mem_hot_only, "tentative only", status="tentative")
        out = tmp_path / "train.jsonl"
        result = export("finetune", mem_hot_only, "test", out)
        assert result["thoughts_exported"] == 0
        assert out.read_text().strip() == ""

    def test_multiple_entries(self, mem_hot_only, tmp_path):
        for i in range(5):
            _insert_thought(mem_hot_only, f"confirmed fact {i}", status="settled")
        out = tmp_path / "train.jsonl"
        result = export("finetune", mem_hot_only, "test", out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Round-trip: JSON export → JSON import
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    def test_json_export_then_import(self, mem_hot_only, tmp_path):
        """Exported JSON should be re-importable via the json importer."""
        from noesis.importers import normalize

        _insert_thought(mem_hot_only, "round trip content here", status="settled",
                       user_id="rt")
        out = tmp_path / "roundtrip.json"
        export("json", mem_hot_only, "rt", out)

        # Now import it into a fresh memory instance
        from noesis.memory.main import Memory
        mem2 = Memory.from_config({
            "vector_store": {"config": {"db_path": str(tmp_path / "rt.db")}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        })
        mem2.embedding.embed("warmup")

        convos = normalize("json", out)
        assert len(convos) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportCLI:
    @pytest.fixture
    def setup_db(self, tmp_path):
        import yaml
        from noesis.memory.main import Memory

        db_path = str(tmp_path / "hot.db")
        vault = str(tmp_path / "vault")
        cfg = {
            "vector_store": {"config": {"db_path": db_path}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
            "cold_store":   {"config": {"vault_path": vault}},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        m = Memory.from_config(cfg)
        m.embedding.embed("warmup")
        _insert_thought(m, "cli settled thought", status="settled", user_id="cli")
        _insert_thought(m, "cli tentative thought", status="tentative", user_id="cli")
        del m
        return str(cfg_path), db_path

    def _cli(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli_main", "cli/main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_json_export_cli(self, setup_db, tmp_path):
        from click.testing import CliRunner
        cfg_path, db_path = setup_db
        mod = self._cli()
        out = tmp_path / "cli_export.json"

        result = CliRunner().invoke(mod.cli, [
            "export", str(out),
            "--format", "json",
            "--user", "cli",
            "--config", cfg_path,
        ])
        assert result.exit_code == 0
        assert "Exported" in result.output
        assert out.exists()
        data = json.loads(out.read_text())
        # get_all excludes superseded but includes tentative — both thoughts present
        assert len(data["conversations"]) == 2

    def test_dry_run(self, setup_db, tmp_path):
        from click.testing import CliRunner
        cfg_path, db_path = setup_db
        mod = self._cli()
        out = tmp_path / "dry.json"

        result = CliRunner().invoke(mod.cli, [
            "export", str(out),
            "--format", "json",
            "--user", "cli",
            "--config", cfg_path,
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert not out.exists()

    def test_finetune_cli(self, setup_db, tmp_path):
        from click.testing import CliRunner
        cfg_path, db_path = setup_db
        mod = self._cli()
        out = tmp_path / "train.jsonl"

        result = CliRunner().invoke(mod.cli, [
            "export", str(out),
            "--format", "finetune",
            "--user", "cli",
            "--config", cfg_path,
        ])
        assert result.exit_code == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        # Only the settled thought should appear
        assert len(lines) == 1

    def test_markdown_cli(self, setup_db, tmp_path):
        from click.testing import CliRunner
        cfg_path, db_path = setup_db
        mod = self._cli()
        out_dir = tmp_path / "md_out"

        result = CliRunner().invoke(mod.cli, [
            "export", str(out_dir),
            "--format", "markdown",
            "--user", "cli",
            "--config", cfg_path,
        ])
        assert result.exit_code == 0
        assert (out_dir / "thoughts").exists()

    def test_unknown_format_errors(self, setup_db, tmp_path):
        from click.testing import CliRunner
        cfg_path, db_path = setup_db
        mod = self._cli()
        out = tmp_path / "bad.xyz"

        result = CliRunner().invoke(mod.cli, [
            "export", str(out),
            "--format", "invalid",
            "--user", "cli",
            "--config", cfg_path,
        ])
        assert result.exit_code != 0
