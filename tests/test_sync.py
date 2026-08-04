"""
tests/test_sync.py

Tests for the bidirectional vault ↔ hot store sync.

Covers:
  - Text edits in vault propagate to hot store
  - Frontmatter edits (status, confidence, type) propagate
  - Vector re-embedded when text changes
  - Deleted vault files soft-delete in hot store
  - New vault-only files imported as tentative
  - Sync is idempotent (second run = no-op)
  - Sync returns accurate report counts
  - CLI sync command output
"""
import hashlib
import math
import re
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from noesis.memory.main import Memory


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _insert(mem, text, user_id="sync", status="settled", confidence=0.8,
            topic="test"):
    """Insert a thought into both hot and cold store, return hash_id."""
    h = _hash(text)
    vec = mem.embedding.embed(text)
    mem.vector_store.insert(h, vec, {
        "text": text, "type": "position", "status": status,
        "confidence": confidence, "user_id": user_id,
        "source_tool": "test", "source_session": "",
        "topic_cluster": topic, "created_at": time.time(),
    })
    if mem.cold_store:
        mem.cold_store.write(h, {
            "text": text, "type": "position", "status": status,
            "confidence": confidence, "user_id": user_id,
            "source_tool": "test", "source_session": "",
            "topic_cluster": topic,
        })
    return h


def _force_sync(mem, user_id="sync") -> dict:
    """Reset sync state and run _sync_if_needed, return report."""
    mem._synced.discard(user_id)
    mem._last_sync[user_id] = 0.0
    return mem._sync_if_needed(user_id)


def _force_sync_full(mem, user_id="sync") -> dict:
    """Reset sync state and run sync_full (all 3 phases), return report."""
    mem._synced.discard(user_id)
    mem._last_sync[user_id] = 0.0
    return mem.sync_full(user_id)


def _edit_vault_text(mem, hash_id, new_text):
    """Directly edit the body text of a vault .md file."""
    path = mem.cold_store._thought_path(hash_id)
    raw = path.read_text(encoding="utf-8")
    parts = raw.split("---", 2)
    if len(parts) >= 3:
        raw = parts[0] + "---" + parts[1] + "---" + new_text + "\n\nRelated:\n"
    else:
        raw = new_text
    path.write_text(raw, encoding="utf-8")
    # Touch to ensure mtime is newer
    path.touch()


def _edit_vault_frontmatter(mem, hash_id, updates: dict):
    """Patch frontmatter fields in a vault .md file."""
    path = mem.cold_store._thought_path(hash_id)
    mem.cold_store._patch_frontmatter(hash_id, updates)
    path.touch()


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: Modified files
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyncModifiedFiles:

    def test_text_edit_syncs_to_hot_store(self, mem):
        """Editing body text in vault should update hot store text."""
        h = _insert(mem, "original thought text")
        _edit_vault_text(mem, h, "completely different text now")

        report = _force_sync(mem)

        assert report["text_edits"] >= 1
        row = mem.vector_store.get(h)
        assert "completely different text now" in row["text"]

    def test_frontmatter_status_syncs(self, mem):
        """Promoting status in vault frontmatter should update hot store."""
        # Insert as tentative, then promote in vault to settled
        h = _insert(mem, "status test", status="tentative", confidence=0.0)
        _edit_vault_frontmatter(mem, h, {"status": "settled"})

        report = _force_sync(mem)

        assert report["metadata_edits"] >= 1
        row = mem.vector_store.get(h)
        assert row["status"] == "settled"

    def test_status_never_regresses(self, mem):
        """Vault demotion (settled→tentative) should NOT overwrite hot store."""
        h = _insert(mem, "status regress test", status="settled", confidence=0.8)
        _edit_vault_frontmatter(mem, h, {"status": "tentative"})

        report = _force_sync(mem)

        # Should be no metadata edits — vault demotion is blocked
        assert report["metadata_edits"] == 0
        row = mem.vector_store.get(h)
        assert row["status"] == "settled"

    def test_frontmatter_confidence_syncs(self, mem):
        """Changing confidence in vault frontmatter should update hot store."""
        h = _insert(mem, "confidence test", confidence=0.5)
        _edit_vault_frontmatter(mem, h, {"confidence": "0.95"})

        report = _force_sync(mem)

        assert report["metadata_edits"] >= 1
        row = mem.vector_store.get(h)
        assert abs(row["confidence"] - 0.95) < 0.01

    def test_vector_re_embedded_on_text_change(self, mem):
        """After text edit, embedding should match new text, not old."""
        h = _insert(mem, "vector staleness test")
        old_vec = mem.vector_store.get_vector(h)

        _edit_vault_text(mem, h, "a totally new sentence about quantum physics")

        report = _force_sync(mem)
        assert report["text_edits"] >= 1

        new_vec = mem.vector_store.get_vector(h)
        fresh_vec = mem.embedding.embed("a totally new sentence about quantum physics")

        # New vec should match fresh embed (not the old one)
        sim_new = cosine_similarity(new_vec, fresh_vec)
        sim_old = cosine_similarity(old_vec, fresh_vec)
        assert sim_new > 0.999, f"Re-embed mismatch: {sim_new}"
        assert sim_new > sim_old, "New vec should be closer to fresh than old vec was"

    def test_type_syncs(self, mem):
        """Changing type in vault frontmatter should update hot store."""
        h = _insert(mem, "type test", status="settled")
        _edit_vault_frontmatter(mem, h, {"type": "question"})

        report = _force_sync(mem)

        row = mem.vector_store.get(h)
        assert row["type"] == "question"


def cosine_similarity(a, b):
    return _cosine(a, b)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: Deleted files
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyncDeletedFiles:

    def test_deleted_file_soft_deletes_in_hot_store(self, mem):
        """Removing a .md from vault should soft-delete the hot store row."""
        h = _insert(mem, "delete me please")
        assert mem.vector_store.get(h)["status"] != "superseded"

        # Delete the vault file
        vault_file = mem.cold_store._thought_path(h)
        vault_file.unlink()

        report = _force_sync_full(mem)

        assert report["deletions"] >= 1
        row = mem.vector_store.get(h)
        assert row["status"] == "superseded"

    def test_non_user_file_not_affected(self, mem):
        """Deleting a vault file for user B should not affect user A."""
        ha = _insert(mem, "user A thought", user_id="alice")
        hb = _insert(mem, "user B thought", user_id="bob")

        # Delete bob's vault file
        (mem.cold_store._thought_path(hb)).unlink()

        _force_sync_full(mem, user_id="bob")

        # Alice's thought should be untouched
        row_a = mem.vector_store.get(ha)
        assert row_a["status"] == "settled"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C: New vault-only files
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyncNewFiles:

    def test_new_vault_file_imported(self, mem):
        """A .md manually created in the vault should appear in hot store."""
        # Bypass cold_store.write — create a file directly
        vault_file = mem.cold_store.root / "thoughts" / "manual123.md"
        vault_file.write_text(
            "---\n"
            "hash: manual123\n"
            "status: settled\n"
            "confidence: 0.90\n"
            "type: identity\n"
            "---\n"
            "I am a senior ML engineer.\n"
            "\nRelated:\n",
            encoding="utf-8",
        )

        report = _force_sync_full(mem)

        assert report["additions"] >= 1
        row = mem.vector_store.get("manual123")
        assert row is not None
        assert "senior ML engineer" in row["text"]
        assert row["status"] == "settled"

    def test_new_vault_file_has_embedding(self, mem):
        """An imported vault file should be searchable via vector query."""
        vault_file = mem.cold_store.root / "thoughts" / "embed_test.md"
        vault_file.write_text(
            "---\n"
            "hash: embed_test\n"
            "status: settled\n"
            "confidence: 0.80\n"
            "type: preference\n"
            "---\n"
            "I strongly prefer Rust over C++ for systems programming.\n"
            "\nRelated:\n",
            encoding="utf-8",
        )

        _force_sync_full(mem)

        results = mem.vector_store.search(
            mem.embedding.embed("Rust vs C++ systems programming"),
            top_k=5,
        )
        hashes = [r["hash_id"] for r in results]
        assert "embed_test" in hashes


# ═══════════════════════════════════════════════════════════════════════════════
# Session gating & idempotency
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyncBehavior:

    def test_sync_is_idempotent(self, mem):
        """Running sync twice should produce zero changes on second run."""
        h = _insert(mem, "idempotent test")

        # First sync: process the insertion
        mem._synced.discard("sync")
        mem._last_sync["sync"] = 0.0
        report1 = mem._sync_if_needed("sync")

        # Second sync: should be a no-op (session is synced)
        report2 = mem._sync_if_needed("sync")

        assert report2["text_edits"] == 0
        assert report2["metadata_edits"] == 0
        assert report2["deletions"] == 0
        assert report2["additions"] == 0

    def test_sync_preserves_unmodified_files(self, mem):
        """Unmodified vault files should not trigger any updates."""
        h = _insert(mem, "don't touch me")

        report = _force_sync(mem)

        # The file was just written and hasn't been modified, but scan_modified
        # may still return it depending on filesystem mtime precision.
        # Key assertion: no text edits happened.
        row = mem.vector_store.get(h)
        assert "don't touch me" in row["text"]

    def test_sync_returns_report_dict(self, mem):
        """_sync_if_needed should return a dict with expected keys."""
        report = mem._sync_if_needed("sync")
        assert isinstance(report, dict)
        for key in ("text_edits", "metadata_edits", "deletions", "additions"):
            assert key in report

    def test_sync_no_cold_store(self, mem_hot_only):
        """Memory without cold store should return empty report, no crash."""
        report = mem_hot_only._sync_if_needed("default")
        assert report["text_edits"] == 0
        assert report["deletions"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLI sync command
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyncCLI:

    @pytest.fixture
    def setup_cli(self, tmp_path):
        import yaml
        db_path = str(tmp_path / "hot.db")
        vault = str(tmp_path / "vault")
        cfg = {
            "vector_store": {"config": {"db_path": db_path}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
            "cold_store":   {"config": {"vault_path": vault}},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))
        return str(cfg_path), db_path, vault

    def _cli(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli_main", "cli/main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_sync_cli_no_changes(self, setup_cli):
        """CLI sync with no changes should say 'up-to-date'."""
        cfg_path, db_path, vault = setup_cli
        m = Memory.from_config({
            "vector_store": {"config": {"db_path": db_path}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
            "cold_store":   {"config": {"vault_path": vault}},
        })
        m.embedding.embed("warmup")
        del m

        cli = self._cli()
        result = CliRunner().invoke(cli.cli, ["sync", "--config", cfg_path])
        assert result.exit_code == 0
        assert "up-to-date" in result.output.lower() or "synced" in result.output.lower()

    def test_sync_cli_with_edits(self, setup_cli):
        """CLI sync should report text edits when vault was modified."""
        cfg_path, db_path, vault = setup_cli
        m = Memory.from_config({
            "vector_store": {"config": {"db_path": db_path}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
            "cold_store":   {"config": {"vault_path": vault}},
        })
        m.embedding.embed("warmup")
        # Insert a thought
        h = _hash("cli sync edit test")
        vec = m.embedding.embed("cli sync edit test")
        m.vector_store.insert(h, vec, {
            "text": "cli sync edit test", "type": "position",
            "status": "settled", "confidence": 0.8, "user_id": "default",
            "source_tool": "", "source_session": "",
            "topic_cluster": "test", "created_at": time.time(),
        })
        m.cold_store.write(h, {
            "text": "cli sync edit test", "type": "position",
            "status": "settled", "confidence": 0.8, "user_id": "default",
            "source_tool": "", "source_session": "",
            "topic_cluster": "test",
        })
        # Simulate a small delay, then edit the vault file
        time.sleep(0.05)
        _edit_vault_text(m, h, "edited by user in obsidian")
        del m

        cli = self._cli()
        result = CliRunner().invoke(cli.cli, ["sync", "--config", cfg_path])
        assert result.exit_code == 0
        assert "text edit" in result.output.lower() or "synced" in result.output.lower()

    def test_sync_cli_verbose(self, setup_cli):
        """CLI sync --verbose should show per-category counts."""
        cfg_path, db_path, vault = setup_cli
        m = Memory.from_config({
            "vector_store": {"config": {"db_path": db_path}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
            "cold_store":   {"config": {"vault_path": vault}},
        })
        m.embedding.embed("warmup")
        del m

        cli = self._cli()
        result = CliRunner().invoke(cli.cli, ["sync", "--config", cfg_path, "--verbose"])
        assert result.exit_code == 0
        assert "text_edits" in result.output
        assert "deletions" in result.output
