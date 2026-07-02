"""
tests/test_clustering.py

Tests for the improved topic-cluster inference and the recluster command.
No API keys needed — keyword inference is pure, embedding inference uses the
local all-MiniLM-L6-v2 model.

Run:
    pytest tests/test_clustering.py -v
"""
import pytest

from noesis.thoughts.extractor import (
    _infer_cluster,
    _infer_cluster_embedding,
    _DOMAIN_KEYWORDS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Keyword inference
# ══════════════════════════════════════════════════════════════════════════════

class TestKeywordInference:

    def test_each_domain_has_keywords(self):
        """Sanity: the domain map is populated and non-empty."""
        assert len(_DOMAIN_KEYWORDS) >= 10
        for cluster, kws in _DOMAIN_KEYWORDS:
            assert cluster, "cluster label must be non-empty"
            assert kws, f"cluster {cluster} has no keywords"

    def test_database_domain(self):
        assert _infer_cluster("I prefer PostgreSQL over MySQL") == "database"
        assert _infer_cluster("我倾向于用 PostgreSQL，因为 JSON 支持更好") == "database"

    def test_languages_domain(self):
        assert _infer_cluster("I prefer Python for backend") == "languages"
        assert _infer_cluster("我决定学习 Rust 做系统编程") == "languages"

    def test_vector_store_domain(self):
        assert _infer_cluster("sqlite-vec is lighter than FAISS") == "vector-store"
        assert _infer_cluster("我用 sqlite-vec 做向量检索") == "vector-store"

    def test_frontend_domain(self):
        assert _infer_cluster("I use Tailwind CSS for styling") == "frontend"
        assert _infer_cluster("我是前端工程师，精通 React") == "frontend"

    def test_cloud_infra_domain(self):
        assert _infer_cluster("We deploy with Kubernetes on AWS") == "cloud-infra"
        assert _infer_cluster("我喜欢用 Docker Compose 做本地开发") == "cloud-infra"

    def test_general_fallback(self):
        """Genuinely topic-agnostic statements should stay 'general'."""
        assert _infer_cluster("I think premature optimization is the root of evil") == "general"
        assert _infer_cluster("我认为过早优化是万恶之源") == "general"

    def test_first_match_wins(self):
        """A text matching multiple domains should pick the first-listed
        (most specific) domain, not the last."""
        # "vector" appears in both vector-store and llm-choice maps, but
        # vector-store is listed first.
        result = _infer_cluster("vector embedding with sqlite-vec")
        assert result == "vector-store"

    def test_empty_text(self):
        assert _infer_cluster("") == "general"
        assert _infer_cluster(None) == "general"


# ══════════════════════════════════════════════════════════════════════════════
# Embedding-aware inference
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingInference:

    @pytest.fixture
    def embedder(self, tmp_path):
        from noesis.memory.main import Memory
        m = Memory.from_config({
            "vector_store": {"config": {"db_path": str(tmp_path / "h.db")}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        })
        m.embedding.embed("warmup")
        return m.embedding

    def test_keyword_short_circuits(self, embedder):
        """If keywords clearly match, embedding isn't needed."""
        # 'PostgreSQL' matches database keyword — should return immediately
        result = _infer_cluster_embedding(
            "I use PostgreSQL", embedder, existing=[]
        )
        assert result == "database"

    def test_no_existing_returns_keyword_or_general(self, embedder):
        """With no reference nodes, falls back to keyword, then general."""
        assert _infer_cluster_embedding(
            "random opinion about life", embedder, existing=[]
        ) == "general"

    def test_assigns_to_nearest_existing(self, embedder):
        """A keyword-miss text should join its nearest existing cluster.

        The embedding path's real value is catching *paraphrases* — texts that
        are semantically near-identical to a reference but use different words
        that the keyword map doesn't list. Both the reference and query here
        keyword-miss to 'general', but they're semantically similar (~0.55)."""
        # Reference: database topic, but no product name the keyword map knows
        db_ref = "I chose a relational data persistence system for our backend"
        fe_ref = "I build visual interfaces with a component framework"
        existing = [("database", embedder.embed(db_ref)),
                    ("frontend", embedder.embed(fe_ref))]

        # Query: same database intent, different phrasing, also keyword-miss
        result = _infer_cluster_embedding(
            "I picked a SQL-based storage engine for transactional workloads",
            embedder, existing=existing,
        )
        assert result == "database"

    def test_below_threshold_stays_general(self, embedder):
        """A text unrelated to all existing clusters stays 'general'."""
        db_vec = embedder.embed("PostgreSQL MySQL relational database")
        existing = [("database", db_vec)]

        result = _infer_cluster_embedding(
            "I love hiking in the mountains on weekends",
            embedder, existing=existing,
        )
        assert result == "general"


# ══════════════════════════════════════════════════════════════════════════════
# recluster CLI command
# ══════════════════════════════════════════════════════════════════════════════

class TestReclusterCommand:

    @pytest.fixture
    def setup_db(self, tmp_path):
        """Set up a tmp DB + config file with several 'general' nodes.
        Returns (config_path, db_path) so the CLI and assertions share state."""
        import yaml, time
        from pathlib import Path
        from noesis.memory.main import Memory

        db_path = str(tmp_path / "hot.db")
        vault = str(tmp_path / "vault")
        cfg = {"vector_store": {"config": {"db_path": db_path}},
               "embedder": {"config": {"model": "all-MiniLM-L6-v2"}},
               "cold_store": {"config": {"vault_path": vault}}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        m = Memory.from_config(cfg)
        m.embedding.embed("warmup")
        for text in [
            "I prefer PostgreSQL over MySQL",
            "I use Python for backend development",
            "I deploy with Docker and Kubernetes",
            "I think life is short and art is long",   # stays general
        ]:
            vec = m.embedding.embed(text)
            h = text[:12].replace(" ", "")
            m.vector_store.insert(h, vec, {
                "text": text, "user_id": "u1", "type": "position",
                "status": "provisional", "confidence": 0.5,
                "topic_cluster": "general", "created_at": time.time(),
            })
        # Force close so the CLI can reopen the DB
        del m
        return str(cfg_path), db_path

    def _cli(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli_main", "cli/main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_dry_run_changes_nothing(self, setup_db):
        """--dry-run must not write to the hot store."""
        from click.testing import CliRunner
        import sqlite3
        cfg_path, db_path = setup_db
        mod = self._cli()

        before = dict(sqlite3.connect(db_path).execute(
            "SELECT hash_id, topic_cluster FROM items").fetchall())
        result = CliRunner().invoke(mod.cli, ["recluster", "--dry-run", "--config", cfg_path])
        after = dict(sqlite3.connect(db_path).execute(
            "SELECT hash_id, topic_cluster FROM items").fetchall())

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert before == after  # nothing written

    def test_recluster_moves_general_out(self, setup_db):
        """Real recluster should move keyword-matchable nodes out of 'general'."""
        from click.testing import CliRunner
        import sqlite3
        from collections import Counter
        cfg_path, db_path = setup_db
        mod = self._cli()

        result = CliRunner().invoke(mod.cli, ["recluster", "--config", cfg_path])
        assert result.exit_code == 0

        clusters = Counter(r[0] for r in sqlite3.connect(db_path).execute(
            "SELECT topic_cluster FROM items").fetchall())
        # 3 of 4 should have moved out of general
        assert clusters["general"] <= 1
        assert clusters.get("database", 0) == 1
        assert clusters.get("languages", 0) == 1

    def test_recluster_idempotent(self, setup_db):
        """Running recluster twice should produce no changes the second time."""
        from click.testing import CliRunner
        cfg_path, db_path = setup_db
        mod = self._cli()
        runner = CliRunner()
        runner.invoke(mod.cli, ["recluster", "--config", cfg_path])
        result2 = runner.invoke(mod.cli, ["recluster", "--config", cfg_path])
        assert "Changed: 0" in result2.output
