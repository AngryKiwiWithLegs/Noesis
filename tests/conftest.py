"""
tests/conftest.py

Shared fixtures available to all test modules.
Import them by name in any test file — pytest discovers them automatically.
"""
import pytest
from noesis.memory.main import Memory


@pytest.fixture(scope="function")
def mem(tmp_path):
    """Fresh Memory instance with hot + cold store, no LLM pipeline."""
    return Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        "cold_store":   {"config": {"vault_path": str(tmp_path / "vault")}},
    })


@pytest.fixture(scope="function")
def mem_hot_only(tmp_path):
    """Memory with hot store only — fastest, no vault I/O."""
    return Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
    })
