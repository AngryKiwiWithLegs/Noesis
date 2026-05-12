"""
noesis/cold_stores/base.py
"""
from abc import ABC, abstractmethod


class ColdStoreBase(ABC):
    @abstractmethod
    def write(self, hash_id: str, metadata: dict): ...

    @abstractmethod
    def read(self, hash_id: str) -> str: ...

    @abstractmethod
    def scan_modified(self, since: float) -> list[str]: ...

    @abstractmethod
    def mark_superseded(self, old_hash: str, new_hash: str): ...
