"""noesis/adapters/base.py"""
from abc import ABC, abstractmethod


class AbstractAdapter(ABC):
    @abstractmethod
    async def start(self): ...

    @abstractmethod
    async def stop(self): ...

    @property
    @abstractmethod
    def name(self) -> str: ...
