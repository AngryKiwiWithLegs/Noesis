"""
noesis/embeddings/local.py

Local embedding via sentence-transformers.
No API calls, no network — ~25-35ms per sentence on M-series CPU
(dominant cost of add()/search(); the vector search itself is <1ms).
Model downloaded once (~80MB), cached in ~/.cache/torch/sentence_transformers.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_MODEL_DIMS    = {"all-MiniLM-L6-v2": 384, "all-mpnet-base-v2": 768}


class LocalEmbedding:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model     = None
        self._dim       = _MODEL_DIMS.get(model_name, 384)

    # ── Lazy load ────────────────────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers required. "
                    "Run: pip install sentence-transformers"
                ) from e
            logger.info(f"Loading embedding model '{self.model_name}' "
                        f"(first call only) …")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded.")
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns normalised float list."""
        if not text.strip():
            return [0.0] * self._dim
        # inference_mode disables autograd graph construction → faster + less
        # memory. This is the single cheap win on CPU; the dominant cost
        # remains the forward pass itself (~20-35ms/sentence on M-series).
        import torch
        with torch.inference_mode():
            return self.model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings efficiently."""
        if not texts:
            return []
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).tolist()

    @property
    def dim(self) -> int:
        return self._dim
