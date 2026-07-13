"""Pluggable embedding layer.

The rest of the pipeline depends only on the `Embedder` protocol, so the
provider can be swapped (e.g. to Voyage `voyage-finance-2` or OpenAI
`text-embedding-3-small`) by adding a new implementation and updating
`EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` — plus the `vector(N)` size in
db/schema.sql — then re-indexing.

Default: model2vec static embeddings (numpy-only). Chosen deliberately:
- No API key / network needed at query time; free; ~instant on CPU.
- The only local option that installs on this dev machine (Intel macOS +
  Python 3.14 — torch and onnxruntime have no wheels here).
Trade-off: static (non-contextual) embeddings retrieve somewhat worse than
transformer/API models; we compensate with structure-aware chunking and a
contextual header prepended to each chunk.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything that can turn a batch of texts into fixed-size vectors."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dimension), dtype float32."""
        ...


class Model2VecEmbedder:
    """Local static embeddings via model2vec (no torch/onnxruntime/API)."""

    def __init__(self, model_name: str | None = None) -> None:
        from model2vec import StaticModel  # lazy: keeps unit tests import-light

        from core.config import settings

        self._model = StaticModel.from_pretrained(model_name or settings.embedding_model)
        self._dimension = int(self._model.dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts)
        return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the process-wide embedder (model loaded once).

    Validates the model's output dimension against settings so a model swap
    that forgets to update `EMBEDDING_DIMENSION` + db/schema.sql fails loudly
    instead of storing truncated/incompatible vectors.
    """
    from core.config import settings

    embedder = Model2VecEmbedder()
    if embedder.dimension != settings.embedding_dimension:
        raise RuntimeError(
            f"Embedding model '{settings.embedding_model}' outputs "
            f"{embedder.dimension}-d vectors but EMBEDDING_DIMENSION is "
            f"{settings.embedding_dimension}. Update the setting and the "
            f"vector({settings.embedding_dimension}) column in db/schema.sql, "
            "then re-index."
        )
    return embedder
