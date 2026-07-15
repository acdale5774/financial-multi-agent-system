"""Pluggable embedding layer.

The rest of the pipeline depends only on the `Embedder` protocol; the
provider is selected by `EMBEDDING_PROVIDER`:

- `openai` (default) — `text-embedding-3-small` via the OpenAI API. Requires
  `OPENAI_API_KEY` (already required for the agent layer) and a network call
  at ingestion AND query time. Chosen after the retrieval benchmark showed
  dense recall was the weakest link under static local embeddings — see
  evals/retrieval_report_model2vec.{md,json} for the before numbers.
- `model2vec` — local static embeddings (numpy-only): no API key, free,
  offline, and the only option that installs on this dev machine (Intel
  macOS + Python 3.14 — torch and onnxruntime have no wheels here). Kept as
  the offline fallback; retrieves worse (non-contextual).

Switching providers means updating `EMBEDDING_MODEL` + `EMBEDDING_DIMENSION`,
the `vector(N)` size in db/schema.sql (re-apply it), and re-embedding the
corpus (`python -m ingestion.reembed`).
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


class OpenAIEmbedder:
    """API embeddings via OpenAI (`text-embedding-3-small` by default).

    `dimensions` is passed explicitly so `EMBEDDING_DIMENSION` is
    authoritative (text-embedding-3 models natively support Matryoshka
    truncation); the returned vectors are unit-normalized by the API.
    """

    # The API rejects batches over 2048 inputs; stay well under it and under
    # the per-request token ceiling (chunks are <= 3,000 chars).
    _MAX_BATCH = 512

    def __init__(self, model_name: str | None = None) -> None:
        from openai import OpenAI  # lazy: keeps unit tests import-light

        from core.config import settings

        self._client = OpenAI(api_key=settings.openai_api_key or None)
        self._model = model_name or settings.embedding_model
        self._dimension = settings.embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        # The API rejects empty strings; embed a single space instead.
        safe = [t if t.strip() else " " for t in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(safe), self._MAX_BATCH):
            response = self._client.embeddings.create(
                model=self._model,
                input=safe[start : start + self._MAX_BATCH],
                dimensions=self._dimension,
            )
            vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
        return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the process-wide embedder (model loaded once).

    Validates the model's output dimension against settings so a model swap
    that forgets to update `EMBEDDING_DIMENSION` + db/schema.sql fails loudly
    instead of storing truncated/incompatible vectors.
    """
    from core.config import settings

    provider = settings.embedding_provider
    embedder: Embedder
    if provider == "openai":
        embedder = OpenAIEmbedder()
    elif provider == "model2vec":
        embedder = Model2VecEmbedder()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER {provider!r}; expected 'openai' or 'model2vec'."
        )
    if embedder.dimension != settings.embedding_dimension:
        raise RuntimeError(
            f"Embedding model '{settings.embedding_model}' outputs "
            f"{embedder.dimension}-d vectors but EMBEDDING_DIMENSION is "
            f"{settings.embedding_dimension}. Update the setting and the "
            f"vector({settings.embedding_dimension}) column in db/schema.sql, "
            "then re-index."
        )
    return embedder
