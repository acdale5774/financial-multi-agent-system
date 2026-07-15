"""Tests for the pluggable embedding layer (pure/deterministic-fake tier).

These prove provider dispatch and the OpenAIEmbedder's request/response
handling with a faked client — they say nothing about embedding *quality*;
that is the retrieval benchmark's job (`python -m evals.retrieval`).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.config import settings
from ingestion.embeddings import OpenAIEmbedder, get_embedder


@pytest.fixture(autouse=True)
def _fresh_embedder_cache():
    get_embedder.cache_clear()
    yield
    get_embedder.cache_clear()


def test_unknown_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_provider", "faiss")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        get_embedder()


def test_openai_provider_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    embedder = get_embedder()
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.dimension == settings.embedding_dimension


class _FakeEmbeddingsAPI:
    """Records requests; returns index-tagged vectors in scrambled order to
    prove the embedder re-sorts by index."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.requests: list[dict] = []

    def create(self, *, model: str, input: list[str], dimensions: int):  # noqa: A002
        self.requests.append({"model": model, "input": input, "dimensions": dimensions})
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * dimensions)
            for i in range(len(input))
        ]
        return SimpleNamespace(data=list(reversed(data)))


def _fake_client_embedder(dimension: int) -> tuple[OpenAIEmbedder, _FakeEmbeddingsAPI]:
    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)  # skip real-client __init__
    fake = _FakeEmbeddingsAPI(dimension)
    embedder._client = SimpleNamespace(embeddings=fake)
    embedder._model = "text-embedding-3-small"
    embedder._dimension = dimension
    return embedder, fake


def test_openai_embedder_sorts_by_index_and_batches() -> None:
    embedder, fake = _fake_client_embedder(dimension=4)
    texts = [f"text {i}" for i in range(OpenAIEmbedder._MAX_BATCH + 3)]

    vectors = embedder.embed(texts)

    assert vectors.shape == (len(texts), 4)
    assert vectors.dtype == np.float32
    # Two API calls: a full batch plus the 3-text remainder.
    assert [len(r["input"]) for r in fake.requests] == [OpenAIEmbedder._MAX_BATCH, 3]
    assert all(r["dimensions"] == 4 for r in fake.requests)
    # Response order is scrambled by the fake; row i must still be vector i.
    assert vectors[0][0] == 0.0 and vectors[1][0] == 1.0


def test_openai_embedder_guards_empty_strings() -> None:
    embedder, fake = _fake_client_embedder(dimension=4)

    embedder.embed(["real text", "", "   "])

    sent = fake.requests[0]["input"]
    assert sent == ["real text", " ", " "], "blank inputs replaced, not dropped"
