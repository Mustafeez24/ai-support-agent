"""Regression tests for src/retrieval/embeddings.py's real
SentenceTransformerEmbedder wrapper (not the FakeEmbedder used elsewhere).

These tests mock `sentence_transformers.SentenceTransformer` so no model
is downloaded and no torch inference actually runs during normal `pytest`
-- they verify the WRAPPER's own logic (lazy loading, correct arguments
passed to `.encode()`, shape/dtype of what it returns, dimension caching)
is correct, independent of whether the real model/DLLs load correctly on
any given machine. This is deliberately NOT a fake/alternative embedding
implementation -- the production code in embeddings.py is untouched and
still requires a real sentence-transformers model at runtime; only the
*test* substitutes a mock for the underlying `SentenceTransformer` class.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.embeddings import SentenceTransformerEmbedder, TfidfEmbedder


def _make_mock_st_class(dimension: int = 384):
    """Returns a mock class whose instances behave like a real
    SentenceTransformer just enough for SentenceTransformerEmbedder's
    logic to exercise correctly."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dimension

    def fake_encode(texts, **kwargs):
        return np.random.RandomState(0).rand(len(texts), dimension).astype(np.float32)

    mock_model.encode.side_effect = fake_encode
    mock_cls = MagicMock(return_value=mock_model)
    return mock_cls, mock_model


def test_embedder_does_not_load_model_at_construction_time():
    """Importing/instantiating must never trigger a model download --
    this is what lets the rest of the test suite (and any script that
    merely constructs a Retriever without calling .search()/.build())
    run without network access."""
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        mock_cls.assert_not_called()
        assert embedder._model is None


def test_embed_lazily_loads_model_exactly_once():
    mock_cls, mock_model = _make_mock_st_class()
    with patch("sentence_transformers.SentenceTransformer", mock_cls):
        embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", batch_size=32)
        embedder.embed(["hello world"])
        embedder.embed(["a second call"])
        mock_cls.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")


def test_embed_returns_correct_shape_and_dtype():
    mock_cls, _ = _make_mock_st_class(dimension=384)
    with patch("sentence_transformers.SentenceTransformer", mock_cls):
        embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        vectors = embedder.embed(["first message", "second message", "third message"])
        assert vectors.shape == (3, 384)
        assert vectors.dtype == np.float32


def test_embed_passes_batch_size_and_normalization_to_encode():
    mock_cls, mock_model = _make_mock_st_class()
    with patch("sentence_transformers.SentenceTransformer", mock_cls):
        embedder = SentenceTransformerEmbedder("some-model", batch_size=64)
        embedder.embed(["x", "y"])
        _, kwargs = mock_model.encode.call_args
        assert kwargs["batch_size"] == 64
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["convert_to_numpy"] is True


def test_dimension_property_triggers_load_and_caches():
    mock_cls, _ = _make_mock_st_class(dimension=768)
    with patch("sentence_transformers.SentenceTransformer", mock_cls):
        embedder = SentenceTransformerEmbedder("some-model")
        assert embedder.dimension == 768
        assert embedder.dimension == 768  # second access must not reload
        mock_cls.assert_called_once()


def test_embed_on_empty_list_does_not_call_encode_with_missing_args():
    mock_cls, mock_model = _make_mock_st_class()
    with patch("sentence_transformers.SentenceTransformer", mock_cls):
        embedder = SentenceTransformerEmbedder("some-model")
        vectors = embedder.embed([])
        assert vectors.shape[0] == 0


# --- TfidfEmbedder (already real/non-mocked -- no heavy dependency to mock) ----


def test_tfidf_embedder_matches_expected_shape():
    embedder = TfidfEmbedder(max_features=50).fit(["order is late", "item is broken", "refund please"])
    vectors = embedder.embed(["order is late"])
    assert vectors.shape[0] == 1
    assert vectors.shape[1] == embedder.dimension
    assert vectors.dtype == np.float32


def test_tfidf_embedder_raises_before_fit():
    embedder = TfidfEmbedder()
    with pytest.raises(RuntimeError):
        embedder.embed(["hello"])
