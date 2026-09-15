"""Tests for the FAISS retrieval layer.

Uses a small deterministic TF-IDF-based FakeEmbedder instead of the real
sentence-transformers model, so these tests never download anything and
run fast -- they exercise the real FAISS index build/search/save/load
path and the leakage guard, which is what actually matters here (the
embedding model itself is a swappable implementation detail behind the
`Embedder` protocol).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.retrieval.retriever import LeakageError, Retriever, assert_no_leakage


class FakeEmbedder:
    """TF-IDF embedder fit once over a fixed corpus -- deterministic,
    no network access, but still produces semantically meaningful
    similarity for the synthetic test corpora below."""

    def __init__(self, corpus: list[str]):
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._vectorizer.fit(corpus)
        self._dim = len(self._vectorizer.vocabulary_)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return normalize(matrix, axis=1)


DELIVERY = [
    "my order is late and still hasn't shipped",
    "package delivery is delayed again this week",
    "when will my late order finally arrive",
]
DAMAGED = [
    "the item arrived broken and cracked",
    "my product is damaged and defective on arrival",
    "received a shattered broken item today",
]


def _pairs_df(texts: list[str], split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "conversation_id": [f"c{i}" for i in range(len(texts))],
            "customer_tweet_id": [f"t{i}" for i in range(len(texts))],
            "customer_text": texts,
            "brand_tweet_id": [f"b{i}" for i in range(len(texts))],
            "brand_text": [f"We're sorry, reply {i}" for i in range(len(texts))],
            "split": split,
        }
    )


@pytest.fixture
def embedder():
    return FakeEmbedder(DELIVERY + DAMAGED)


def test_build_rejects_non_train_split_by_default(embedder):
    pairs = _pairs_df(DELIVERY, split="golden_eval")
    retriever = Retriever(embedder)
    with pytest.raises(LeakageError):
        retriever.build(pairs)


def test_build_allows_non_train_split_when_explicitly_forced(embedder):
    pairs = _pairs_df(DELIVERY, split="golden_eval")
    retriever = Retriever(embedder)
    retriever.build(pairs, allow_any_split=True)
    assert retriever.index.ntotal == len(DELIVERY)


def test_search_returns_semantically_closest_evidence(embedder):
    pairs = _pairs_df(DELIVERY + DAMAGED, split="train_retrieval")
    retriever = Retriever(embedder, top_k=2)
    retriever.build(pairs)

    results = retriever.search("my package never showed up, still waiting")
    assert len(results) == 2
    top_texts = {r.customer_text for r in results}
    assert top_texts.issubset(set(DELIVERY))
    assert results[0].similarity >= results[1].similarity


def test_search_before_build_raises(embedder):
    retriever = Retriever(embedder)
    with pytest.raises(RuntimeError):
        retriever.search("hello")


def test_save_and_load_round_trip(tmp_path, embedder):
    pairs = _pairs_df(DELIVERY + DAMAGED, split="train_retrieval")
    retriever = Retriever(embedder, top_k=3)
    retriever.build(pairs)

    index_path = tmp_path / "idx.faiss"
    meta_path = tmp_path / "meta.parquet"
    retriever.save(index_path, meta_path)

    loaded = Retriever.load(embedder, index_path, meta_path, top_k=3)
    original_results = retriever.search("broken item arrived")
    loaded_results = loaded.search("broken item arrived")
    assert [r.customer_tweet_id for r in original_results] == [r.customer_tweet_id for r in loaded_results]


def test_assert_no_leakage_passes_when_disjoint(embedder):
    pairs = _pairs_df(DELIVERY, split="train_retrieval")
    retriever = Retriever(embedder)
    retriever.build(pairs)
    assert_no_leakage(retriever.metadata, golden_tweet_ids={"some_other_id"})


def test_assert_no_leakage_raises_on_overlap(embedder):
    pairs = _pairs_df(DELIVERY, split="train_retrieval")
    retriever = Retriever(embedder)
    retriever.build(pairs)
    with pytest.raises(LeakageError):
        assert_no_leakage(retriever.metadata, golden_tweet_ids={"t0", "t1"})
