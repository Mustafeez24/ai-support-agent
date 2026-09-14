"""Local, free text embeddings via sentence-transformers.

No paid API involved. The model is downloaded once (from Hugging Face,
cached locally by the library) and everything after that runs on CPU.
Kept as a small, swappable wrapper (rather than calling
`SentenceTransformer` directly everywhere) so retrieval code can be tested
against a fake embedder without downloading any model.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Loaded lazily so importing this
    module never triggers a model download."""

    def __init__(self, model_name: str, batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._dimension: int | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # so inner-product search == cosine similarity
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class TfidfEmbedder:
    """TF-IDF "embedding" used only by Baseline 2 (src/intents/baselines.py)
    -- a real, classical, non-LLM/non-neural retrieval representation, kept
    separate from `SentenceTransformerEmbedder` so the production system's
    dense retrieval and the simple baseline's TF-IDF retrieval are
    unambiguously different implementations, not the same code compared to
    itself.
    """

    def __init__(self, max_features: int = 5000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=(1, 2), min_df=1
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(corpus)
        self._fitted = True
        return self

    @property
    def dimension(self) -> int:
        return len(self._vectorizer.vocabulary_)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit() before embed().")
        from sklearn.preprocessing import normalize

        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return normalize(matrix, axis=1)
