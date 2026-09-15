"""FAISS-backed nearest-neighbor retrieval over historical AmazonHelp
resolution pairs.

Given a new customer message, retrieves the most similar historical
customer messages and returns their paired brand responses as grounding
evidence for reply generation (Phase 7).

Leakage prevention: `Retriever.build` refuses to index anything outside
the `train_retrieval` split (see src/data/pipeline.py and DECISIONS.md
#3) -- the golden evaluation set's resolutions must never be retrievable
evidence for grading that same set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievedEvidence:
    customer_tweet_id: str
    customer_text: str
    brand_tweet_id: str
    brand_text: str
    conversation_id: str
    similarity: float

    def to_dict(self) -> dict:
        return {
            "customer_tweet_id": self.customer_tweet_id,
            "customer_text": self.customer_text,
            "brand_tweet_id": self.brand_tweet_id,
            "brand_text": self.brand_text,
            "conversation_id": self.conversation_id,
            "similarity": self.similarity,
        }


class LeakageError(ValueError):
    """Raised when an index build or evaluation would leak golden-set
    resolutions into the retrieval corpus."""


class Retriever:
    def __init__(self, embedder: Embedder, top_k: int = 5):
        self.embedder = embedder
        self.top_k = top_k
        self.index = None
        self.metadata: pd.DataFrame | None = None

    def build(self, pairs_df: pd.DataFrame, allow_any_split: bool = False) -> None:
        """Build the index from `pairs_df` (must have `customer_text`,
        `brand_text`, `customer_tweet_id`, `brand_tweet_id`,
        `conversation_id`, `split` columns).

        Refuses non-`train_retrieval` rows unless `allow_any_split=True`
        (only intended for tests / explicit ad-hoc exploration, never for
        the real index used at evaluation time).
        """
        if not allow_any_split:
            bad = pairs_df[pairs_df["split"] != "train_retrieval"]
            if len(bad) > 0:
                raise LeakageError(
                    f"{len(bad)} rows outside the train_retrieval split were "
                    "passed to Retriever.build(). Filter to split == "
                    "'train_retrieval' first, or pass allow_any_split=True "
                    "if this is intentional (e.g. a test)."
                )
        pairs_df = pairs_df[pairs_df["customer_text"].fillna("").str.strip() != ""].reset_index(drop=True)
        if len(pairs_df) == 0:
            raise ValueError("No rows with non-empty customer_text to index.")

        # Embed BEFORE importing faiss. On Windows, faiss-cpu bundles its own
        # MKL/OpenMP runtime; if it initializes first in the process, torch's
        # later DLL init (triggered by the embedder's lazy sentence-transformers
        # import) can fail with WinError 1114. Embedding first guarantees
        # torch/sentence-transformers -- whatever the embedder needs -- loads
        # and initializes before faiss ever enters the process. See
        # DECISIONS.md for the full diagnosis.
        vectors = self.embedder.embed(pairs_df["customer_text"].tolist())

        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index = index
        self.metadata = pairs_df
        logger.info("Built retrieval index with %d entries", len(pairs_df))

    def search(self, query_text: str, top_k: int | None = None) -> list[RetrievedEvidence]:
        if self.index is None or self.metadata is None:
            raise RuntimeError("Retriever.build() or .load() must be called before search().")
        k = top_k or self.top_k
        query_vec = self.embedder.embed([query_text])
        scores, idxs = self.index.search(query_vec, min(k, len(self.metadata)))
        results = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            row = self.metadata.iloc[idx]
            results.append(
                RetrievedEvidence(
                    customer_tweet_id=str(row["customer_tweet_id"]),
                    customer_text=str(row["customer_text"]),
                    brand_tweet_id=str(row["brand_tweet_id"]),
                    brand_text=str(row["brand_text"]),
                    conversation_id=str(row["conversation_id"]),
                    similarity=float(score),
                )
            )
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        if self.index is None or self.metadata is None:
            raise RuntimeError("Nothing to save -- call build() first.")
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        self.metadata.to_parquet(metadata_path, index=False)
        logger.info("Saved index (%d entries) to %s", len(self.metadata), index_path)

    @classmethod
    def load(cls, embedder: Embedder, index_path: Path, metadata_path: Path, top_k: int = 5) -> "Retriever":
        # Same DLL-init-order concern as build() (see comment there): warm
        # the embedder (triggers its lazy model load, e.g. torch) before
        # faiss enters the process for the first time. Best-effort --
        # an embedder that isn't ready to load yet (e.g. an unfit
        # TfidfEmbedder) just skips the warm-up rather than breaking load().
        try:
            _ = embedder.dimension
        except Exception:
            pass

        import faiss

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_path} / {metadata_path}. Build it first: "
                "python scripts/build_index.py"
            )
        obj = cls(embedder, top_k=top_k)
        obj.index = faiss.read_index(str(index_path))
        obj.metadata = pd.read_parquet(metadata_path)
        return obj


def assert_no_leakage(index_metadata: pd.DataFrame, golden_tweet_ids: set[str]) -> None:
    """Hard check used by evaluation (Phase 9): none of the golden set's
    customer_tweet_ids may appear as indexed evidence."""
    indexed_ids = set(index_metadata["customer_tweet_id"].astype(str))
    overlap = indexed_ids & {str(t) for t in golden_tweet_ids}
    if overlap:
        raise LeakageError(
            f"{len(overlap)} golden-set tweet_ids found inside the retrieval "
            f"index: {sorted(overlap)[:10]}..."
        )
