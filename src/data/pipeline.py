"""Orchestrates load -> clean -> thread-reconstruction -> split into cached,
reusable processed files under `data/processed/`.

This module exists so the (potentially slow, full-file) preprocessing pass
happens exactly once. `scripts/preprocess.py` calls this and writes parquet
files that every downstream script (golden-set creation, index building,
baselines, evaluation) reads instead of re-scanning the 517MB CSV.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import clean, load, threads

logger = logging.getLogger(__name__)


def build_brand_conversation_index(
    csv_path: Path,
    brand_author_id: str,
    chunksize: int = 50_000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Single streaming pass: build the structural index, assign
    conversation_ids, and return only the rows belonging to conversations
    that include the brand.
    """
    index_df = load.build_full_index(csv_path, chunksize=chunksize, max_rows=max_rows)
    index_df = clean.drop_duplicate_tweet_ids(index_df)
    index_df["conversation_id"] = threads.assign_conversation_ids(index_df)
    brand_conv_ids = threads.brand_conversation_ids(index_df, brand_author_id)
    brand_index = index_df[index_df["conversation_id"].isin(brand_conv_ids)].copy()
    logger.info(
        "Brand index: %d rows across %d conversations (from %d total rows)",
        len(brand_index),
        len(brand_conv_ids),
        len(index_df),
    )
    return brand_index


def materialize_full_rows(
    csv_path: Path,
    tweet_ids: set[str],
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Second streaming pass: pull full rows (with `text`) for a known set
    of tweet_ids, keeping memory bounded to `chunksize` per pass."""
    parts = [
        chunk
        for chunk in load.iter_full_chunks(csv_path, chunksize=chunksize, tweet_ids=tweet_ids)
    ]
    if not parts:
        return pd.DataFrame(columns=load.EXPECTED_COLUMNS)
    full_df = pd.concat(parts, ignore_index=True)
    full_df = clean.drop_duplicate_tweet_ids(full_df)
    full_df["text"] = full_df["text"].map(clean.clean_text)
    return full_df


def assign_thread_split(
    conversation_ids: list[str],
    train_frac: float,
    golden_frac: float,
    seed: int,
) -> dict[str, str]:
    """Deterministic, conversation-level split so no single conversation's
    messages appear in both the retrieval/training corpus and the golden
    evaluation set (thread-level split avoids leakage).
    """
    unique_ids = sorted(set(conversation_ids))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_ids)
    n = len(shuffled)
    n_train = int(round(n * train_frac))
    train_ids = set(shuffled[:n_train])
    split_map = {
        cid: ("train_retrieval" if cid in train_ids else "golden_eval")
        for cid in shuffled
    }
    logger.info(
        "Thread split: %d train_retrieval / %d golden_eval (train_frac=%.2f)",
        sum(1 for v in split_map.values() if v == "train_retrieval"),
        sum(1 for v in split_map.values() if v == "golden_eval"),
        train_frac,
    )
    return split_map


def run_preprocessing(
    csv_path: Path,
    brand_author_id: str,
    processed_dir: Path,
    chunksize: int = 50_000,
    max_rows: int | None = None,
    train_frac: float = 0.80,
    golden_frac: float = 0.20,
    seed: int = 42,
) -> dict:
    """Full preprocessing pass. Writes:

    - `{brand}_full.parquet`: cleaned full rows for the brand's conversations
    - `{brand}_resolution_pairs.parquet`: (customer_message -> brand_response)
      pairs with a `split` column (`train_retrieval` / `golden_eval`)

    Returns a summary dict for logging/analysis.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    brand_index = build_brand_conversation_index(
        csv_path, brand_author_id, chunksize=chunksize, max_rows=max_rows
    )
    tweet_ids = set(brand_index["tweet_id"].tolist())
    full_df = materialize_full_rows(csv_path, tweet_ids, chunksize=chunksize)
    full_df = full_df.merge(
        brand_index[["tweet_id", "conversation_id"]], on="tweet_id", how="left"
    )

    pairs = threads.extract_resolution_pairs(full_df, brand_author_id)
    split_map = assign_thread_split(
        full_df["conversation_id"].dropna().unique().tolist(),
        train_frac=train_frac,
        golden_frac=golden_frac,
        seed=seed,
    )
    pairs["split"] = pairs["conversation_id"].map(split_map)
    full_df["split"] = full_df["conversation_id"].map(split_map)

    full_path = processed_dir / f"{brand_author_id.lower()}_full.parquet"
    pairs_path = processed_dir / f"{brand_author_id.lower()}_resolution_pairs.parquet"
    full_df.to_parquet(full_path, index=False)
    pairs.to_parquet(pairs_path, index=False)

    summary = {
        "brand": brand_author_id,
        "total_brand_rows": len(full_df),
        "total_conversations": full_df["conversation_id"].nunique(),
        "total_resolution_pairs": len(pairs),
        "split_counts": pairs["split"].value_counts().to_dict(),
        "full_path": str(full_path),
        "pairs_path": str(pairs_path),
    }
    logger.info("Preprocessing summary: %s", summary)
    return summary
