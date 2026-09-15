"""Memory-efficient loading of the Customer Support on Twitter CSV.

The raw file is ~517MB / ~2.8M rows. We never call `pd.read_csv` without a
`chunksize` on it. Two access patterns are provided:

- `read_index_chunks` / `build_full_index`: streams only the *structural*
  columns (ids, author, inbound flag, reply pointers, timestamp) needed to
  reconstruct conversation threads and compute dataset-wide statistics.
  Dropping `text` here keeps the in-memory index small even for the full
  2.8M rows.
- `iter_full_chunks`: streams full rows (including `text`), optionally
  restricted to a set of `tweet_id`s already known to matter (e.g. only the
  rows belonging to AmazonHelp threads), so we only pay the memory/parsing
  cost for `text` on the subset we actually need downstream.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# Exact schema of the Customer Support on Twitter dataset (twcs.csv).
EXPECTED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

INDEX_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# dtypes chosen to minimize memory. IDs are read as nullable strings rather
# than int64 because a handful of malformed rows in this dataset can contain
# non-numeric junk; we coerce to numeric downstream where it matters.
DTYPE_MAP = {
    "tweet_id": "string",
    "author_id": "string",
    "inbound": "string",
    "created_at": "string",
    "text": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "string",
}


def assert_file_exists(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place the Customer Support on "
            "Twitter CSV at this path (see README for download + placement "
            "instructions). The raw file is intentionally gitignored."
        )


def inspect_schema(csv_path: Path, sample_rows: int = 5) -> dict:
    """Read only the header + a few sample rows to report the schema.

    Does not load the full file into memory.
    """
    assert_file_exists(csv_path)
    sample = pd.read_csv(
        csv_path,
        nrows=sample_rows,
        dtype=DTYPE_MAP,
        keep_default_na=True,
    )
    columns = list(sample.columns)
    missing_expected = [c for c in EXPECTED_COLUMNS if c not in columns]
    extra_columns = [c for c in columns if c not in EXPECTED_COLUMNS]
    return {
        "columns": columns,
        "dtypes": {c: str(sample[c].dtype) for c in columns},
        "missing_expected_columns": missing_expected,
        "unexpected_extra_columns": extra_columns,
        "sample_rows": sample.head(sample_rows).to_dict(orient="records"),
    }


def read_index_chunks(
    csv_path: Path, chunksize: int = 50_000
) -> Iterator[pd.DataFrame]:
    """Stream the structural (non-text) columns of the full CSV in chunks."""
    assert_file_exists(csv_path)
    reader = pd.read_csv(
        csv_path,
        usecols=lambda c: c in INDEX_COLUMNS,
        dtype={k: v for k, v in DTYPE_MAP.items() if k in INDEX_COLUMNS},
        chunksize=chunksize,
        keep_default_na=True,
    )
    for chunk in reader:
        chunk["inbound"] = chunk["inbound"].map({"True": True, "False": False})
        yield chunk


def build_full_index(
    csv_path: Path, chunksize: int = 50_000, max_rows: int | None = None
) -> pd.DataFrame:
    """Materialize the full structural index (all rows, no `text`) in memory.

    For the real ~2.8M-row file this index is on the order of a few hundred
    MB, which is fine to hold in memory once `text` (the bulk of the file's
    bytes) is excluded. `max_rows` allows capping this for fast local dev via
    `DATA_SAMPLE_SIZE`.
    """
    parts: list[pd.DataFrame] = []
    rows_read = 0
    for chunk in read_index_chunks(csv_path, chunksize=chunksize):
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        parts.append(chunk)
        rows_read += len(chunk)
        if max_rows is not None and rows_read >= max_rows:
            break
    if not parts:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    index_df = pd.concat(parts, ignore_index=True)
    logger.info("Built structural index with %d rows", len(index_df))
    return index_df


def iter_full_chunks(
    csv_path: Path,
    chunksize: int = 50_000,
    tweet_ids: Iterable[str] | None = None,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream full rows (including `text`), optionally filtered to a set of
    `tweet_id` values. Filtering happens per-chunk so memory stays bounded by
    `chunksize`, not by the full file size.
    """
    assert_file_exists(csv_path)
    keep_ids = set(tweet_ids) if tweet_ids is not None else None
    reader = pd.read_csv(
        csv_path,
        dtype=DTYPE_MAP,
        chunksize=chunksize,
        keep_default_na=True,
    )
    rows_yielded = 0
    for chunk in reader:
        chunk["inbound"] = chunk["inbound"].map({"True": True, "False": False})
        if keep_ids is not None:
            chunk = chunk[chunk["tweet_id"].isin(keep_ids)]
        if len(chunk) == 0:
            continue
        if max_rows is not None:
            remaining = max_rows - rows_yielded
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        yield chunk
        rows_yielded += len(chunk)
        if max_rows is not None and rows_yielded >= max_rows:
            break


def count_rows(csv_path: Path, chunksize: int = 100_000) -> int:
    """Exact row count via a streaming pass (no full-file load)."""
    assert_file_exists(csv_path)
    total = 0
    for chunk in pd.read_csv(csv_path, usecols=[0], chunksize=chunksize):
        total += len(chunk)
    return total
