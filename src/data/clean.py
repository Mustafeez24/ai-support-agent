"""Text cleaning and de-duplication for the Customer Support on Twitter data.

Cleaning is intentionally conservative: this text becomes both retrieval
evidence and the basis for grounded replies, so we normalize noise (URLs,
whitespace, repeated mentions) without destroying meaning.
"""
from __future__ import annotations

import re

import pandas as pd

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None, *, strip_mentions: bool = False) -> str:
    """Normalize a single tweet's text.

    - Replaces URLs with a `[link]` placeholder (URLs carry no retrievable
      meaning but their presence can be a weak signal, e.g. "track your
      order [link]").
    - Optionally strips @mentions (kept by default: the leading mention is
      how customer tweets address the brand, and agent replies opening with
      "@customer" is part of the brand's authentic voice — stripping it
      would make retrieved evidence look unlike what the LLM should produce).
    - Collapses whitespace/newlines to single spaces and trims.
    """
    if text is None or pd.isna(text):
        return ""
    cleaned = _URL_RE.sub("[link]", str(text))
    if strip_mentions:
        cleaned = _MENTION_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def leading_mention(text: str) -> str | None:
    """Return the first @mention in a tweet, if any (often the addressee)."""
    if not text:
        return None
    match = _MENTION_RE.search(text)
    return match.group(0)[1:] if match else None


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and percentage of missing/empty values."""
    n = len(df)
    rows = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        if series.dtype == object or str(series.dtype).startswith("string"):
            n_missing += int((series.fillna("").astype(str).str.strip() == "").sum() - int(series.isna().sum()))
        rows.append(
            {
                "column": col,
                "missing_count": n_missing,
                "missing_pct": round(100 * n_missing / n, 4) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def duplicate_report(df: pd.DataFrame) -> dict:
    """Report exact-duplicate tweet_ids and exact-duplicate text rows."""
    dup_ids = int(df["tweet_id"].duplicated().sum()) if "tweet_id" in df.columns else None
    dup_text = None
    if "text" in df.columns:
        non_empty = df["text"].fillna("").astype(str).str.strip()
        dup_text = int(non_empty[non_empty != ""].duplicated().sum())
    return {
        "duplicate_tweet_id_rows": dup_ids,
        "duplicate_text_rows": dup_text,
        "total_rows": len(df),
    }


def drop_duplicate_tweet_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each tweet_id.

    Duplicate tweet_ids should not occur in a well-formed export, but we
    guard against them explicitly since they would otherwise silently
    corrupt thread reconstruction (a tweet_id must be a unique node in the
    reply graph).
    """
    if "tweet_id" not in df.columns:
        return df
    return df.drop_duplicates(subset="tweet_id", keep="first")
