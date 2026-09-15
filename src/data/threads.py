"""Conversation/thread reconstruction from the reply-pointer columns.

The dataset encodes a reply graph via two columns:
- `in_response_to_tweet_id`: the single tweet_id this tweet replies to
  (empty for a thread's root message).
- `response_tweet_id`: a comma-separated list of tweet_ids that reply to
  this tweet (the inverse edge, often but not always redundant with the
  above).

We treat each connected component of this graph as one "conversation"
(`conversation_id`), using union-find over both edge directions for
robustness against rows where only one side of the pointer is populated.
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd

_TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        parent = self.parent
        if x not in parent:
            parent[x] = x
            self.rank[x] = 0
            return x
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        rank = self.rank
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1


def _split_response_ids(raw: object) -> list[str]:
    if raw is None or pd.isna(raw):
        return []
    raw = str(raw).strip()
    if not raw or raw.lower() == "nan":
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def assign_conversation_ids(index_df: pd.DataFrame) -> pd.Series:
    """Return a `conversation_id` Series aligned to `index_df`'s row order.

    `conversation_id` is the tweet_id of the union-find root of each row's
    connected component — deterministic given the same input rows.
    """
    uf = _UnionFind()
    tweet_ids = index_df["tweet_id"].tolist()
    in_response_to = index_df["in_response_to_tweet_id"].tolist()
    response_ids_raw = (
        index_df["response_tweet_id"].tolist()
        if "response_tweet_id" in index_df.columns
        else [None] * len(index_df)
    )

    for tid in tweet_ids:
        uf.find(tid)  # ensure every tweet_id is a registered node

    for tid, parent_id in zip(tweet_ids, in_response_to):
        if parent_id is not None and not pd.isna(parent_id):
            parent_id = str(parent_id).strip()
            if parent_id and parent_id.lower() != "nan":
                uf.union(tid, parent_id)

    for tid, raw in zip(tweet_ids, response_ids_raw):
        for child_id in _split_response_ids(raw):
            uf.union(tid, child_id)

    roots = [uf.find(tid) for tid in tweet_ids]
    return pd.Series(roots, index=index_df.index, name="conversation_id")


def brand_conversation_ids(
    index_with_conv: pd.DataFrame, brand_author_id: str
) -> set[str]:
    """Conversation ids that include at least one outbound message from the
    brand's support account."""
    is_brand_outbound = (index_with_conv["author_id"] == brand_author_id) & (
        index_with_conv["inbound"] == False  # noqa: E712
    )
    return set(index_with_conv.loc[is_brand_outbound, "conversation_id"].unique())


def parse_created_at(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format=_TWITTER_DATE_FORMAT, errors="coerce", utc=True)
    still_missing = parsed.isna() & series.notna()
    if still_missing.any():
        parsed.loc[still_missing] = pd.to_datetime(
            series[still_missing], errors="coerce", utc=True
        )
    return parsed


def order_conversation(conv_df: pd.DataFrame) -> pd.DataFrame:
    """Order a single conversation's rows chronologically.

    Falls back to the reply chain (via `in_response_to_tweet_id`) for rows
    with unparseable timestamps, by leaving ties in original (file) order,
    which for this dataset is already close to chronological.
    """
    df = conv_df.copy()
    df["_created_at_parsed"] = parse_created_at(df["created_at"])
    return df.sort_values("_created_at_parsed", kind="stable").drop(
        columns="_created_at_parsed"
    )


def extract_resolution_pairs(
    full_df: pd.DataFrame, brand_author_id: str
) -> pd.DataFrame:
    """Extract direct (customer_message -> brand_response) pairs.

    A pair is a brand outbound tweet whose `in_response_to_tweet_id` points
    directly at a customer inbound tweet. This is the unit of "historical
    resolution" evidence used by retrieval: the customer's issue text paired
    with exactly how the brand actually replied to it.

    `full_df` must contain the full columns (including `text`) for both the
    customer and brand rows, plus a `conversation_id` column.
    """
    required = {"tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"}
    missing = required - set(full_df.columns)
    if missing:
        raise ValueError(f"full_df missing required columns: {missing}")

    by_tweet_id = full_df.set_index("tweet_id", drop=False)

    brand_replies = full_df[
        (full_df["author_id"] == brand_author_id) & (full_df["inbound"] == False)  # noqa: E712
    ]

    records = []
    for row in brand_replies.itertuples(index=False):
        parent_id = getattr(row, "in_response_to_tweet_id")
        if parent_id is None or pd.isna(parent_id):
            continue
        parent_id = str(parent_id).strip()
        if not parent_id or parent_id not in by_tweet_id.index:
            continue
        parent = by_tweet_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):  # duplicate index guard
            parent = parent.iloc[0]
        if not bool(parent["inbound"]):
            continue  # only pair against genuine customer (inbound) messages
        records.append(
            {
                "conversation_id": getattr(row, "conversation_id", None),
                "customer_tweet_id": parent["tweet_id"],
                "customer_text": parent["text"],
                "customer_created_at": parent["created_at"],
                "brand_tweet_id": getattr(row, "tweet_id"),
                "brand_text": getattr(row, "text"),
                "brand_created_at": getattr(row, "created_at"),
            }
        )
    return pd.DataFrame.from_records(records)


def iter_conversations(
    full_df_with_conv: pd.DataFrame,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (conversation_id, ordered rows) for each conversation, sorted
    chronologically within the conversation."""
    for conv_id, group in full_df_with_conv.groupby("conversation_id", sort=False):
        yield conv_id, order_conversation(group)
