"""Tests for the data pipeline (load, clean, threads, pipeline).

These run entirely against `tests/fixtures/sample_twcs.csv`, a small
hand-written CSV with the exact twcs schema, multiple brands, a multi-turn
thread, a duplicate tweet_id, and a missing-text row. No network access and
no dependency on the real (gitignored) dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import clean, load, pipeline, threads

FIXTURE = Path(__file__).parent / "fixtures" / "sample_twcs.csv"


# --- load.py ---------------------------------------------------------------


def test_inspect_schema_matches_expected_columns():
    schema = load.inspect_schema(FIXTURE)
    assert schema["columns"] == load.EXPECTED_COLUMNS
    assert schema["missing_expected_columns"] == []
    assert schema["unexpected_extra_columns"] == []


def test_inspect_schema_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load.inspect_schema(Path("does/not/exist.csv"))


def test_count_rows_matches_raw_line_count():
    # 14 data rows written in the fixture (see file).
    assert load.count_rows(FIXTURE) == 14


def test_build_full_index_excludes_text_column():
    index_df = load.build_full_index(FIXTURE, chunksize=5)
    assert "text" not in index_df.columns
    assert len(index_df) == 14
    assert index_df["inbound"].dtype == bool


def test_build_full_index_respects_max_rows():
    index_df = load.build_full_index(FIXTURE, chunksize=5, max_rows=3)
    assert len(index_df) == 3


def test_iter_full_chunks_filters_by_tweet_id():
    chunks = list(
        load.iter_full_chunks(FIXTURE, chunksize=4, tweet_ids={"1001", "1002"})
    )
    all_rows = pd.concat(chunks, ignore_index=True)
    # 1001 appears twice (duplicate row in fixture) + 1002 once.
    assert set(all_rows["tweet_id"]) == {"1001", "1002"}
    assert len(all_rows) == 3


# --- clean.py ----------------------------------------------------------------


def test_clean_text_replaces_urls_and_collapses_whitespace():
    raw = "Check   this out:\nhttps://amzn.to/track456   thanks"
    cleaned = clean.clean_text(raw)
    assert "https://" not in cleaned
    assert "[link]" in cleaned
    assert "  " not in cleaned


def test_clean_text_handles_none_and_nan():
    assert clean.clean_text(None) == ""
    assert clean.clean_text(float("nan")) == ""


def test_leading_mention_extracts_first_handle():
    assert clean.leading_mention("@AmazonHelp thanks for the help") == "AmazonHelp"
    assert clean.leading_mention("no mention here") is None


def test_duplicate_report_detects_the_seeded_duplicate():
    raw = pd.read_csv(FIXTURE, dtype=str)
    report = clean.duplicate_report(raw)
    assert report["duplicate_tweet_id_rows"] == 1  # tweet_id 1001 appears twice
    assert report["total_rows"] == 14


def test_drop_duplicate_tweet_ids_keeps_first_occurrence():
    raw = pd.read_csv(FIXTURE, dtype=str)
    deduped = clean.drop_duplicate_tweet_ids(raw)
    assert len(deduped) == 13
    kept = deduped[deduped["tweet_id"] == "1001"]["text"].iloc[0]
    assert "duplicate" not in kept


def test_missing_value_report_flags_empty_text_row():
    raw = pd.read_csv(FIXTURE, dtype=str)
    report = clean.missing_value_report(raw).set_index("column")
    assert report.loc["text", "missing_count"] >= 1


# --- threads.py --------------------------------------------------------------


def test_assign_conversation_ids_groups_the_multiturn_thread():
    index_df = load.build_full_index(FIXTURE, chunksize=5)
    index_df = clean.drop_duplicate_tweet_ids(index_df)
    index_df["conversation_id"] = threads.assign_conversation_ids(index_df)

    conv_of = index_df.set_index("tweet_id")["conversation_id"]
    # The 4-message AmazonHelp thread (1001-1004) must share one conversation_id.
    assert conv_of["1001"] == conv_of["1002"] == conv_of["1003"] == conv_of["1004"]
    # An unrelated conversation must NOT share that id.
    assert conv_of["2001"] != conv_of["1001"]
    # A lone, unlinked tweet is its own singleton conversation.
    assert conv_of["6001"] not in {conv_of["1001"], conv_of["2001"], conv_of["4001"]}


def test_brand_conversation_ids_finds_only_amazonhelp_threads():
    index_df = load.build_full_index(FIXTURE, chunksize=5)
    index_df = clean.drop_duplicate_tweet_ids(index_df)
    index_df["conversation_id"] = threads.assign_conversation_ids(index_df)

    amazon_convs = threads.brand_conversation_ids(index_df, "AmazonHelp")
    apple_convs = threads.brand_conversation_ids(index_df, "AppleSupport")
    assert len(amazon_convs) == 3  # threads starting 1001, 2001, 3001
    assert len(apple_convs) == 1  # thread starting 4001
    assert amazon_convs.isdisjoint(apple_convs)


def test_order_conversation_sorts_chronologically():
    index_df = load.build_full_index(FIXTURE, chunksize=5)
    index_df = clean.drop_duplicate_tweet_ids(index_df)
    thread = index_df[index_df["tweet_id"].isin(["1004", "1001", "1003", "1002"])]
    ordered = threads.order_conversation(thread)
    assert ordered["tweet_id"].tolist() == ["1001", "1002", "1003", "1004"]


def test_extract_resolution_pairs_pairs_customer_with_direct_brand_reply():
    index_df = load.build_full_index(FIXTURE, chunksize=5)
    index_df = clean.drop_duplicate_tweet_ids(index_df)
    index_df["conversation_id"] = threads.assign_conversation_ids(index_df)
    tweet_ids = set(index_df["tweet_id"])
    full_df = pd.concat(
        list(load.iter_full_chunks(FIXTURE, chunksize=5, tweet_ids=tweet_ids)),
        ignore_index=True,
    )
    full_df = clean.drop_duplicate_tweet_ids(full_df)
    full_df = full_df.merge(index_df[["tweet_id", "conversation_id"]], on="tweet_id")

    pairs = threads.extract_resolution_pairs(full_df, "AmazonHelp")
    # 3 direct AmazonHelp reply pairs: 1001->1002, 1003->1004, 2001->2002, 3001->3002
    assert len(pairs) == 4
    assert set(pairs["customer_tweet_id"]) == {"1001", "1003", "2001", "3001"}


# --- pipeline.py ---------------------------------------------------------------


def test_assign_thread_split_is_deterministic_and_thread_level():
    conv_ids = [f"c{i}" for i in range(20)]
    split_a = pipeline.assign_thread_split(conv_ids, 0.8, 0.2, seed=42)
    split_b = pipeline.assign_thread_split(conv_ids, 0.8, 0.2, seed=42)
    assert split_a == split_b  # deterministic given the same seed
    assert set(split_a.values()) == {"train_retrieval", "golden_eval"}
    n_train = sum(1 for v in split_a.values() if v == "train_retrieval")
    assert n_train == 16  # 80% of 20


def test_run_preprocessing_end_to_end_on_fixture(tmp_path):
    summary = pipeline.run_preprocessing(
        csv_path=FIXTURE,
        brand_author_id="AmazonHelp",
        processed_dir=tmp_path,
        chunksize=5,
        train_frac=0.8,
        golden_frac=0.2,
        seed=42,
    )
    assert summary["brand"] == "AmazonHelp"
    assert summary["total_conversations"] == 3
    assert summary["total_resolution_pairs"] == 4
    assert (tmp_path / "amazonhelp_full.parquet").exists()
    assert (tmp_path / "amazonhelp_resolution_pairs.parquet").exists()

    pairs = pd.read_parquet(tmp_path / "amazonhelp_resolution_pairs.parquet")
    assert set(pairs["split"].unique()) <= {"train_retrieval", "golden_eval"}
    # No conversation should straddle both splits (thread-level split, no leakage).
    conv_split_counts = pairs.groupby("conversation_id")["split"].nunique()
    assert (conv_split_counts == 1).all()
