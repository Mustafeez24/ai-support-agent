"""Tests for the golden-set sampling logic in scripts/build_golden_set.py.

Only the non-interactive parts (`stratified_sample`, `cmd_report`) are
tested here -- `cmd_label` requires a human at the keyboard by design (see
data/golden/METHODOLOGY.md) and is never exercised automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import build_golden_set as bgs  # noqa: E402


def _synthetic_pairs(n_train: int, n_golden: int) -> pd.DataFrame:
    rows = []
    topics = [
        "my order is late and hasn't shipped",
        "the item arrived broken and damaged",
        "I want to return this for a refund",
        "cannot log into my account at all",
        "charged twice for the same order",
    ]
    total = n_train + n_golden
    for i in range(total):
        rows.append(
            {
                "conversation_id": f"c{i}",
                "customer_tweet_id": f"t{i}",
                "customer_text": f"{topics[i % len(topics)]} (case {i})",
                "customer_created_at": "2017-01-01",
                "brand_tweet_id": f"b{i}",
                "brand_text": "We're sorry to hear that, we're looking into it.",
                "brand_created_at": "2017-01-01",
                "split": "train_retrieval" if i < n_train else "golden_eval",
            }
        )
    return pd.DataFrame(rows)


def test_stratified_sample_respects_target_size():
    pairs = _synthetic_pairs(n_train=50, n_golden=150)
    sample = bgs.stratified_sample(pairs, target_size=40, seed=42)
    assert len(sample) == 40
    assert (sample["split"] == "golden_eval").all()


def test_stratified_sample_covers_multiple_strata():
    pairs = _synthetic_pairs(n_train=50, n_golden=150)
    sample = bgs.stratified_sample(pairs, target_size=40, seed=42)
    # With 5 real topics in a 150-row pool, stratified sampling of 40 should
    # not collapse onto a single cluster.
    assert sample["sampling_stratum"].nunique() > 1


def test_stratified_sample_falls_back_to_all_when_pool_too_small():
    pairs = _synthetic_pairs(n_train=10, n_golden=5)
    sample = bgs.stratified_sample(pairs, target_size=40, seed=42)
    assert len(sample) == 5
    assert (sample["sampling_stratum"] == "all").all()


def test_cmd_sample_writes_expected_columns(tmp_path, monkeypatch):
    from src import config as cfg

    processed_dir = tmp_path / "processed"
    golden_dir = tmp_path / "golden"
    processed_dir.mkdir()
    golden_dir.mkdir()
    pairs = _synthetic_pairs(n_train=50, n_golden=150)
    pairs_path = processed_dir / f"{cfg.CONFIG.brand.author_id.lower()}_resolution_pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)

    monkeypatch.setattr(cfg, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(cfg, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(bgs.cfg, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(bgs.cfg, "GOLDEN_DIR", golden_dir)

    args = type("Args", (), {"target_size": 20, "seed": 42})()
    bgs.cmd_sample(args)

    out_path = golden_dir / "golden_set_candidates.csv"
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == 20
    assert list(df.columns) == bgs.CANDIDATES_COLUMNS
    assert (df["intent"].fillna("") == "").all()  # genuinely unlabeled


def test_cmd_report_computes_distribution_from_labeled_rows(tmp_path, monkeypatch):
    from src import config as cfg

    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    df = pd.DataFrame(
        {
            "id": ["g0", "g1", "g2"],
            "tweet_id": ["1", "2", "3"],
            "conversation_id": ["c1", "c2", "c3"],
            "text": ["a", "b", "c"],
            "brand_historical_reply": ["x", "y", "z"],
            "sampling_stratum": ["0", "0", "1"],
            "intent": ["delivery_delay", "delivery_delay", "damaged_or_defective_item"],
            "expected_escalation": ["False", "False", "True"],
            "notes": ["", "", ""],
            "labeled_at": ["2024-01-01", "2024-01-01", "2024-01-01"],
        }
    )
    df.to_csv(golden_dir / "golden_set.csv", index=False)

    monkeypatch.setattr(bgs.cfg, "GOLDEN_DIR", golden_dir)

    bgs.cmd_report(type("Args", (), {})())

    out_path = golden_dir / "label_distribution.json"
    assert out_path.exists()
    import json

    dist = json.loads(out_path.read_text())
    assert dist["total_labeled"] == 3
    assert dist["labeling_complete"] is True
    assert dist["intent_distribution"]["delivery_delay"] == 2
