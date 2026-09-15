#!/usr/bin/env python3
"""Train the production intent classifier on weak (taxonomy-seed-derived)
labels over the train_retrieval split -- see src/intents/weak_labels.py for
why this is weak supervision, not ground truth, and why the golden set is
never used for training.

Saves to outputs/cache/intent_classifier.joblib (gitignored -- regenerate
locally; it's a byproduct of the CSV + taxonomy, not a source artifact).

Windows:
    python scripts\\train_classifier.py

macOS/Linux:
    python scripts/train_classifier.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config as cfg  # noqa: E402
from src.intents.classifier import IntentClassifier, load_taxonomy  # noqa: E402
from src.intents.weak_labels import seed_centroid_labels  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_classifier")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs-path",
        type=Path,
        default=cfg.PROCESSED_DIR / f"{cfg.CONFIG.brand.author_id.lower()}_resolution_pairs.parquet",
    )
    p.add_argument("--taxonomy-path", type=Path, default=cfg.CONFIG_DIR / "intents.yaml")
    p.add_argument("--out-path", type=Path, default=cfg.CACHE_DIR / "intent_classifier.joblib")
    p.add_argument("--seed", type=int, default=cfg.CONFIG.data.random_seed)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pairs_path.exists():
        raise FileNotFoundError(f"{args.pairs_path} not found. Run: python scripts/preprocess.py")

    pairs = pd.read_parquet(args.pairs_path)
    train = pairs[pairs["split"] == "train_retrieval"]
    train = train[train["customer_text"].fillna("").str.strip() != ""]
    logger.info("Weak-labeling %d train_retrieval customer messages", len(train))

    taxonomy = load_taxonomy(args.taxonomy_path)
    texts = train["customer_text"].tolist()
    labels = seed_centroid_labels(texts, taxonomy)

    label_counts = pd.Series(labels).value_counts()
    logger.info("Weak-label distribution:\n%s", label_counts.to_string())

    clf = IntentClassifier(random_seed=args.seed)
    clf.fit(texts, labels)
    clf.save(args.out_path)
    logger.info("Saved trained classifier to %s", args.out_path)
    logger.warning(
        "This classifier was trained on WEAK (taxonomy-seed-similarity) "
        "labels, not human-verified ground truth. Its real accuracy is only "
        "known after evaluating against data/golden/golden_set.csv "
        "(python scripts/evaluate.py) -- do not treat pre-evaluation "
        "confidence scores as calibrated."
    )


if __name__ == "__main__":
    main()
