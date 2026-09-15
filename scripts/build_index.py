#!/usr/bin/env python3
"""Build the FAISS retrieval index over AmazonHelp historical resolution
pairs (train_retrieval split only -- see DECISIONS.md #3 on leakage).

The index and its metadata are written under outputs/cache/faiss_index/,
which is gitignored: this is a large, fully reproducible generated
artifact, not something to commit. Re-run this script any time
data/processed/*.parquet changes.

Windows:
    python scripts\\build_index.py

macOS/Linux:
    python scripts/build_index.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config as cfg  # noqa: E402
from src.retrieval.embeddings import SentenceTransformerEmbedder  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs-path",
        type=Path,
        default=cfg.PROCESSED_DIR / f"{cfg.CONFIG.brand.author_id.lower()}_resolution_pairs.parquet",
    )
    p.add_argument("--index-path", type=Path, default=cfg.CONFIG.retrieval.index_path)
    p.add_argument("--metadata-path", type=Path, default=cfg.CONFIG.retrieval.metadata_path)
    p.add_argument("--model", type=str, default=cfg.CONFIG.retrieval.embedding_model)
    p.add_argument("--batch-size", type=int, default=cfg.CONFIG.retrieval.batch_size)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pairs_path.exists():
        raise FileNotFoundError(f"{args.pairs_path} not found. Run: python scripts/preprocess.py")

    pairs = pd.read_parquet(args.pairs_path)
    train_pairs = pairs[pairs["split"] == "train_retrieval"]
    logger.info(
        "Indexing %d/%d resolution pairs (train_retrieval split only)",
        len(train_pairs),
        len(pairs),
    )

    embedder = SentenceTransformerEmbedder(args.model, batch_size=args.batch_size)
    retriever = Retriever(embedder)
    retriever.build(train_pairs)
    retriever.save(args.index_path, args.metadata_path)
    logger.info("Index build complete: %s (%d entries)", args.index_path, len(train_pairs))


if __name__ == "__main__":
    main()
