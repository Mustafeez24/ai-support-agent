#!/usr/bin/env python3
"""Run the full data preprocessing pass once and cache results under
data/processed/. Every later script (intent discovery, golden-set sampling,
index building, baselines, evaluation) reads the cached parquet files
instead of re-scanning the raw CSV.

Windows:
    python scripts\\preprocess.py
    python scripts\\preprocess.py --sample-size 200000   (fast dev run)

macOS/Linux:
    python scripts/preprocess.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src.data import pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocess")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-path", type=Path, default=cfg.CONFIG.data.csv_path)
    p.add_argument("--brand", type=str, default=cfg.CONFIG.brand.author_id)
    p.add_argument("--sample-size", type=int, default=cfg.CONFIG.data.sample_size)
    p.add_argument("--chunksize", type=int, default=cfg.CONFIG.data.chunksize)
    p.add_argument("--out-dir", type=Path, default=cfg.PROCESSED_DIR)
    p.add_argument("--train-frac", type=float, default=cfg.CONFIG.split.train_frac)
    p.add_argument("--golden-frac", type=float, default=cfg.CONFIG.split.golden_frac)
    p.add_argument("--seed", type=int, default=cfg.CONFIG.split.random_seed)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg.ensure_output_dirs()
    summary = pipeline.run_preprocessing(
        csv_path=args.csv_path,
        brand_author_id=args.brand,
        processed_dir=args.out_dir,
        chunksize=args.chunksize,
        max_rows=args.sample_size or None,
        train_frac=args.train_frac,
        golden_frac=args.golden_frac,
        seed=args.seed,
    )
    summary_path = cfg.ANALYSIS_DIR / "preprocessing_summary.json"
    cfg.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Done. Summary written to %s", summary_path)
    logger.info(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
