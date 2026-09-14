#!/usr/bin/env python3
"""Golden evaluation set: sampling + a real human-labeling CLI workflow.

This script does NOT auto-label anything. It has three stages:

  sample  - deterministically samples ~target_size candidate messages from
            the golden_eval (held-out) split, stratified by discovered
            cluster (see src/intents/discover.py) so rare issue types
            aren't sampled away by chance. Writes an UNLABELED template.

  label   - an interactive terminal CLI. Shows one candidate at a time
            (customer text, the brand's actual historical reply for
            context, and the taxonomy) and asks a human to enter the real
            intent, whether it should have been auto-handled or escalated,
            and optional notes. Writes incrementally so it's safe to
            Ctrl-C and resume later. THIS STAGE REQUIRES AN ACTUAL HUMAN
            AT THE KEYBOARD -- it is never run non-interactively, and this
            codebase never fabricates its output.

  report  - once labeling is done (or partially done), computes and writes
            the label-distribution statistics required by the assignment
            (data/golden/label_distribution.json).

Windows:
    python scripts\\build_golden_set.py sample
    python scripts\\build_golden_set.py label
    python scripts\\build_golden_set.py report

macOS/Linux:
    python scripts/build_golden_set.py sample
    python scripts/build_golden_set.py label
    python scripts/build_golden_set.py report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as cfg  # noqa: E402
from src.intents import classifier as clf_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_golden_set")

CANDIDATES_COLUMNS = [
    "id",
    "tweet_id",
    "conversation_id",
    "text",
    "brand_historical_reply",
    "sampling_stratum",
    "intent",
    "expected_escalation",
    "notes",
    "labeled_at",
]


def _pairs_path() -> Path:
    return cfg.PROCESSED_DIR / f"{cfg.CONFIG.brand.author_id.lower()}_resolution_pairs.parquet"


def _candidates_path() -> Path:
    return cfg.GOLDEN_DIR / "golden_set_candidates.csv"


def _golden_set_path() -> Path:
    return cfg.GOLDEN_DIR / "golden_set.csv"


def _methodology_path() -> Path:
    return cfg.GOLDEN_DIR / "METHODOLOGY.md"


def stratified_sample(
    pairs: pd.DataFrame, target_size: int, seed: int
) -> pd.DataFrame:
    """Sample `target_size` rows from the golden_eval split, stratified by
    an unsupervised cluster id so the sample spans the data's real
    diversity rather than whatever's most common.

    Falls back to a plain random sample (documented) if clustering isn't
    available yet (e.g. too few golden_eval rows, or discover.py hasn't
    been run) -- never blocks golden-set creation on an optional step.
    """
    golden = pairs[pairs["split"] == "golden_eval"].copy()
    golden = golden[golden["customer_text"].fillna("").str.strip() != ""]
    golden = golden.drop_duplicates(subset="customer_tweet_id")

    rng = np.random.RandomState(seed)

    if len(golden) <= target_size:
        logger.warning(
            "Only %d golden_eval candidates available (< target %d); "
            "taking all of them.",
            len(golden),
            target_size,
        )
        golden["sampling_stratum"] = "all"
        return golden

    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        n_clusters = min(10, max(2, len(golden) // 50))
        vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=2)
        matrix = vec.fit_transform(golden["customer_text"].tolist())
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        golden["sampling_stratum"] = model.fit_predict(matrix).astype(str)
        method = f"kmeans_k{n_clusters}"
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Clustering for stratification failed (%s); falling back to random.", exc)
        golden["sampling_stratum"] = "random"
        method = "random"

    if method == "random":
        return golden.sample(n=target_size, random_state=seed)

    # Proportional allocation per stratum, with at least 1 slot per stratum
    # so small-but-real clusters aren't sampled away entirely.
    strata_sizes = golden["sampling_stratum"].value_counts()
    allocation = (strata_sizes / strata_sizes.sum() * target_size).round().astype(int)
    allocation = allocation.clip(lower=1)
    while allocation.sum() > target_size:
        allocation[allocation.idxmax()] -= 1
    while allocation.sum() < target_size:
        allocation[allocation.idxmin()] += 1

    parts = []
    for stratum, n in allocation.items():
        pool = golden[golden["sampling_stratum"] == stratum]
        n = min(n, len(pool))
        parts.append(pool.sample(n=n, random_state=seed))
    sample = pd.concat(parts, ignore_index=True)
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle presentation order


def cmd_sample(args: argparse.Namespace) -> None:
    pairs_path = _pairs_path()
    if not pairs_path.exists():
        raise FileNotFoundError(f"{pairs_path} not found. Run: python scripts/preprocess.py")
    pairs = pd.read_parquet(pairs_path)

    sample = stratified_sample(pairs, args.target_size, args.seed)
    out = pd.DataFrame(
        {
            "id": [f"golden_{i:04d}" for i in range(len(sample))],
            "tweet_id": sample["customer_tweet_id"].values,
            "conversation_id": sample["conversation_id"].values,
            "text": sample["customer_text"].values,
            "brand_historical_reply": sample["brand_text"].values,
            "sampling_stratum": sample["sampling_stratum"].values,
            "intent": "",
            "expected_escalation": "",
            "notes": "",
            "labeled_at": "",
        }
    )
    cfg.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(_candidates_path(), index=False)
    logger.info(
        "Wrote %d unlabeled candidates to %s. Strata: %s",
        len(out),
        _candidates_path(),
        out["sampling_stratum"].value_counts().to_dict(),
    )
    logger.info("Next: python scripts/build_golden_set.py label")


def _load_in_progress() -> pd.DataFrame:
    if _golden_set_path().exists():
        return pd.read_csv(_golden_set_path(), dtype=str, keep_default_na=False)
    if not _candidates_path().exists():
        raise FileNotFoundError(
            f"No candidates found. Run: python scripts/build_golden_set.py sample"
        )
    return pd.read_csv(_candidates_path(), dtype=str, keep_default_na=False)


def cmd_label(args: argparse.Namespace) -> None:
    import datetime

    taxonomy = clf_mod.load_taxonomy(cfg.CONFIG_DIR / "intents.yaml")
    intent_names = clf_mod.taxonomy_intent_names(taxonomy)

    df = _load_in_progress()
    unlabeled_idx = df.index[df["intent"].fillna("").str.strip() == ""]

    if len(unlabeled_idx) == 0:
        print("All candidates are already labeled. Run `report` to summarize.")
        return

    print(f"\n{len(unlabeled_idx)} of {len(df)} examples remain unlabeled.")
    print("Intent options:")
    for i, name in enumerate(intent_names, start=1):
        print(f"  {i}. {name}")
    print("Enter the intent NUMBER, 'skip' to skip for now, or 'quit' to stop and save.\n")

    for idx in unlabeled_idx:
        row = df.loc[idx]
        print("-" * 72)
        print(f"[{row['id']}] Customer: {row['text']}")
        print(f"Historical {cfg.CONFIG.brand.author_id} reply: {row['brand_historical_reply']}")
        while True:
            choice = input("Intent #: ").strip().lower()
            if choice == "quit":
                df.to_csv(_golden_set_path(), index=False)
                print(f"Saved progress to {_golden_set_path()}. Resume any time with the same command.")
                return
            if choice == "skip":
                break
            if choice.isdigit() and 1 <= int(choice) <= len(intent_names):
                intent = intent_names[int(choice) - 1]
                escalation = input("Should this have been ESCALATED to a human? (y/n): ").strip().lower()
                notes = input("Notes (optional, e.g. why ambiguous): ").strip()
                df.loc[idx, "intent"] = intent
                df.loc[idx, "expected_escalation"] = "True" if escalation.startswith("y") else "False"
                df.loc[idx, "notes"] = notes
                df.loc[idx, "labeled_at"] = datetime.datetime.utcnow().isoformat()
                break
            print("Invalid input. Enter a number 1-%d, 'skip', or 'quit'." % len(intent_names))
        df.to_csv(_golden_set_path(), index=False)  # save after every example

    print(f"\nAll examples labeled. Saved to {_golden_set_path()}.")


def cmd_report(args: argparse.Namespace) -> None:
    if not _golden_set_path().exists():
        raise FileNotFoundError(
            f"{_golden_set_path()} not found. Run `sample` then `label` first."
        )
    df = pd.read_csv(_golden_set_path(), dtype=str, keep_default_na=False)
    labeled = df[df["intent"].fillna("").str.strip() != ""]

    distribution = {
        "total_candidates": len(df),
        "total_labeled": len(labeled),
        "labeling_complete": len(labeled) == len(df),
        "intent_distribution": labeled["intent"].value_counts().to_dict(),
        "escalation_distribution": labeled["expected_escalation"].value_counts().to_dict(),
        "sampling_stratum_distribution": df["sampling_stratum"].value_counts().to_dict(),
    }
    if len(labeled) < cfg.CONFIG.golden.min_size:
        logger.warning(
            "Only %d labeled examples; target range is %d-%d.",
            len(labeled),
            cfg.CONFIG.golden.min_size,
            cfg.CONFIG.golden.max_size,
        )

    out_path = cfg.GOLDEN_DIR / "label_distribution.json"
    out_path.write_text(json.dumps(distribution, indent=2))
    logger.info("Wrote %s", out_path)
    logger.info(json.dumps(distribution, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    p_sample = sub.add_parser("sample", help="Sample unlabeled candidates from the golden_eval split.")
    p_sample.add_argument("--target-size", type=int, default=cfg.CONFIG.golden.target_size)
    p_sample.add_argument("--seed", type=int, default=cfg.CONFIG.golden.random_seed)
    p_sample.set_defaults(func=cmd_sample)

    p_label = sub.add_parser("label", help="Interactive CLI to hand-label candidates.")
    p_label.set_defaults(func=cmd_label)

    p_report = sub.add_parser("report", help="Summarize label distribution once labeling is done.")
    p_report.set_defaults(func=cmd_report)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg.ensure_output_dirs()
    args.func(args)


if __name__ == "__main__":
    main()
