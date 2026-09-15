"""Data-driven intent discovery over AmazonHelp customer messages.

Runs TF-IDF + MiniBatchKMeans clustering on the *training-split* customer
messages (never the golden/held-out split, so the taxonomy is not designed
by peeking at evaluation data) and reports, per cluster, the top terms and
a sample of real messages. A human reviews this report and edits
`config/intents.yaml` accordingly — this module does not invent a taxonomy
on its own, it produces the evidence a taxonomy should be based on.

Usage:
    python -m src.intents.discover --pairs-path data/processed/amazonhelp_resolution_pairs.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config as cfg  # noqa: E402

logger = logging.getLogger(__name__)


def load_training_customer_texts(pairs_path: Path) -> pd.DataFrame:
    """Load customer_text values from the train_retrieval split only."""
    pairs = pd.read_parquet(pairs_path)
    train = pairs[pairs["split"] == "train_retrieval"].copy()
    train = train[train["customer_text"].fillna("").str.strip() != ""]
    return train.reset_index(drop=True)


def vectorize(texts: list[str], max_features: int = 5000) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def cluster(matrix, n_clusters: int, seed: int) -> MiniBatchKMeans:
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=10, batch_size=1024)
    model.fit(matrix)
    return model


def top_terms_per_cluster(
    vectorizer: TfidfVectorizer, model: MiniBatchKMeans, top_n: int = 15
) -> dict[int, list[str]]:
    terms = np.array(vectorizer.get_feature_names_out())
    result = {}
    for cluster_id, center in enumerate(model.cluster_centers_):
        top_idx = np.argsort(center)[::-1][:top_n]
        result[cluster_id] = terms[top_idx].tolist()
    return result


def sample_texts_per_cluster(
    texts: list[str], labels: np.ndarray, n_samples: int, seed: int
) -> dict[int, list[str]]:
    rng = np.random.RandomState(seed)
    result: dict[int, list[str]] = {}
    for cluster_id in sorted(set(labels.tolist())):
        idx = np.where(labels == cluster_id)[0]
        chosen = rng.choice(idx, size=min(n_samples, len(idx)), replace=False)
        result[cluster_id] = [texts[i] for i in chosen]
    return result


def elbow_inertias(matrix, k_range: range, seed: int) -> dict[int, float]:
    """Inertia per k, to help a human pick a sensible cluster count in the
    8-12 range the assignment asks for (look for the elbow, not the minimum)."""
    scores = {}
    for k in k_range:
        model = cluster(matrix, k, seed)
        scores[k] = float(model.inertia_)
    return scores


def run_discovery(
    pairs_path: Path,
    out_path: Path,
    k_range: range = range(8, 13),
    chosen_k: int = 10,
    seed: int = 42,
) -> dict:
    train = load_training_customer_texts(pairs_path)
    texts = train["customer_text"].tolist()
    if len(texts) < 20:
        logger.warning(
            "Only %d training customer messages available -- clustering "
            "quality will be poor below a few hundred; treat this run as a "
            "pipeline smoke test, not a real taxonomy source.",
            len(texts),
        )

    vectorizer, matrix = vectorize(texts)
    inertias = elbow_inertias(matrix, k_range, seed)

    model = cluster(matrix, chosen_k, seed)
    labels = model.labels_
    terms = top_terms_per_cluster(vectorizer, model)
    samples = sample_texts_per_cluster(texts, labels, n_samples=8, seed=seed)

    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    report = {
        "n_training_messages": len(texts),
        "chosen_k": chosen_k,
        "elbow_inertias_by_k": inertias,
        "clusters": [
            {
                "cluster_id": int(cid),
                "size": int(cluster_sizes.get(cid, 0)),
                "top_terms": terms[cid],
                "sample_messages": samples.get(cid, []),
            }
            for cid in sorted(terms.keys())
        ],
        "instructions": (
            "Review each cluster's top_terms and sample_messages. Map it to "
            "an intent name in config/intents.yaml (merge clusters that "
            "represent the same intent, split ones that don't). Replace "
            "each intent's `examples` field with real sample_messages from "
            "here, then set config/intents.yaml status to VALIDATED."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote intent discovery report to %s", out_path)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs-path",
        type=Path,
        default=cfg.PROCESSED_DIR / f"{cfg.CONFIG.brand.author_id.lower()}_resolution_pairs.parquet",
    )
    p.add_argument("--out-path", type=Path, default=cfg.ANALYSIS_DIR / "intent_clusters_report.json")
    p.add_argument("--k-min", type=int, default=8)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--chosen-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=cfg.CONFIG.data.random_seed)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.pairs_path.exists():
        raise FileNotFoundError(
            f"{args.pairs_path} not found. Run preprocessing first: "
            "python scripts/preprocess.py"
        )
    run_discovery(
        args.pairs_path,
        args.out_path,
        k_range=range(args.k_min, args.k_max + 1),
        chosen_k=args.chosen_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
