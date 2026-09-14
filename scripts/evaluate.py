#!/usr/bin/env python3
"""Phase 9 evaluation CLI. Three stages:

  run                  - runs the production agent + both baselines over
                          data/golden/golden_set.csv, computes intent/
                          escalation/retrieval metrics for each, and (if
                          NVIDIA_API_KEY is configured) LLM-judges every
                          auto-handled reply against the quality rubric.
                          Writes outputs/metrics/evaluation_report.json
                          and outputs/metrics/comparison_table.csv.

  human-review-template - samples `--sample-size` (default from config)
                          of the judged examples and writes an UNRATED
                          CSV (outputs/metrics/human_review_template.csv)
                          with the same rubric columns, for a real person
                          to fill in -- the assignment requires evidence
                          the judge agrees with a human, not just a judge
                          score.

  agreement             - once outputs/metrics/human_review_ratings.csv
                          exists (the template above, filled in by a
                          human), computes human-vs-judge agreement
                          (weighted kappa, correlation) and writes
                          outputs/metrics/judge_agreement_report.json.

Windows:
    python scripts\\evaluate.py run
    python scripts\\evaluate.py human-review-template
    (fill in outputs\\metrics\\human_review_template.csv by hand, save as
     outputs\\metrics\\human_review_ratings.csv)
    python scripts\\evaluate.py agreement

macOS/Linux: same commands with scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config as cfg  # noqa: E402
from src.agent.agent import load_default_agent  # noqa: E402
from src.evaluation import agreement as agreement_mod  # noqa: E402
from src.evaluation import evaluate as evaluate_mod  # noqa: E402
from src.evaluation.judge import ALL_RUBRIC_DIMENSIONS  # noqa: E402
from src.intents.baselines import TfidfNearestNeighborBaseline, TrivialBaseline  # noqa: E402
from src.intents.classifier import load_taxonomy  # noqa: E402
from src.intents.weak_labels import seed_centroid_labels  # noqa: E402
from src.retrieval.embeddings import TfidfEmbedder  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


def _golden_set_path() -> Path:
    return cfg.GOLDEN_DIR / "golden_set.csv"


def _pairs_path() -> Path:
    return cfg.PROCESSED_DIR / f"{cfg.CONFIG.brand.author_id.lower()}_resolution_pairs.parquet"


def _load_golden_set() -> pd.DataFrame:
    path = _golden_set_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the golden set first: "
            "python scripts/build_golden_set.py sample && "
            "python scripts/build_golden_set.py label"
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    unlabeled = df[df["intent"].fillna("").str.strip() == ""]
    if len(unlabeled) > 0:
        raise ValueError(
            f"{len(unlabeled)}/{len(df)} golden set examples are unlabeled. "
            "Finish labeling first: python scripts/build_golden_set.py label"
        )
    return df


def _build_baseline2(pairs_path: Path, taxonomy: dict) -> TfidfNearestNeighborBaseline:
    from src.intents.classifier import IntentClassifier

    pairs = pd.read_parquet(pairs_path)
    train = pairs[pairs["split"] == "train_retrieval"]
    train = train[train["customer_text"].fillna("").str.strip() != ""]

    labels = seed_centroid_labels(train["customer_text"].tolist(), taxonomy)
    classifier = IntentClassifier(random_seed=cfg.CONFIG.data.random_seed)
    classifier.fit(train["customer_text"].tolist(), labels)

    embedder = TfidfEmbedder().fit(train["customer_text"].tolist())
    retriever = Retriever(embedder, top_k=cfg.CONFIG.retrieval.top_k)
    retriever.build(train)

    return TfidfNearestNeighborBaseline(classifier, retriever, taxonomy, cfg.CONFIG.escalation)


def _build_baseline1(pairs_path: Path, taxonomy: dict) -> TrivialBaseline:
    pairs = pd.read_parquet(pairs_path)
    train = pairs[pairs["split"] == "train_retrieval"]
    train = train[train["customer_text"].fillna("").str.strip() != ""]
    labels = seed_centroid_labels(train["customer_text"].tolist(), taxonomy)
    return TrivialBaseline.fit(labels)


def _evidence_intent_fn(classifier):
    def fn(evidence: list[dict]) -> list[str]:
        if not evidence:
            return []
        texts = [e["customer_text"] for e in evidence]
        return [p.intent for p in classifier.predict(texts)]

    return fn


def cmd_run(args: argparse.Namespace) -> None:
    cfg.ensure_output_dirs()
    golden_df = _load_golden_set()
    texts = golden_df["text"].tolist()
    true_intents = golden_df["intent"].tolist()
    true_escalations = evaluate_mod.parse_expected_escalation(golden_df["expected_escalation"]).tolist()

    taxonomy = load_taxonomy(cfg.CONFIG_DIR / "intents.yaml")
    pairs_path = _pairs_path()
    if not pairs_path.exists():
        raise FileNotFoundError(f"{pairs_path} not found. Run: python scripts/preprocess.py")

    logger.info("Running production agent over %d golden examples", len(golden_df))
    agent = load_default_agent()
    prod_preds = evaluate_mod.run_system_over_golden_set(
        agent, texts, "production", evidence_intent_fn=_evidence_intent_fn(agent.classifier)
    )

    logger.info("Running Baseline 1 (trivial)")
    baseline1 = _build_baseline1(pairs_path, taxonomy)
    b1_preds = evaluate_mod.run_system_over_golden_set(baseline1, texts, "baseline1_trivial")

    logger.info("Running Baseline 2 (TF-IDF, non-LLM)")
    baseline2 = _build_baseline2(pairs_path, taxonomy)
    b2_preds = evaluate_mod.run_system_over_golden_set(
        baseline2, texts, "baseline2_tfidf", evidence_intent_fn=_evidence_intent_fn(baseline2.classifier)
    )

    reports = [
        evaluate_mod.evaluate_system(prod_preds, true_intents, true_escalations),
        evaluate_mod.evaluate_system(b1_preds, true_intents, true_escalations),
        evaluate_mod.evaluate_system(b2_preds, true_intents, true_escalations),
    ]
    comparison = evaluate_mod.build_comparison_table(reports)
    comparison.to_csv(cfg.METRICS_DIR / "comparison_table.csv", index=False, encoding="utf-8")
    (cfg.METRICS_DIR / "evaluation_report.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Wrote comparison table and evaluation report to %s", cfg.METRICS_DIR)
    logger.info("\n%s", comparison.to_string(index=False))

    judge_results = []
    if agent.llm.is_configured():
        logger.info("LLM configured -- judging auto-handled production replies")
        judge_results = evaluate_mod.judge_auto_handled_replies(
            agent.llm, golden_df["id"].tolist(), texts, prod_preds, cfg.CONFIG.brand.author_id
        )
        pd.DataFrame(judge_results).to_csv(cfg.METRICS_DIR / "judge_scores.csv", index=False, encoding="utf-8")
        logger.info("Judged %d/%d auto-handled replies -> %s", len(judge_results), sum(1 for e in prod_preds.escalations if not e), cfg.METRICS_DIR / "judge_scores.csv")
    else:
        logger.warning(
            "NVIDIA_API_KEY not configured -- skipping LLM-judge reply-quality "
            "scoring. Set it in .env to enable this step."
        )


def cmd_human_review_template(args: argparse.Namespace) -> None:
    scores_path = cfg.METRICS_DIR / "judge_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found. Run: python scripts/evaluate.py run")
    judged = pd.read_csv(scores_path)
    sample = judged.sample(
        n=min(args.sample_size, len(judged)), random_state=cfg.CONFIG.evaluation.random_seed
    )
    template = sample[["id", "text", "reply"]].copy()
    for dim in ALL_RUBRIC_DIMENSIONS:
        template[dim] = ""
    template["reviewer_notes"] = ""
    out_path = cfg.METRICS_DIR / "human_review_template.csv"
    template.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(
        "Wrote %d unrated examples to %s. A human must fill in columns %s "
        "(scores 1-5, no_hallucination as True/False), save as "
        "human_review_ratings.csv, then run: python scripts/evaluate.py agreement",
        len(template),
        out_path,
        list(ALL_RUBRIC_DIMENSIONS),
    )


def cmd_agreement(args: argparse.Namespace) -> None:
    ratings_path = cfg.METRICS_DIR / "human_review_ratings.csv"
    scores_path = cfg.METRICS_DIR / "judge_scores.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"{ratings_path} not found. Run `human-review-template`, have a "
            "human fill it in, and save it at this exact path first."
        )
    human_df = pd.read_csv(ratings_path)
    judge_df = pd.read_csv(scores_path)
    report = agreement_mod.compute_agreement_report(human_df, judge_df)
    out_path = cfg.METRICS_DIR / "judge_agreement_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info(json.dumps(report, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    p_run = sub.add_parser("run", help="Run all systems over the golden set and compute metrics.")
    p_run.set_defaults(func=cmd_run)

    p_template = sub.add_parser("human-review-template", help="Sample judged examples for human rating.")
    p_template.add_argument(
        "--sample-size", type=int, default=cfg.CONFIG.evaluation.judge_human_review_sample_size
    )
    p_template.set_defaults(func=cmd_human_review_template)

    p_agreement = sub.add_parser("agreement", help="Compute human-vs-judge agreement.")
    p_agreement.set_defaults(func=cmd_agreement)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
