"""Human-vs-LLM-judge agreement on the reply-quality rubric.

The assignment requires evidence the judge agrees with a human, not just
that a judge score exists. This module computes that agreement on a
small, real, manually-reviewed subset -- it never fabricates human
ratings; every function here takes real human ratings as input, and
`compute_agreement_report` refuses to run on synthetic/placeholder data
(see `scripts/evaluate.py`'s `human_review` stage, which -- like golden-set
labeling -- is an interactive step requiring a real person).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

from src.evaluation.judge import RUBRIC_BOOLEAN_DIMENSIONS, RUBRIC_SCORE_DIMENSIONS


def weighted_kappa(a: list[int], b: list[int]) -> float:
    """Linear-weighted Cohen's kappa for ordinal 1-5 rubric scores."""
    if len(set(a)) < 2 and len(set(b)) < 2 and set(a) == set(b):
        return 1.0  # both raters gave the same single value on every item
    return float(cohen_kappa_score(a, b, weights="linear"))


def unweighted_kappa(a: list[bool], b: list[bool]) -> float:
    """Plain Cohen's kappa for the boolean no_hallucination dimension."""
    if len(set(a)) < 2 and len(set(b)) < 2 and set(a) == set(b):
        return 1.0
    return float(cohen_kappa_score(a, b))


def pearson_r(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2:
        return None  # undefined (no variance) -- report as None, never fabricate 0 or 1
    r, _ = pearsonr(a, b)
    return float(r)


def compute_agreement_report(
    human_df: pd.DataFrame, judge_df: pd.DataFrame, id_col: str = "id"
) -> dict:
    """`human_df` and `judge_df` must each have `id_col` plus every column
    in RUBRIC_SCORE_DIMENSIONS + RUBRIC_BOOLEAN_DIMENSIONS. Merges on
    `id_col` (inner join -- only examples both rated) and computes, per
    dimension, weighted kappa (or plain kappa for the boolean dimension),
    Pearson correlation, and mean absolute difference.
    """
    required = {id_col, *RUBRIC_SCORE_DIMENSIONS, *RUBRIC_BOOLEAN_DIMENSIONS}
    missing_human = required - set(human_df.columns)
    missing_judge = required - set(judge_df.columns)
    if missing_human:
        raise ValueError(f"human_df missing columns: {missing_human}")
    if missing_judge:
        raise ValueError(f"judge_df missing columns: {missing_judge}")

    merged = human_df.merge(judge_df, on=id_col, suffixes=("_human", "_judge"))
    if len(merged) == 0:
        raise ValueError("No overlapping ids between human_df and judge_df -- nothing to compare.")

    report: dict = {"n_examples": len(merged)}
    for dim in RUBRIC_SCORE_DIMENSIONS:
        h = merged[f"{dim}_human"].astype(int).tolist()
        j = merged[f"{dim}_judge"].astype(int).tolist()
        report[dim] = {
            "weighted_kappa": weighted_kappa(h, j),
            "pearson_r": pearson_r([float(x) for x in h], [float(x) for x in j]),
            "mean_abs_diff": float(np.mean(np.abs(np.array(h) - np.array(j)))),
        }
    for dim in RUBRIC_BOOLEAN_DIMENSIONS:
        h = merged[f"{dim}_human"].astype(bool).tolist()
        j = merged[f"{dim}_judge"].astype(bool).tolist()
        agreement_rate = sum(1 for x, y in zip(h, j) if x == y) / len(h)
        report[dim] = {
            "kappa": unweighted_kappa(h, j),
            "agreement_rate": agreement_rate,
        }

    overall_kappa = float(
        np.mean([report[d]["weighted_kappa"] for d in RUBRIC_SCORE_DIMENSIONS])
    )
    report["overall_mean_weighted_kappa"] = overall_kappa
    report["interpretation"] = _interpret_kappa(overall_kappa)
    return report


def _interpret_kappa(kappa: float) -> str:
    # Landis & Koch (1977) conventional bands -- widely used, not this
    # project's invention.
    if kappa < 0:
        return "poor (worse than chance)"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"
