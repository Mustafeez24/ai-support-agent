"""Deterministic, local intent classifier (TF-IDF + Logistic Regression).

This is the classifier the agent uses in production (Phase 7): fast,
free, and — per the assignment's own guidance — preferred over an LLM call
for a well-scoped classification task. It is also reused as-is for
"Baseline 2" (the required simple non-LLM baseline), since a documented,
tuned baseline that *is* the production component is more honest than
building a second, deliberately-worse classifier just to have two numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    all_scores: dict[str, float]


class IntentClassifier:
    """Thin, serializable wrapper around a TF-IDF + LogisticRegression
    pipeline for multi-class intent classification."""

    def __init__(self, random_seed: int = 42, max_features: int = 5000):
        self.random_seed = random_seed
        self.max_features = max_features
        self.pipeline: Pipeline | None = None
        self.classes_: list[str] | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "IntentClassifier":
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=self.random_seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, texts: list[str]) -> list[IntentPrediction]:
        if self.pipeline is None:
            raise RuntimeError("IntentClassifier must be fit() or load()ed before predict().")
        proba = self.pipeline.predict_proba(texts)
        predictions = []
        for row in proba:
            scores = dict(zip(self.classes_, row.tolist()))
            best_idx = int(np.argmax(row))
            predictions.append(
                IntentPrediction(
                    intent=self.classes_[best_idx],
                    confidence=float(row[best_idx]),
                    all_scores=scores,
                )
            )
        return predictions

    def predict_one(self, text: str) -> IntentPrediction:
        return self.predict([text])[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "classes_": self.classes_, "random_seed": self.random_seed}, path)

    @classmethod
    def load(cls, path: Path) -> "IntentClassifier":
        payload = joblib.load(path)
        obj = cls(random_seed=payload["random_seed"])
        obj.pipeline = payload["pipeline"]
        obj.classes_ = payload["classes_"]
        return obj


def train_from_labeled_csv(csv_path: Path, text_col: str = "text", label_col: str = "intent") -> IntentClassifier:
    """Convenience trainer for a CSV with text+label columns (e.g. the
    golden set's train-side counterpart, or weak cluster-derived labels)."""
    df = pd.read_csv(csv_path)
    clf = IntentClassifier()
    clf.fit(df[text_col].tolist(), df[label_col].tolist())
    return clf


def load_taxonomy(path: Path) -> dict:
    """Load config/intents.yaml as a plain dict (kept dependency-light: a
    thin wrapper so callers don't need to know it's YAML)."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def taxonomy_intent_names(taxonomy: dict) -> list[str]:
    return [i["name"] for i in taxonomy["intents"]]


def always_escalate_intents(taxonomy: dict) -> set[str]:
    return {i["name"] for i in taxonomy["intents"] if i.get("always_escalate")}
