"""Tests for intent discovery (clustering) and the TF-IDF classifier.

Discovery is tested on a small synthetic corpus with two obviously
separable topics (not the real dataset). Classifier tests train/predict on
synthetic labeled examples. The taxonomy-loading tests run against the
real `config/intents.yaml` shipped in this repo.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.intents import classifier as clf_mod
from src.intents import discover

REPO_ROOT = Path(__file__).parent.parent
TAXONOMY_PATH = REPO_ROOT / "config" / "intents.yaml"

DELIVERY_TEXTS = [
    "My order still hasn't arrived and it's 3 days late",
    "Where is my package, it was supposed to ship yesterday",
    "Any update on my delivery status for order 555",
    "The tracking hasn't moved in days, when will it arrive",
    "I'm still waiting on my order, it's very late",
    "Order is delayed, no updates on shipping",
    "When is my package going to be delivered",
    "Still no sign of my order, it's overdue",
    "My delivery is late again this week",
    "Package tracking shows no movement for 5 days",
]

DAMAGED_TEXTS = [
    "The item I received is completely broken",
    "This product arrived damaged and unusable",
    "My order came in with a shattered screen",
    "The box arrived crushed and the item inside is damaged",
    "Received a defective unit, it won't turn on",
    "The item is damaged, I need a replacement",
    "This arrived broken out of the box",
    "The product I got is defective and cracked",
    "My item was damaged during shipping",
    "Got a broken item, need a replacement quickly",
]


# --- discover.py -------------------------------------------------------------


def _make_synthetic_pairs_parquet(tmp_path: Path) -> Path:
    rows = []
    for i, text in enumerate(DELIVERY_TEXTS + DAMAGED_TEXTS):
        rows.append(
            {
                "conversation_id": f"c{i}",
                "customer_tweet_id": f"t{i}",
                "customer_text": text,
                "customer_created_at": "2017-01-01",
                "brand_tweet_id": f"b{i}",
                "brand_text": "We're on it!",
                "brand_created_at": "2017-01-01",
                "split": "train_retrieval",
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "pairs.parquet"
    df.to_parquet(path, index=False)
    return path


def test_load_training_customer_texts_excludes_golden_split(tmp_path):
    pairs_path = _make_synthetic_pairs_parquet(tmp_path)
    df = pd.read_parquet(pairs_path)
    df.loc[0, "split"] = "golden_eval"
    df.to_parquet(pairs_path, index=False)

    train = discover.load_training_customer_texts(pairs_path)
    assert "golden_eval" not in train["split"].unique()
    assert len(train) == len(DELIVERY_TEXTS) + len(DAMAGED_TEXTS) - 1


def test_run_discovery_separates_two_obvious_topics(tmp_path):
    pairs_path = _make_synthetic_pairs_parquet(tmp_path)
    out_path = tmp_path / "report.json"
    report = discover.run_discovery(
        pairs_path, out_path, k_range=range(2, 3), chosen_k=2, seed=42
    )
    assert report["n_training_messages"] == 20
    assert len(report["clusters"]) == 2
    assert out_path.exists()

    # Each cluster's top terms should be dominated by one topic's vocabulary.
    all_terms = {c["cluster_id"]: set(c["top_terms"][:5]) for c in report["clusters"]}
    delivery_vocab = {"order", "package", "arrive", "delivery", "delivered", "late", "shipping", "days"}
    damaged_vocab = {"damaged", "broken", "defective", "arrived", "item", "replacement", "cracked"}
    overlaps = [
        (len(terms & delivery_vocab), len(terms & damaged_vocab)) for terms in all_terms.values()
    ]
    # At least one cluster should lean clearly delivery, another clearly damaged.
    assert any(d > dm for d, dm in overlaps)
    assert any(dm > d for d, dm in overlaps)


# --- classifier.py -------------------------------------------------------------


def test_intent_classifier_fits_and_predicts_synthetic_topics():
    texts = DELIVERY_TEXTS + DAMAGED_TEXTS
    labels = ["delivery_delay"] * len(DELIVERY_TEXTS) + ["damaged_or_defective_item"] * len(DAMAGED_TEXTS)
    model = clf_mod.IntentClassifier(random_seed=42)
    model.fit(texts, labels)

    preds = model.predict(["Still waiting on my late order", "My item arrived shattered"])
    assert preds[0].intent == "delivery_delay"
    assert preds[1].intent == "damaged_or_defective_item"
    assert 0.0 <= preds[0].confidence <= 1.0
    assert set(preds[0].all_scores.keys()) == {"delivery_delay", "damaged_or_defective_item"}


def test_intent_classifier_predict_before_fit_raises():
    model = clf_mod.IntentClassifier()
    with pytest.raises(RuntimeError):
        model.predict(["hello"])


def test_intent_classifier_save_load_round_trip(tmp_path):
    texts = DELIVERY_TEXTS + DAMAGED_TEXTS
    labels = ["delivery_delay"] * len(DELIVERY_TEXTS) + ["damaged_or_defective_item"] * len(DAMAGED_TEXTS)
    model = clf_mod.IntentClassifier(random_seed=42)
    model.fit(texts, labels)
    path = tmp_path / "model.joblib"
    model.save(path)

    loaded = clf_mod.IntentClassifier.load(path)
    original_preds = model.predict(["My package never arrived"])
    loaded_preds = loaded.predict(["My package never arrived"])
    assert original_preds[0].intent == loaded_preds[0].intent
    assert original_preds[0].confidence == pytest.approx(loaded_preds[0].confidence)


# --- taxonomy loading (real config/intents.yaml) --------------------------------


def test_taxonomy_file_loads_and_has_required_fields():
    taxonomy = clf_mod.load_taxonomy(TAXONOMY_PATH)
    assert 8 <= len(taxonomy["intents"]) <= 12
    for intent in taxonomy["intents"]:
        for field in ("name", "description", "examples", "inclusion_criteria", "exclusion_criteria"):
            assert field in intent, f"intent {intent.get('name')} missing {field}"
        assert len(intent["examples"]) >= 1


def test_taxonomy_intent_names_are_unique():
    taxonomy = clf_mod.load_taxonomy(TAXONOMY_PATH)
    names = clf_mod.taxonomy_intent_names(taxonomy)
    assert len(names) == len(set(names))


def test_always_escalate_intents_matches_config_defaults():
    from src import config as cfg

    taxonomy = clf_mod.load_taxonomy(TAXONOMY_PATH)
    yaml_always_escalate = clf_mod.always_escalate_intents(taxonomy)
    assert yaml_always_escalate == set(cfg.CONFIG.escalation.always_escalate_intents)


# --- weak_labels.py -------------------------------------------------------------


def test_seed_centroid_labels_matches_obvious_topics():
    from src.intents import weak_labels

    taxonomy = clf_mod.load_taxonomy(TAXONOMY_PATH)
    texts = [
        "my package has not arrived and it was supposed to be here days ago",
        "this item arrived broken and damaged out of the box",
        "gibberish nonsense text with no meaning at all zzz",
    ]
    labels = weak_labels.seed_centroid_labels(texts, taxonomy, min_similarity=0.05)
    assert labels[0] == "delivery_delay"
    assert labels[1] == "damaged_or_defective_item"
    assert labels[2] == "other_unclear"


def test_seed_centroid_labels_empty_input():
    from src.intents import weak_labels

    taxonomy = clf_mod.load_taxonomy(TAXONOMY_PATH)
    assert weak_labels.seed_centroid_labels([], taxonomy) == []


# --- baselines.py -------------------------------------------------------------


def test_trivial_baseline_always_escalates_with_majority_intent():
    from src.intents.baselines import ESCALATE_TO_HUMAN, TrivialBaseline

    baseline = TrivialBaseline.fit(["delivery_delay", "delivery_delay", "damaged_or_defective_item"])
    result = baseline.predict("anything at all")
    assert result.intent == "delivery_delay"
    assert result.decision == ESCALATE_TO_HUMAN
    assert result.evidence == []


def test_trivial_baseline_fit_requires_labels():
    from src.intents.baselines import TrivialBaseline

    with pytest.raises(ValueError):
        TrivialBaseline.fit([])


def test_tfidf_nearest_neighbor_baseline_returns_verbatim_reply():
    import pandas as pd

    from src.intents.baselines import AUTO_HANDLE, TfidfNearestNeighborBaseline
    from src.retrieval.embeddings import TfidfEmbedder
    from src.retrieval.retriever import Retriever

    texts = DELIVERY_TEXTS + DAMAGED_TEXTS
    labels = ["delivery_delay"] * len(DELIVERY_TEXTS) + ["damaged_or_defective_item"] * len(DAMAGED_TEXTS)
    classifier = clf_mod.IntentClassifier(random_seed=42)
    classifier.fit(texts, labels)

    pairs = pd.DataFrame(
        {
            "conversation_id": [f"c{i}" for i in range(len(texts))],
            "customer_tweet_id": [f"t{i}" for i in range(len(texts))],
            "customer_text": texts,
            "brand_tweet_id": [f"b{i}" for i in range(len(texts))],
            "brand_text": [f"HISTORICAL REPLY {i}" for i in range(len(texts))],
            "split": "train_retrieval",
        }
    )
    embedder = TfidfEmbedder().fit(texts)
    retriever = Retriever(embedder, top_k=3)
    retriever.build(pairs)

    taxonomy = clf_mod.load_taxonomy(TAXONOMY_PATH)
    from src.config import EscalationConfig

    lenient_config = EscalationConfig(
        min_intent_confidence=0.0,
        min_evidence_similarity=0.0,
        min_evidence_count=1,
        always_escalate_intents=(),
    )
    baseline = TfidfNearestNeighborBaseline(classifier, retriever, taxonomy, lenient_config)
    result = baseline.predict("My package never arrived, still waiting")
    assert result.decision == AUTO_HANDLE
    assert result.reply.startswith("HISTORICAL REPLY")  # verbatim, not generated


def test_tfidf_embedder_requires_fit_before_embed():
    from src.retrieval.embeddings import TfidfEmbedder

    embedder = TfidfEmbedder()
    with pytest.raises(RuntimeError):
        embedder.embed(["hello"])
