"""Bootstrap ("weak") intent labels from the taxonomy's own seed examples.

The production/baseline classifier (src/intents/classifier.py) needs
labeled training data. We don't have human-labeled training data -- the
one hand-labeled set this project produces (data/golden/golden_set.csv) is
deliberately held out for evaluation only, never used for training (that
would be leakage of the exact kind DECISIONS.md #3 exists to prevent).

Instead, we bootstrap pseudo-labels via nearest-centroid similarity
against each taxonomy intent's `examples` field: every historical
customer message is compared (TF-IDF cosine similarity) against the
taxonomy's seed examples for each intent, and assigned the closest
intent's label -- or `other_unclear` if no intent is similar enough.

This is explicitly weak/distant supervision, not ground truth. The
classifier trained on these labels is evaluated for real against the
actual hand-labeled golden set in Phase 9; if that evaluation shows poor
accuracy, the honest conclusion is "the weak-label bootstrap wasn't good
enough," not that the golden-set evaluation is wrong. See DECISIONS.md.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OTHER_UNCLEAR = "other_unclear"


def seed_centroid_labels(
    texts: list[str], taxonomy: dict, min_similarity: float = 0.05
) -> list[str]:
    """Return one pseudo-label per input text.

    `min_similarity` is a low floor (not a confidence threshold in the
    statistical sense) purely to catch texts with essentially no lexical
    overlap with any intent's seed examples, routing them to
    `other_unclear` rather than an arbitrary best-of-a-bad-lot match.
    """
    if not texts:
        return []

    anchor_texts: list[str] = []
    anchor_intents: list[str] = []
    for intent in taxonomy["intents"]:
        for example in intent.get("examples", []):
            anchor_texts.append(example)
            anchor_intents.append(intent["name"])
    if not anchor_texts:
        return [OTHER_UNCLEAR] * len(texts)

    vectorizer = TfidfVectorizer(stop_words="english")
    combined = vectorizer.fit_transform(anchor_texts + texts)
    anchor_matrix = combined[: len(anchor_texts)]
    text_matrix = combined[len(anchor_texts) :]

    sims = cosine_similarity(text_matrix, anchor_matrix)
    labels = []
    for row in sims:
        best_idx = int(np.argmax(row))
        best_score = row[best_idx]
        labels.append(anchor_intents[best_idx] if best_score >= min_similarity else OTHER_UNCLEAR)
    return labels
