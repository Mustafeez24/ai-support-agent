"""Pure metric functions: intent classification, escalation policy, and a
retrieval "relevance proxy" metric. No LLM calls, no I/O -- everything here
takes plain Python/lists/DataFrames in and returns plain dicts out, so it's
cheap to unit test exhaustively.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def intent_metrics(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict:
    """Accuracy, macro F1, per-intent precision/recall/F1, confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) length mismatch")
    if len(y_true) == 0:
        raise ValueError("Cannot compute intent_metrics on empty input.")

    labels = labels or sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_intent = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "n_examples": len(y_true),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_intent": per_intent,
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }


def escalation_metrics(y_true_escalate: list[bool], y_pred_escalate: list[bool]) -> dict:
    """Escalation treated as the positive class: accuracy, precision,
    recall, F1, and the "false auto-handle rate" -- the fraction of
    examples that SHOULD have been escalated but the system auto-handled
    instead. This is the single most safety-relevant number in the whole
    evaluation: a false auto-handle means a customer got an unsupervised
    reply on an issue a human judged needed one.
    """
    if len(y_true_escalate) != len(y_pred_escalate):
        raise ValueError("y_true_escalate and y_pred_escalate length mismatch")
    if len(y_true_escalate) == 0:
        raise ValueError("Cannot compute escalation_metrics on empty input.")

    accuracy = accuracy_score(y_true_escalate, y_pred_escalate)
    precision = precision_score(y_true_escalate, y_pred_escalate, zero_division=0)
    recall = recall_score(y_true_escalate, y_pred_escalate, zero_division=0)
    f1 = f1_score(y_true_escalate, y_pred_escalate, zero_division=0)

    should_escalate_count = sum(1 for t in y_true_escalate if t)
    false_auto_handle_count = sum(
        1 for t, p in zip(y_true_escalate, y_pred_escalate) if t and not p
    )
    false_auto_handle_rate = (
        false_auto_handle_count / should_escalate_count if should_escalate_count else 0.0
    )

    return {
        "n_examples": len(y_true_escalate),
        "accuracy": float(accuracy),
        "precision_escalate": float(precision),
        "recall_escalate": float(recall),
        "f1_escalate": float(f1),
        "should_escalate_count": should_escalate_count,
        "false_auto_handle_count": false_auto_handle_count,
        "false_auto_handle_rate": float(false_auto_handle_rate),
    }


def retrieval_intent_match_at_k(
    true_intents: list[str],
    retrieved_intents_per_query: list[list[str]],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """PROXY retrieval-relevance metric: for each query, does the true
    intent label appear among the (classifier-predicted) intents of the
    top-K retrieved historical customer messages?

    This is explicitly a proxy, not true Recall@K against hand-labeled
    relevance judgments -- the dataset has no "this retrieved item is/isn't
    relevant to this query" ground truth, and creating one was out of
    scope. "Intent-consistency" is a reasonable stand-in (good retrieval
    should mostly surface same-intent historical cases) but is a known
    limitation -- see report.md Section 12.
    """
    if len(true_intents) != len(retrieved_intents_per_query):
        raise ValueError("true_intents and retrieved_intents_per_query length mismatch")
    n = len(true_intents)
    if n == 0:
        return {f"recall_at_{k}": 0.0 for k in k_values} | {"n_examples": 0}

    results: dict = {"n_examples": n}
    for k in k_values:
        hits = sum(
            1
            for true_intent, retrieved in zip(true_intents, retrieved_intents_per_query)
            if true_intent in retrieved[:k]
        )
        results[f"recall_at_{k}"] = hits / n
    return results
