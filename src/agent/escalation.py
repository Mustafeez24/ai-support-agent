"""Explicit, configurable AUTO_HANDLE vs ESCALATE_TO_HUMAN policy.

Every rule below produces a human-readable reason string, concatenated
into the final `escalation_reason` when more than one rule fires. Nothing
here calls the LLM: the decision is deterministic given
(intent, confidence, evidence, taxonomy, thresholds), which is what makes
it auditable and cheaply testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import EscalationConfig
from src.retrieval.retriever import RetrievedEvidence

AUTO_HANDLE = "AUTO_HANDLE"
ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"

# Coarse action-keyword buckets used only to detect *conflicting* historical
# guidance (e.g. one similar past case was refunded, another was told to
# wait) -- a deliberately simple heuristic, not a semantic understanding of
# the replies. Documented limitation: see report.md Section 12.
_ACTION_KEYWORDS: dict[str, set[str]] = {
    "refund": {"refund", "refunded", "money back", "reimburse"},
    "replacement": {"replacement", "replace", "resend", "resending"},
    "return": {"return", "returned", "returning"},
    "cancel": {"cancel", "cancelled", "canceled", "cancellation"},
}


@dataclass
class EscalationDecision:
    decision: str  # AUTO_HANDLE | ESCALATE_TO_HUMAN
    reason: str
    triggered_rules: list[str] = field(default_factory=list)


def _extract_actions(text: str) -> set[str]:
    text_l = text.lower()
    return {action for action, kws in _ACTION_KEYWORDS.items() if any(kw in text_l for kw in kws)}


def has_conflicting_evidence(evidence: list[RetrievedEvidence], top_n: int = 3) -> bool:
    """True if the top retrieved historical resolutions suggest mutually
    exclusive actions (no common action-keyword bucket across all of them
    that mention an action at all)."""
    action_sets = [_extract_actions(e.brand_text) for e in evidence[:top_n]]
    non_empty = [a for a in action_sets if a]
    if len(non_empty) < 2:
        return False
    common = set.intersection(*non_empty)
    return len(common) == 0


def decide_escalation(
    intent: str,
    intent_confidence: float,
    evidence: list[RetrievedEvidence],
    taxonomy: dict,
    config: EscalationConfig,
) -> EscalationDecision:
    """Apply, in order:

    1. Taxonomy-level always_escalate intents (account access, payment
       disputes, open-ended complaints, unclear messages) -- unconditional,
       regardless of confidence or evidence quality.
    2. Threshold-based rules (low intent confidence, too little evidence,
       weak best-match similarity, conflicting historical guidance) -- any
       number can fire together; all reasons are reported.
    """
    intent_def = next((i for i in taxonomy["intents"] if i["name"] == intent), None)

    if intent_def is not None and intent_def.get("always_escalate"):
        reason = (intent_def.get("escalation_reason") or f"Intent '{intent}' is always escalated per taxonomy policy.").strip()
        return EscalationDecision(ESCALATE_TO_HUMAN, reason, ["always_escalate_intent"])

    triggered: list[str] = []
    if intent_confidence < config.min_intent_confidence:
        triggered.append("low_intent_confidence")
    if len(evidence) < config.min_evidence_count:
        triggered.append("insufficient_evidence_count")
    max_similarity = max((e.similarity for e in evidence), default=0.0)
    if max_similarity < config.min_evidence_similarity:
        triggered.append("low_evidence_similarity")
    if has_conflicting_evidence(evidence):
        triggered.append("conflicting_evidence")

    if not triggered:
        return EscalationDecision(
            AUTO_HANDLE,
            "Intent classified with sufficient confidence and grounded in "
            "consistent historical evidence.",
            [],
        )

    reason_text = {
        "low_intent_confidence": (
            f"Intent classification confidence ({intent_confidence:.2f}) is "
            f"below the auto-handle threshold ({config.min_intent_confidence})."
        ),
        "insufficient_evidence_count": (
            f"Only {len(evidence)} historical resolution(s) were retrieved "
            f"(minimum {config.min_evidence_count} required)."
        ),
        "low_evidence_similarity": (
            f"The best-matching historical case has similarity "
            f"{max_similarity:.2f}, below the threshold "
            f"({config.min_evidence_similarity})."
        ),
        "conflicting_evidence": (
            "The retrieved historical resolutions suggest conflicting "
            "actions (e.g. refund vs. replacement) for similar issues."
        ),
    }
    reason = " ".join(reason_text[rule] for rule in triggered)
    return EscalationDecision(ESCALATE_TO_HUMAN, reason, triggered)
