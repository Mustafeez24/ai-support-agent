"""Two required baselines the full agent is compared against (Phase 8/9).

BASELINE 1 (trivial): majority-intent classifier + a fixed generic reply +
an always-escalate policy. The floor -- any real system must beat doing
none of the actual work.

BASELINE 2 (simple, non-LLM): the SAME TF-IDF+LogisticRegression intent
classifier used in production (see DECISIONS.md #11 for why that's one
model, not two), TF-IDF nearest-neighbor retrieval (src/retrieval/
embeddings.py's TfidfEmbedder, NOT the dense sentence-transformer model)
whose top-1 historical brand reply is returned VERBATIM with no
generation step, and the SAME rule/threshold-based escalation policy used
in production (src/agent/escalation.py was already non-LLM, so reusing it
here is not "cheating" -- it isolates exactly what the LLM adds: grounded
*generation* instead of copying the nearest historical reply verbatim).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from src.agent.escalation import AUTO_HANDLE, ESCALATE_TO_HUMAN, decide_escalation
from src.config import EscalationConfig
from src.retrieval.retriever import Retriever

GENERIC_FALLBACK_REPLY = (
    "Thanks for contacting us. A member of our support team will follow up "
    "with you shortly."
)


@dataclass
class BaselineResult:
    intent: str
    intent_confidence: float
    reply: str
    decision: str
    escalation_reason: str
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class TrivialBaseline:
    """Baseline 1: majority-intent classifier + generic fallback reply +
    always-escalate policy."""

    def __init__(self, majority_intent: str):
        self.majority_intent = majority_intent

    @classmethod
    def fit(cls, labels: list[str]) -> "TrivialBaseline":
        if not labels:
            raise ValueError("Cannot fit TrivialBaseline on an empty label list.")
        majority = Counter(labels).most_common(1)[0][0]
        return cls(majority)

    def predict(self, message: str) -> BaselineResult:
        return BaselineResult(
            intent=self.majority_intent,
            intent_confidence=1.0,  # trivially "confident": always predicts the same label
            reply=GENERIC_FALLBACK_REPLY,
            decision=ESCALATE_TO_HUMAN,
            escalation_reason="Trivial baseline policy: always escalate to a human.",
            evidence=[],
        )


class TfidfNearestNeighborBaseline:
    """Baseline 2: TF-IDF classification + TF-IDF nearest-neighbor
    retrieval + verbatim top-1 reply + rule-based escalation. No LLM call
    anywhere in this class."""

    def __init__(self, classifier, retriever: Retriever, taxonomy: dict, escalation_config: EscalationConfig):
        self.classifier = classifier
        self.retriever = retriever
        self.taxonomy = taxonomy
        self.escalation_config = escalation_config

    def predict(self, message: str) -> BaselineResult:
        prediction = self.classifier.predict_one(message)
        evidence = self.retriever.search(message)
        decision = decide_escalation(
            intent=prediction.intent,
            intent_confidence=prediction.confidence,
            evidence=evidence,
            taxonomy=self.taxonomy,
            config=self.escalation_config,
        )

        if decision.decision == AUTO_HANDLE and evidence:
            reply = evidence[0].brand_text  # verbatim nearest-neighbor match, no generation
        else:
            reply = GENERIC_FALLBACK_REPLY

        return BaselineResult(
            intent=prediction.intent,
            intent_confidence=prediction.confidence,
            reply=reply,
            decision=decision.decision,
            escalation_reason=decision.reason,
            evidence=[e.to_dict() for e in evidence],
        )
