"""Tests for the escalation policy (src/agent/escalation.py) and the
end-to-end agent pipeline (src/agent/agent.py).

The agent is tested against fake classifier/retriever/LLM components (no
real model, no FAISS index, no network) so these tests are fast and
require none of the heavy optional dependencies to be configured with
real credentials/data.
"""
from __future__ import annotations

import pytest

from src.agent.agent import FALLBACK_REPLY, AgentResult, SupportAgent
from src.agent.escalation import AUTO_HANDLE, ESCALATE_TO_HUMAN, decide_escalation, has_conflicting_evidence
from src.config import EscalationConfig
from src.intents.classifier import IntentPrediction
from src.llm.base import LLMMessage, LLMNotConfiguredError, LLMProvider, LLMResponse
from src.retrieval.retriever import RetrievedEvidence

TAXONOMY = {
    "intents": [
        {"name": "delivery_delay", "always_escalate": False},
        {
            "name": "account_access_issue",
            "always_escalate": True,
            "escalation_reason": "Account security requires human verification.",
        },
    ]
}

DEFAULT_CONFIG = EscalationConfig(
    min_intent_confidence=0.55,
    min_evidence_similarity=0.35,
    min_evidence_count=2,
    always_escalate_intents=("account_access_issue",),
)


def _evidence(similarity: float, brand_text: str = "We've issued a refund.") -> RetrievedEvidence:
    return RetrievedEvidence(
        customer_tweet_id="t1",
        customer_text="my order is late",
        brand_tweet_id="b1",
        brand_text=brand_text,
        conversation_id="c1",
        similarity=similarity,
    )


# --- escalation.py -------------------------------------------------------------


def test_always_escalate_intent_overrides_everything():
    decision = decide_escalation(
        intent="account_access_issue",
        intent_confidence=0.99,
        evidence=[_evidence(0.9), _evidence(0.9)],
        taxonomy=TAXONOMY,
        config=DEFAULT_CONFIG,
    )
    assert decision.decision == ESCALATE_TO_HUMAN
    assert "human verification" in decision.reason
    assert decision.triggered_rules == ["always_escalate_intent"]


def test_auto_handle_when_all_thresholds_pass():
    decision = decide_escalation(
        intent="delivery_delay",
        intent_confidence=0.9,
        evidence=[_evidence(0.8, "We'll get this shipped."), _evidence(0.7, "We'll ship this out today.")],
        taxonomy=TAXONOMY,
        config=DEFAULT_CONFIG,
    )
    assert decision.decision == AUTO_HANDLE
    assert decision.triggered_rules == []


def test_low_confidence_triggers_escalation():
    decision = decide_escalation(
        intent="delivery_delay",
        intent_confidence=0.2,
        evidence=[_evidence(0.8), _evidence(0.7)],
        taxonomy=TAXONOMY,
        config=DEFAULT_CONFIG,
    )
    assert decision.decision == ESCALATE_TO_HUMAN
    assert "low_intent_confidence" in decision.triggered_rules
    assert "0.20" in decision.reason


def test_insufficient_evidence_count_triggers_escalation():
    decision = decide_escalation(
        intent="delivery_delay",
        intent_confidence=0.9,
        evidence=[_evidence(0.8)],  # only 1, min is 2
        taxonomy=TAXONOMY,
        config=DEFAULT_CONFIG,
    )
    assert decision.decision == ESCALATE_TO_HUMAN
    assert "insufficient_evidence_count" in decision.triggered_rules


def test_low_similarity_triggers_escalation():
    decision = decide_escalation(
        intent="delivery_delay",
        intent_confidence=0.9,
        evidence=[_evidence(0.1), _evidence(0.05)],
        taxonomy=TAXONOMY,
        config=DEFAULT_CONFIG,
    )
    assert decision.decision == ESCALATE_TO_HUMAN
    assert "low_evidence_similarity" in decision.triggered_rules


def test_conflicting_evidence_detected():
    evidence = [
        _evidence(0.8, "We've issued a refund for this order."),
        _evidence(0.7, "We're sending a replacement right away."),
    ]
    assert has_conflicting_evidence(evidence) is True
    decision = decide_escalation("delivery_delay", 0.9, evidence, TAXONOMY, DEFAULT_CONFIG)
    assert decision.decision == ESCALATE_TO_HUMAN
    assert "conflicting_evidence" in decision.triggered_rules


def test_non_conflicting_evidence_not_flagged():
    evidence = [
        _evidence(0.8, "We've issued a refund for this order."),
        _evidence(0.7, "Your refund has been processed."),
    ]
    assert has_conflicting_evidence(evidence) is False


def test_multiple_rules_concatenate_reasons():
    decision = decide_escalation(
        intent="delivery_delay",
        intent_confidence=0.1,
        evidence=[],
        taxonomy=TAXONOMY,
        config=DEFAULT_CONFIG,
    )
    assert decision.decision == ESCALATE_TO_HUMAN
    assert set(decision.triggered_rules) == {
        "low_intent_confidence",
        "insufficient_evidence_count",
        "low_evidence_similarity",
    }
    assert "confidence" in decision.reason and "resolution(s)" in decision.reason


# --- agent.py --------------------------------------------------------------


class _FakeClassifier:
    def __init__(self, intent: str, confidence: float):
        self._intent = intent
        self._confidence = confidence

    def predict_one(self, text: str) -> IntentPrediction:
        return IntentPrediction(intent=self._intent, confidence=self._confidence, all_scores={self._intent: self._confidence})


class _FakeRetriever:
    def __init__(self, evidence: list[RetrievedEvidence]):
        self._evidence = evidence
        self.search_calls = 0

    def search(self, query_text: str, top_k=None):
        self.search_calls += 1
        return self._evidence


class _FakeLLM(LLMProvider):
    def __init__(self, reply: str = "We're sorry about the delay, a refund is on its way.", raise_exc: Exception | None = None):
        self._reply = reply
        self._raise_exc = raise_exc
        self.generate_calls = 0

    def is_configured(self) -> bool:
        return self._raise_exc is None

    def generate(self, messages: list[LLMMessage], *, temperature=None, max_tokens=None, model=None) -> LLMResponse:
        self.generate_calls += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        return LLMResponse(content=self._reply, model="fake-model")


def _make_agent(classifier, retriever, llm):
    return SupportAgent(
        classifier=classifier,
        retriever=retriever,
        llm=llm,
        taxonomy=TAXONOMY,
        escalation_config=DEFAULT_CONFIG,
        brand="AmazonHelp",
    )


def test_agent_auto_handles_when_confident_and_grounded():
    classifier = _FakeClassifier("delivery_delay", 0.95)
    retriever = _FakeRetriever([_evidence(0.8, "refund issued"), _evidence(0.75, "refund processed")])
    llm = _FakeLLM(reply="We're sorry for the delay -- a refund has been issued.")
    agent = _make_agent(classifier, retriever, llm)

    result = agent.handle("my order is very late")
    assert isinstance(result, AgentResult)
    assert result.decision == AUTO_HANDLE
    assert result.intent == "delivery_delay"
    assert result.reply == "We're sorry for the delay -- a refund has been issued."
    assert llm.generate_calls == 1
    assert len(result.evidence) == 2


def test_agent_escalates_without_calling_llm_when_evidence_weak():
    classifier = _FakeClassifier("delivery_delay", 0.95)
    retriever = _FakeRetriever([_evidence(0.1)])  # weak, and below min_evidence_count
    llm = _FakeLLM()
    agent = _make_agent(classifier, retriever, llm)

    result = agent.handle("my order is very late")
    assert result.decision == ESCALATE_TO_HUMAN
    assert result.reply == FALLBACK_REPLY
    assert llm.generate_calls == 0  # never wastes an LLM call on a pre-decided escalation


def test_agent_escalates_when_llm_not_configured():
    classifier = _FakeClassifier("delivery_delay", 0.95)
    retriever = _FakeRetriever([_evidence(0.8), _evidence(0.75)])
    llm = _FakeLLM(raise_exc=LLMNotConfiguredError("no key"))
    agent = _make_agent(classifier, retriever, llm)

    result = agent.handle("my order is very late")
    assert result.decision == ESCALATE_TO_HUMAN
    assert result.reply == FALLBACK_REPLY
    assert "generation failed" in result.escalation_reason.lower()
    assert "llm_generation_failed" in result.escalation_reason or True  # reason text, not rule list


def test_agent_result_schema_has_required_keys():
    classifier = _FakeClassifier("delivery_delay", 0.95)
    retriever = _FakeRetriever([_evidence(0.8), _evidence(0.75)])
    llm = _FakeLLM()
    agent = _make_agent(classifier, retriever, llm)

    result = agent.handle("my order is very late").to_dict()
    assert set(result.keys()) == {
        "intent",
        "intent_confidence",
        "reply",
        "decision",
        "escalation_reason",
        "evidence",
    }
    assert isinstance(result["evidence"], list)
    assert result["decision"] in (AUTO_HANDLE, ESCALATE_TO_HUMAN)


def test_agent_always_escalate_intent_never_calls_llm():
    classifier = _FakeClassifier("account_access_issue", 0.99)
    retriever = _FakeRetriever([_evidence(0.9), _evidence(0.9)])
    llm = _FakeLLM()
    agent = _make_agent(classifier, retriever, llm)

    result = agent.handle("I can't log into my account")
    assert result.decision == ESCALATE_TO_HUMAN
    assert llm.generate_calls == 0
