"""The end-to-end agent pipeline:

    customer message
        -> intent classification (src/intents/classifier.py)
        -> retrieval (src/retrieval/retriever.py)
        -> escalation decision (src/agent/escalation.py)
        -> reply generation, grounded in evidence (src/agent/prompts.py + LLM),
           skipped entirely when the decision is already ESCALATE_TO_HUMAN
        -> structured AgentResult
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from src.agent.escalation import AUTO_HANDLE, ESCALATE_TO_HUMAN, decide_escalation
from src.agent.prompts import build_reply_prompt
from src.config import EscalationConfig
from src.intents.classifier import IntentClassifier
from src.llm.base import LLMNotConfiguredError, LLMProvider, LLMProviderError
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

FALLBACK_REPLY = (
    "Thanks for reaching out -- we've flagged this for a member of our "
    "team, who will follow up with you shortly to help resolve it."
)


@dataclass
class AgentResult:
    intent: str
    intent_confidence: float
    reply: str
    decision: str
    escalation_reason: str
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class SupportAgent:
    def __init__(
        self,
        classifier: IntentClassifier,
        retriever: Retriever,
        llm: LLMProvider,
        taxonomy: dict,
        escalation_config: EscalationConfig,
        brand: str,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.llm = llm
        self.taxonomy = taxonomy
        self.escalation_config = escalation_config
        self.brand = brand

    def handle(self, message: str) -> AgentResult:
        prediction = self.classifier.predict_one(message)
        evidence = self.retriever.search(message)

        decision = decide_escalation(
            intent=prediction.intent,
            intent_confidence=prediction.confidence,
            evidence=evidence,
            taxonomy=self.taxonomy,
            config=self.escalation_config,
        )

        if decision.decision == AUTO_HANDLE:
            try:
                reply = self._generate_reply(message, evidence)
            except (LLMNotConfiguredError, LLMProviderError) as exc:
                logger.warning("Reply generation failed, escalating instead: %s", exc)
                decision.decision = ESCALATE_TO_HUMAN
                decision.reason = (
                    f"{decision.reason} Reply generation failed and could not "
                    f"be safely auto-sent: {exc}"
                ).strip()
                decision.triggered_rules.append("llm_generation_failed")
                reply = FALLBACK_REPLY
        else:
            reply = FALLBACK_REPLY

        return AgentResult(
            intent=prediction.intent,
            intent_confidence=prediction.confidence,
            reply=reply,
            decision=decision.decision,
            escalation_reason=decision.reason,
            evidence=[e.to_dict() for e in evidence],
        )

    def _generate_reply(self, message: str, evidence) -> str:
        messages = build_reply_prompt(message, evidence, self.brand)
        response = self.llm.generate(messages)
        reply = response.content.strip()
        if not reply:
            raise LLMProviderError("LLM returned an empty reply.")
        return reply


def load_default_agent() -> SupportAgent:
    """Build a SupportAgent from the artifacts on disk: the taxonomy
    (config/intents.yaml), the trained classifier
    (outputs/cache/intent_classifier.joblib), and the FAISS retrieval index
    (outputs/cache/faiss_index/). Raises a clear FileNotFoundError naming
    the exact command to run if any prerequisite is missing.
    """
    from src import config as cfg
    from src.intents.classifier import IntentClassifier, load_taxonomy
    from src.llm.nvidia import NvidiaLLMProvider
    from src.retrieval.embeddings import SentenceTransformerEmbedder
    from src.retrieval.retriever import Retriever

    taxonomy_path = cfg.CONFIG_DIR / "intents.yaml"
    classifier_path = cfg.CACHE_DIR / "intent_classifier.joblib"

    if not classifier_path.exists():
        raise FileNotFoundError(
            f"{classifier_path} not found. Run: python scripts/train_classifier.py"
        )
    if not cfg.CONFIG.retrieval.index_path.exists():
        raise FileNotFoundError(
            f"{cfg.CONFIG.retrieval.index_path} not found. Run: python scripts/build_index.py"
        )

    taxonomy = load_taxonomy(taxonomy_path)
    classifier = IntentClassifier.load(classifier_path)
    embedder = SentenceTransformerEmbedder(cfg.CONFIG.retrieval.embedding_model)
    retriever = Retriever.load(
        embedder,
        cfg.CONFIG.retrieval.index_path,
        cfg.CONFIG.retrieval.metadata_path,
        top_k=cfg.CONFIG.retrieval.top_k,
    )
    llm = NvidiaLLMProvider(cfg.CONFIG.llm)

    return SupportAgent(
        classifier=classifier,
        retriever=retriever,
        llm=llm,
        taxonomy=taxonomy,
        escalation_config=cfg.CONFIG.escalation,
        brand=cfg.CONFIG.brand.author_id,
    )


def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run the AmazonHelp support agent on a single message."
    )
    parser.add_argument("--message", required=True, help="Customer message to handle.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a formatted summary.")
    args = parser.parse_args()

    agent = load_default_agent()
    result = agent.handle(args.message)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"Intent:     {result.intent} (confidence={result.intent_confidence:.2f})")
    print(f"Decision:   {result.decision}")
    print(f"Reason:     {result.escalation_reason}")
    print(f"Reply:      {result.reply}")
    print("Evidence:")
    if not result.evidence:
        print("  (none retrieved)")
    for i, e in enumerate(result.evidence, start=1):
        print(f"  {i}. [{e['similarity']:.2f}] customer: {e['customer_text']!r}")
        print(f"     -> reply: {e['brand_text']!r}")


if __name__ == "__main__":
    _cli()
