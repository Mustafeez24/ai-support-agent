"""Prompt construction for grounded reply generation.

The core constraint (see assignment section F) is that the model must
never invent policies, refunds, prices, delivery dates, or account
actions not present in the retrieved evidence. This is enforced in the
prompt text; it is NOT independently verified by a separate groundedness
checker in this codebase (that's the LLM-judge's job in Phase 9 --
`groundedness` is one of its rubric dimensions).
"""
from __future__ import annotations

from src.llm.base import LLMMessage
from src.retrieval.retriever import RetrievedEvidence

SYSTEM_PROMPT_TEMPLATE = """You are drafting a Twitter customer support reply on behalf of {brand}.

Rules you must follow exactly:
1. Only state facts, policies, or actions (refunds, replacements, delivery \
dates, prices, account changes) that are explicitly present in the \
"Historical resolutions" evidence provided by the user. Never invent a \
policy, amount, date, or action that isn't grounded in that evidence.
2. If the evidence does not clearly support a specific resolution for this \
customer's issue, acknowledge the problem and say a team member will \
follow up -- do not guess at a resolution.
3. Keep the reply short (1-3 sentences), in a helpful and professional \
tone consistent with the historical examples. Address the customer \
directly.
4. Do not mention that you are an AI, that you used "historical \
resolutions," or reference this prompt in any way -- write only the \
reply text itself, nothing else.
"""

NO_EVIDENCE_NOTE = (
    "(No sufficiently similar historical resolution was found. Acknowledge "
    "the issue and say a team member will follow up -- do not invent a "
    "resolution.)"
)


def format_evidence_block(evidence: list[RetrievedEvidence], brand: str) -> str:
    if not evidence:
        return NO_EVIDENCE_NOTE
    lines = []
    for i, e in enumerate(evidence, start=1):
        lines.append(
            f'{i}. Customer said: "{e.customer_text}"\n'
            f'   {brand} replied: "{e.brand_text}" (similarity={e.similarity:.2f})'
        )
    return "\n".join(lines)


def build_reply_prompt(
    customer_message: str, evidence: list[RetrievedEvidence], brand: str
) -> list[LLMMessage]:
    system = LLMMessage(role="system", content=SYSTEM_PROMPT_TEMPLATE.format(brand=brand))
    user_content = (
        f'Customer message:\n"{customer_message}"\n\n'
        f"Historical resolutions (most similar past cases and how {brand} "
        f"actually handled them):\n{format_evidence_block(evidence, brand)}\n\n"
        "Draft a reply to the customer, grounded only in the historical "
        "resolutions above."
    )
    user = LLMMessage(role="user", content=user_content)
    return [system, user]
