"""LLM-as-judge: scores a generated reply against a fixed rubric.

Rubric dimensions (assignment section I):
    relevance, groundedness, correctness, helpfulness, tone  -- each 1-5
    no_hallucination                                          -- boolean

The judge's scores are NOT ground truth. Phase 9's `agreement.py` measures
how well they agree with real human ratings on a small subset -- that
agreement number, not the judge's raw scores, is what makes the reply
quality metric trustworthy (or not).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from src.llm.base import LLMMessage, LLMProvider

RUBRIC_SCORE_DIMENSIONS = ("relevance", "groundedness", "correctness", "helpfulness", "tone")
RUBRIC_BOOLEAN_DIMENSIONS = ("no_hallucination",)
ALL_RUBRIC_DIMENSIONS = RUBRIC_SCORE_DIMENSIONS + RUBRIC_BOOLEAN_DIMENSIONS

JUDGE_SYSTEM_PROMPT = """You are an expert QA reviewer grading a customer support reply for {brand}.

You will be given the customer's message, the historical resolutions the \
reply was supposed to be grounded in, and the reply itself. Score the \
reply on each dimension:

- relevance (1-5): does the reply address the customer's actual issue?
- groundedness (1-5): is the reply's content supported by the historical \
resolutions provided (not invented)?
- correctness (1-5): given the evidence, is the reply factually/procedurally \
sound (no contradictions, no wrong information)?
- helpfulness (1-5): would this reply actually help the customer make \
progress on their issue?
- tone (1-5): is the tone professional, empathetic, and appropriate?
- no_hallucination (true/false): true only if the reply contains NO \
claims, promises, prices, dates, or actions beyond what the historical \
resolutions support.

Respond with ONLY a single JSON object, no other text, no markdown fences:
{{"relevance": <1-5>, "groundedness": <1-5>, "correctness": <1-5>, \
"helpfulness": <1-5>, "tone": <1-5>, "no_hallucination": <true|false>, \
"rationale": "<one sentence>"}}
"""


@dataclass
class JudgeScore:
    relevance: int
    groundedness: int
    correctness: int
    helpfulness: int
    tone: int
    no_hallucination: bool
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


class JudgeParseError(ValueError):
    """Raised when the judge's response can't be parsed into a JudgeScore."""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    candidate = fence_match.group(1) if fence_match else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: grab the first {...} span.
        brace_match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))
        raise


def build_judge_prompt(
    customer_message: str, evidence_block: str, reply: str, brand: str
) -> list[LLMMessage]:
    system = LLMMessage(role="system", content=JUDGE_SYSTEM_PROMPT.format(brand=brand))
    user = LLMMessage(
        role="user",
        content=(
            f'Customer message:\n"{customer_message}"\n\n'
            f"Historical resolutions provided to the reply-writer:\n{evidence_block}\n\n"
            f'Reply to grade:\n"{reply}"'
        ),
    )
    return [system, user]


def parse_judge_response(raw_content: str) -> JudgeScore:
    try:
        data = _extract_json(raw_content)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"Judge response was not valid JSON: {raw_content!r}") from exc

    missing = [d for d in ALL_RUBRIC_DIMENSIONS if d not in data]
    if missing:
        raise JudgeParseError(f"Judge response missing fields {missing}: {data!r}")

    try:
        return JudgeScore(
            relevance=int(data["relevance"]),
            groundedness=int(data["groundedness"]),
            correctness=int(data["correctness"]),
            helpfulness=int(data["helpfulness"]),
            tone=int(data["tone"]),
            no_hallucination=bool(data["no_hallucination"]),
            rationale=str(data.get("rationale", "")),
        )
    except (TypeError, ValueError) as exc:
        raise JudgeParseError(f"Judge response had malformed fields: {data!r}") from exc


def judge_reply(
    llm: LLMProvider,
    customer_message: str,
    evidence_block: str,
    reply: str,
    brand: str,
    model: str | None = None,
) -> JudgeScore:
    messages = build_judge_prompt(customer_message, evidence_block, reply, brand)
    response = llm.generate(messages, temperature=0.0, max_tokens=250, model=model)
    return parse_judge_response(response.content)
