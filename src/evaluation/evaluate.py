"""Orchestrates the Phase 9 evaluation: runs the production agent and both
baselines over the golden set, computes intent/escalation/retrieval
metrics per system, and -- if the LLM is configured -- judges
auto-handled replies against the quality rubric.

Human-vs-judge agreement is a deliberately separate, later step (see
agreement.py and scripts/evaluate.py's `human-review-template` /
`agreement` stages): it requires a real person rating a real subset and is
never computed automatically here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.agent.escalation import ESCALATE_TO_HUMAN
from src.evaluation import metrics as metrics_mod
from src.evaluation.judge import JudgeParseError, judge_reply
from src.llm.base import LLMNotConfiguredError, LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


def parse_expected_escalation(series: pd.Series) -> pd.Series:
    """golden_set.csv stores this as the literal strings 'True'/'False'
    (see scripts/build_golden_set.py) -- pandas' bool(str) would treat any
    non-empty string as True, so this is parsed explicitly rather than via
    `.astype(bool)`.
    """
    mapping = {"True": True, "False": False, True: True, False: False}
    parsed = series.map(mapping)
    if parsed.isna().any():
        bad = series[parsed.isna()].unique().tolist()
        raise ValueError(f"Unparseable expected_escalation values: {bad}")
    return parsed.astype(bool)


@dataclass
class SystemPredictions:
    name: str
    intents: list[str] = field(default_factory=list)
    escalations: list[bool] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    evidence_blocks: list[str] = field(default_factory=list)
    evidence_intents: list[list[str]] = field(default_factory=list)


def run_system_over_golden_set(
    system,
    texts: list[str],
    name: str,
    evidence_intent_fn=None,
) -> SystemPredictions:
    """`system` must have either `.handle(text)` (the real agent) or
    `.predict(text)` (a baseline), returning an object with
    `.intent`, `.decision`, `.reply`, `.evidence` (list of dicts with at
    least `customer_text`/`brand_text`).
    """
    call = system.handle if hasattr(system, "handle") else system.predict
    preds = SystemPredictions(name=name)
    for text in texts:
        result = call(text)
        preds.intents.append(result.intent)
        preds.escalations.append(result.decision == ESCALATE_TO_HUMAN)
        preds.replies.append(result.reply)
        evidence_text = "\n".join(
            f'- customer: "{e["customer_text"]}" -> reply: "{e["brand_text"]}"' for e in result.evidence
        ) or "(no evidence retrieved)"
        preds.evidence_blocks.append(evidence_text)
        preds.evidence_intents.append(evidence_intent_fn(result.evidence) if evidence_intent_fn else [])
    return preds


def evaluate_system(preds: SystemPredictions, true_intents: list[str], true_escalations: list[bool]) -> dict:
    report = {
        "system": preds.name,
        "intent": metrics_mod.intent_metrics(true_intents, preds.intents),
        "escalation": metrics_mod.escalation_metrics(true_escalations, preds.escalations),
    }
    if any(preds.evidence_intents):
        report["retrieval"] = metrics_mod.retrieval_intent_match_at_k(true_intents, preds.evidence_intents)
    return report


def judge_auto_handled_replies(
    llm: LLMProvider,
    ids: list[str],
    texts: list[str],
    preds: SystemPredictions,
    brand: str,
    model: str | None = None,
) -> list[dict]:
    """Judges only AUTO_HANDLE replies (escalations return a fixed fallback
    string not worth spending an LLM call to grade). Returns one dict per
    successfully-judged example; parse failures are logged and skipped,
    never silently substituted with a fabricated score.
    """
    results = []
    for id_, text, reply, escalated, evidence_block in zip(
        ids, texts, preds.replies, preds.escalations, preds.evidence_blocks
    ):
        if escalated:
            continue
        try:
            score = judge_reply(llm, text, evidence_block, reply, brand, model=model)
        except (JudgeParseError, LLMNotConfiguredError, LLMProviderError) as exc:
            logger.warning("Judge failed for id=%s: %s", id_, exc)
            continue
        results.append({"id": id_, "text": text, "reply": reply, **score.to_dict()})
    return results


def build_comparison_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        row = {
            "system": r["system"],
            "intent_accuracy": r["intent"]["accuracy"],
            "intent_macro_f1": r["intent"]["macro_f1"],
            "escalation_accuracy": r["escalation"]["accuracy"],
            "escalation_recall": r["escalation"]["recall_escalate"],
            "false_auto_handle_rate": r["escalation"]["false_auto_handle_rate"],
        }
        if "retrieval" in r:
            row["retrieval_recall_at_5"] = r["retrieval"].get("recall_at_5")
        rows.append(row)
    return pd.DataFrame(rows)
