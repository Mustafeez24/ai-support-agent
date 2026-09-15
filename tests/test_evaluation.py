"""Tests for src/evaluation/{metrics,judge,agreement}.py.

All synthetic data. judge.py is tested only on response *parsing* (no
real NVIDIA calls, consistent with the rest of this suite); agreement.py
is tested against fabricated human/judge rating pairs constructed to have
known, verifiable agreement levels (perfect agreement, plausible partial
agreement, chance-level agreement) -- these are NOT presented anywhere as
real evaluation results, only as unit-test fixtures proving the math is
correct.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import agreement, metrics
from src.evaluation.judge import JudgeParseError, JudgeScore, parse_judge_response

# --- metrics.py: intent_metrics -----------------------------------------------


def test_intent_metrics_perfect_predictions():
    y_true = ["a", "b", "a", "c"]
    y_pred = ["a", "b", "a", "c"]
    result = metrics.intent_metrics(y_true, y_pred)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["per_intent"]["a"]["f1"] == 1.0


def test_intent_metrics_reports_per_intent_and_confusion_matrix():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    result = metrics.intent_metrics(y_true, y_pred, labels=["a", "b"])
    assert result["per_intent"]["a"]["support"] == 2
    assert result["per_intent"]["b"]["support"] == 2
    assert result["confusion_matrix"]["labels"] == ["a", "b"]
    assert result["accuracy"] == 0.75


def test_intent_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics.intent_metrics(["a"], ["a", "b"])


def test_intent_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        metrics.intent_metrics([], [])


# --- metrics.py: escalation_metrics -------------------------------------------


def test_escalation_metrics_perfect_predictions():
    y_true = [True, False, True, False]
    y_pred = [True, False, True, False]
    result = metrics.escalation_metrics(y_true, y_pred)
    assert result["accuracy"] == 1.0
    assert result["false_auto_handle_count"] == 0
    assert result["false_auto_handle_rate"] == 0.0


def test_escalation_metrics_false_auto_handle_rate():
    # 2 examples SHOULD escalate (True); one of them was wrongly auto-handled.
    y_true = [True, True, False]
    y_pred = [True, False, False]
    result = metrics.escalation_metrics(y_true, y_pred)
    assert result["should_escalate_count"] == 2
    assert result["false_auto_handle_count"] == 1
    assert result["false_auto_handle_rate"] == 0.5


def test_escalation_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        metrics.escalation_metrics([], [])


# --- metrics.py: retrieval_intent_match_at_k -----------------------------------


def test_retrieval_intent_match_at_k_basic():
    true_intents = ["delivery_delay", "damaged_or_defective_item"]
    retrieved = [
        ["damaged_or_defective_item", "delivery_delay", "other_unclear"],
        ["damaged_or_defective_item", "other_unclear", "other_unclear"],
    ]
    result = metrics.retrieval_intent_match_at_k(true_intents, retrieved, k_values=(1, 2, 3))
    assert result["recall_at_1"] == 0.5  # only query 2 has an exact top-1 match
    assert result["recall_at_2"] == 1.0  # query 1's match appears within top 2
    assert result["recall_at_3"] == 1.0


def test_retrieval_intent_match_at_k_empty_input():
    result = metrics.retrieval_intent_match_at_k([], [])
    assert result["n_examples"] == 0


# --- judge.py: response parsing -----------------------------------------------


def test_parse_judge_response_plain_json():
    raw = (
        '{"relevance": 5, "groundedness": 4, "correctness": 5, '
        '"helpfulness": 4, "tone": 5, "no_hallucination": true, '
        '"rationale": "Accurate and on-topic."}'
    )
    score = parse_judge_response(raw)
    assert isinstance(score, JudgeScore)
    assert score.relevance == 5
    assert score.no_hallucination is True


def test_parse_judge_response_handles_markdown_fence():
    raw = (
        "Here is my evaluation:\n```json\n"
        '{"relevance": 3, "groundedness": 2, "correctness": 3, '
        '"helpfulness": 3, "tone": 4, "no_hallucination": false, '
        '"rationale": "Invents a refund amount not in evidence."}\n```'
    )
    score = parse_judge_response(raw)
    assert score.groundedness == 2
    assert score.no_hallucination is False


def test_parse_judge_response_missing_field_raises():
    raw = '{"relevance": 5, "groundedness": 4}'
    with pytest.raises(JudgeParseError):
        parse_judge_response(raw)


def test_parse_judge_response_invalid_json_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response("this is not json at all")


# --- agreement.py --------------------------------------------------------------


def _rubric_row(id_, relevance, groundedness, correctness, helpfulness, tone, no_hallucination):
    return {
        "id": id_,
        "relevance": relevance,
        "groundedness": groundedness,
        "correctness": correctness,
        "helpfulness": helpfulness,
        "tone": tone,
        "no_hallucination": no_hallucination,
    }


def test_compute_agreement_report_perfect_agreement():
    rows = [
        _rubric_row("g0001", 5, 5, 5, 5, 5, True),
        _rubric_row("g0002", 3, 2, 4, 3, 3, False),
        _rubric_row("g0003", 1, 1, 2, 1, 2, False),
        _rubric_row("g0004", 4, 4, 4, 5, 4, True),
    ]
    human_df = pd.DataFrame(rows)
    judge_df = pd.DataFrame(rows)  # identical -> perfect agreement
    report = agreement.compute_agreement_report(human_df, judge_df)
    assert report["n_examples"] == 4
    assert report["relevance"]["weighted_kappa"] == pytest.approx(1.0)
    assert report["no_hallucination"]["agreement_rate"] == 1.0
    assert report["overall_mean_weighted_kappa"] == pytest.approx(1.0)
    assert report["interpretation"] == "almost perfect"


def test_compute_agreement_report_partial_agreement():
    human_rows = [
        _rubric_row("g0001", 5, 5, 5, 5, 5, True),
        _rubric_row("g0002", 3, 2, 4, 3, 3, False),
        _rubric_row("g0003", 1, 1, 2, 1, 2, False),
        _rubric_row("g0004", 4, 4, 4, 5, 4, True),
    ]
    judge_rows = [
        _rubric_row("g0001", 4, 5, 5, 4, 5, True),  # close but not identical
        _rubric_row("g0002", 3, 3, 3, 3, 4, False),
        _rubric_row("g0003", 2, 1, 2, 2, 2, True),  # judge missed a hallucination
        _rubric_row("g0004", 4, 4, 5, 5, 4, True),
    ]
    human_df = pd.DataFrame(human_rows)
    judge_df = pd.DataFrame(judge_rows)
    report = agreement.compute_agreement_report(human_df, judge_df)
    assert 0.0 <= report["overall_mean_weighted_kappa"] <= 1.0
    assert report["no_hallucination"]["agreement_rate"] == 0.75  # 3 of 4 match


def test_compute_agreement_report_missing_columns_raises():
    human_df = pd.DataFrame([{"id": "g0001", "relevance": 5}])
    judge_df = pd.DataFrame([_rubric_row("g0001", 5, 5, 5, 5, 5, True)])
    with pytest.raises(ValueError):
        agreement.compute_agreement_report(human_df, judge_df)


def test_compute_agreement_report_no_overlapping_ids_raises():
    human_df = pd.DataFrame([_rubric_row("g0001", 5, 5, 5, 5, 5, True)])
    judge_df = pd.DataFrame([_rubric_row("different_id", 5, 5, 5, 5, 5, True)])
    with pytest.raises(ValueError):
        agreement.compute_agreement_report(human_df, judge_df)


def test_pearson_r_returns_none_without_variance():
    assert agreement.pearson_r([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]) is None


# --- evaluate.py ----------------------------------------------------------------


from src.evaluation import evaluate as evaluate_mod


class _FakeResult:
    def __init__(self, intent, decision, reply, evidence):
        self.intent = intent
        self.decision = decision
        self.reply = reply
        self.evidence = evidence


class _FakeSystemWithHandle:
    """Mimics SupportAgent's .handle() interface."""

    def __init__(self, results: list[_FakeResult]):
        self._results = iter(results)

    def handle(self, text):
        return next(self._results)


class _FakeSystemWithPredict:
    """Mimics a baseline's .predict() interface."""

    def __init__(self, results: list[_FakeResult]):
        self._results = iter(results)

    def predict(self, text):
        return next(self._results)


def test_parse_expected_escalation_maps_string_true_false():
    series = pd.Series(["True", "False", "True"])
    parsed = evaluate_mod.parse_expected_escalation(series)
    assert parsed.tolist() == [True, False, True]


def test_parse_expected_escalation_rejects_unparseable_values():
    series = pd.Series(["True", "maybe"])
    with pytest.raises(ValueError):
        evaluate_mod.parse_expected_escalation(series)


def test_run_system_over_golden_set_with_handle_interface():
    results = [
        _FakeResult("delivery_delay", "AUTO_HANDLE", "reply1", [{"customer_text": "a", "brand_text": "b"}]),
        _FakeResult("damaged_or_defective_item", "ESCALATE_TO_HUMAN", "fallback", []),
    ]
    system = _FakeSystemWithHandle(results)
    preds = evaluate_mod.run_system_over_golden_set(
        system, ["msg1", "msg2"], name="production", evidence_intent_fn=lambda ev: ["delivery_delay"] if ev else []
    )
    assert preds.intents == ["delivery_delay", "damaged_or_defective_item"]
    assert preds.escalations == [False, True]
    assert preds.evidence_intents == [["delivery_delay"], []]


def test_run_system_over_golden_set_with_predict_interface():
    results = [_FakeResult("delivery_delay", "ESCALATE_TO_HUMAN", "fallback", [])]
    system = _FakeSystemWithPredict(results)
    preds = evaluate_mod.run_system_over_golden_set(system, ["msg1"], name="baseline1")
    assert preds.name == "baseline1"
    assert preds.escalations == [True]


def test_evaluate_system_produces_intent_and_escalation_reports():
    preds = evaluate_mod.SystemPredictions(
        name="test_system",
        intents=["delivery_delay", "damaged_or_defective_item"],
        escalations=[False, True],
        replies=["r1", "r2"],
        evidence_blocks=["", ""],
        evidence_intents=[[], []],
    )
    report = evaluate_mod.evaluate_system(
        preds, true_intents=["delivery_delay", "damaged_or_defective_item"], true_escalations=[False, True]
    )
    assert report["system"] == "test_system"
    assert report["intent"]["accuracy"] == 1.0
    assert report["escalation"]["accuracy"] == 1.0
    assert "retrieval" not in report  # no evidence_intents provided


def test_judge_auto_handled_replies_skips_escalated_examples():
    class _FakeJudgeLLM:
        def is_configured(self):
            return True

        def generate(self, messages, **kwargs):
            from src.llm.base import LLMResponse

            content = (
                '{"relevance": 5, "groundedness": 5, "correctness": 5, '
                '"helpfulness": 5, "tone": 5, "no_hallucination": true, "rationale": "ok"}'
            )
            return LLMResponse(content=content, model="fake")

    preds = evaluate_mod.SystemPredictions(
        name="production",
        intents=["delivery_delay", "damaged_or_defective_item"],
        escalations=[False, True],  # second one escalated -> should be skipped
        replies=["We're sorry, refund issued.", "fallback"],
        evidence_blocks=["evidence1", "evidence2"],
        evidence_intents=[[], []],
    )
    results = evaluate_mod.judge_auto_handled_replies(
        _FakeJudgeLLM(), ids=["g1", "g2"], texts=["msg1", "msg2"], preds=preds, brand="AmazonHelp"
    )
    assert len(results) == 1
    assert results[0]["id"] == "g1"
    assert results[0]["relevance"] == 5


def test_build_comparison_table_has_expected_columns():
    reports = [
        {
            "system": "production",
            "intent": {"accuracy": 0.8, "macro_f1": 0.75},
            "escalation": {"accuracy": 0.9, "recall_escalate": 0.85, "false_auto_handle_rate": 0.1},
            "retrieval": {"recall_at_5": 0.7},
        },
        {
            "system": "baseline1_trivial",
            "intent": {"accuracy": 0.2, "macro_f1": 0.05},
            "escalation": {"accuracy": 0.3, "recall_escalate": 1.0, "false_auto_handle_rate": 0.0},
        },
    ]
    table = evaluate_mod.build_comparison_table(reports)
    assert list(table["system"]) == ["production", "baseline1_trivial"]
    assert "retrieval_recall_at_5" in table.columns
    assert pd.isna(table.loc[1, "retrieval_recall_at_5"])
