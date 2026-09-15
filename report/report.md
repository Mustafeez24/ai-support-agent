# AI Support Agent for AmazonHelp — Report

**Status of this report:** the full pipeline (data processing, intent
taxonomy tooling, retrieval, agent, baselines, evaluation, LLM judge) is
built, unit-tested (89/89 passing at time of writing), and wired
end-to-end — verified with a synthetic fixture standing in for the real
data. **The real 517MB dataset (`data/raw/twcs.csv`) was not available in
the environment that wrote this code**, so dataset-derived numbers below
are explicitly marked **PENDING**, with the exact command that produces
them. Nothing in this report is fabricated; see `DECISIONS.md` for every
design choice and its rationale.

## 1. Executive Summary

This project builds a support agent for **AmazonHelp** (the assignment's
suggested brand — see Section 3) that classifies incoming customer
messages into one of 11 discovered intents, retrieves how AmazonHelp
historically resolved similar issues, drafts a reply grounded in that
evidence, and decides whether to auto-handle or escalate to a human, with
a stated reason. The system is compared against two required baselines —
a trivial majority-class/always-escalate baseline, and a non-LLM TF-IDF
classification + nearest-neighbor-reply baseline that otherwise shares the
production classifier and escalation policy (see Decision #23) — so the
comparison isolates what dense retrieval and LLM generation specifically
add. Headline accuracy/F1/kappa numbers are **PENDING** a real run against
`data/raw/twcs.csv`; see Section 10.

## 2. Problem Framing

Support teams answer the same handful of issue types thousands of times.
An agent that (a) knows which of a *small, real* set of issues a new
message represents, (b) can point to how the brand actually resolved
similar issues before, and (c) knows when *not* to act autonomously, is
more trustworthy than a general-purpose chatbot with no memory of the
brand's real behavior and no off-ramp to a human. The core engineering
challenge is not generating plausible-sounding replies — it's **proving**
the system's classifications, groundedness, and escalation judgment are
correct enough to trust, which is why this project treats evaluation
methodology (Section 9) as seriously as the agent itself.

## 3. Dataset and Brand Selection

The dataset is the Kaggle *Customer Support on Twitter* corpus
(`twcs.csv`; columns: `tweet_id`, `author_id`, `inbound`, `created_at`,
`text`, `response_tweet_id`, `in_response_to_tweet_id`). `src/data/`
reconstructs conversations as connected components of the reply graph
(Decision #2), splits them **at the thread level** into a
`train_retrieval` corpus and a held-out `golden_eval` pool (Decision #3 —
this is the leakage guard the whole evaluation depends on), and extracts
direct customer→brand "resolution pairs" as the unit of historical
evidence (Decision #4).

**Brand:** AmazonHelp, per the assignment's own guidance (~169,840
messages in the full dataset, comfortably above the volume this task
needs). `scripts/analyze_dataset.py` implements the full scored
candidate-brand comparison methodology (Decision #6) and would confirm
this against real statistics — **PENDING**:
```
python scripts/analyze_dataset.py
```
writes `outputs/analysis/{dataset_summary.json, brand_statistics.csv,
candidate_brands.md}` with the real row counts, per-brand volume/
conversation/resolution-pair statistics, and the top-10 scored ranking.

## 4. Intent Taxonomy

`config/intents.yaml` defines 11 intents (delivery_delay,
missing_or_lost_package, damaged_or_defective_item,
return_or_refund_request, order_cancellation_or_change,
account_access_issue, payment_or_billing_issue,
product_or_device_technical_support, subscription_or_membership_issue,
general_complaint_or_feedback, other_unclear), each with a description,
2+ examples, inclusion/exclusion criteria, and an `always_escalate` flag.
It ships marked `status: DRAFT_PENDING_DATA_VALIDATION` (Decision #10):
seeded from domain knowledge of what `@AmazonHelp` actually handles, not
from Banking77 or any generic set, but not yet checked against real
cluster evidence. `src/intents/discover.py` (TF-IDF + MiniBatchKMeans)
produces that evidence — **PENDING**:
```
python scripts/preprocess.py
python -m src.intents.discover
```
writes `outputs/analysis/intent_clusters_report.json`: per-cluster top
terms and real sample messages, to validate/revise the taxonomy against.

## 5. System Architecture

```
customer message
    -> intent classification   (src/intents/classifier.py: TF-IDF + LogisticRegression)
    -> retrieval                (src/retrieval/retriever.py: FAISS over sentence-transformer embeddings)
    -> escalation decision      (src/agent/escalation.py: deterministic rule policy)
    -> reply generation         (src/agent/prompts.py + src/llm/nvidia.py, grounded in evidence,
                                  SKIPPED if the decision is already ESCALATE_TO_HUMAN)
    -> structured AgentResult   {intent, intent_confidence, reply, decision, escalation_reason, evidence}
```
The LLM (NVIDIA `nemotron-3-ultra-550b-a55b`, OpenAI-compatible endpoint,
behind a provider-agnostic `LLMProvider` interface — Decision #17) is
called **only** for reply generation and evaluation judging, never for
classification or the ~2.8M-row dataset scan, per the assignment's
"no unnecessary API calls" instruction.

## 6. Retrieval

`src/retrieval/embeddings.py`'s `SentenceTransformerEmbedder`
(`sentence-transformers/all-MiniLM-L6-v2`, local, free, no paid API) embeds
every `train_retrieval`-split customer message; `Retriever` (FAISS
`IndexFlatIP` over normalized vectors = cosine similarity, Decision #16)
indexes them and surfaces the **paired historical brand reply** as
grounding evidence for a new query (Decision #14). `Retriever.build`
hard-fails with `LeakageError` on any non-`train_retrieval` input unless
explicitly overridden (Decision #15) — the golden set's resolutions can
never be retrievable evidence for grading that same set.

## 7. Agent Design

`SupportAgent.handle` (`src/agent/agent.py`) runs the pipeline in Section
5. Reply generation uses a prompt (`src/agent/prompts.py`) that
instructs the model to state only facts/policies/actions present in the
retrieved evidence and never invent refunds, prices, dates, or account
actions. If evidence is insufficient, escalation fires **before**
generation is even attempted (no wasted LLM call); if generation itself
fails (missing API key, provider error, empty response), the decision is
flipped to `ESCALATE_TO_HUMAN` with a fixed fallback reply rather than
ever returning a broken response (Decision #21) — this is a fail-safe,
not fail-open, design.

## 8. Escalation Policy

`src/agent/escalation.py`'s `decide_escalation` applies, in order:
1. **Taxonomy always-escalate intents** (account_access_issue,
   payment_or_billing_issue, general_complaint_or_feedback,
   other_unclear) — unconditional.
2. **Threshold rules**, any combination of which can fire together, each
   contributing to a human-readable reason:
   - `low_intent_confidence` (< `ESCALATION_MIN_INTENT_CONFIDENCE`, default 0.55)
   - `insufficient_evidence_count` (< `ESCALATION_MIN_EVIDENCE_COUNT`, default 2)
   - `low_evidence_similarity` (best match < `ESCALATION_MIN_EVIDENCE_SIMILARITY`, default 0.35)
   - `conflicting_evidence` (top historical replies suggest mutually
     exclusive actions, e.g. refund vs. replacement — a keyword-bucket
     heuristic, Decision #22)

All thresholds live in `src/config.py` / `.env`, not scattered through the
code (Decision #8), so they can be re-tuned once Section 10's real numbers
exist without touching agent logic.

## 9. Evaluation Methodology

**Golden set** (`data/golden/`): ~200 examples sampled from the held-out
`golden_eval` thread split, stratified by an unsupervised clustering of
that pool (not the taxonomy, to avoid circular bias — Decision #12), hand-
labeled via an interactive CLI (`scripts/build_golden_set.py`) that saves
after every example and requires a real person (Decision #13). Full
methodology, representativeness rationale, and ambiguous-case handling:
`data/golden/METHODOLOGY.md`.

**Metrics** (`src/evaluation/metrics.py`): intent accuracy/macro-F1/
per-intent P-R-F1/confusion matrix; escalation accuracy/precision/recall
and — the safety-critical number — **false auto-handle rate** (fraction of
examples a human said should escalate that the system auto-handled
instead); retrieval "intent-match@K", an explicitly-labeled proxy for
Recall@K since no hand-labeled per-item relevance judgments exist
(Decision #24).

**Reply quality**: an LLM judge (`src/evaluation/judge.py`) scores every
**auto-handled** reply (escalations produce identical fallback text, not
worth judging — Decision #25) on relevance, groundedness, correctness,
helpfulness, tone (1-5) and no_hallucination (bool). Judge scores are
**not** trusted at face value: `src/evaluation/agreement.py` computes
weighted Cohen's kappa / Pearson correlation between the judge and a real
human rating a sample of the same replies (`scripts/evaluate.py
human-review-template` → a person fills it in → `scripts/evaluate.py
agreement`). This agreement number, not the judge's raw scores, is what
makes the reply-quality metric trustworthy (Decision #26).

**Baselines** (Section 10): Baseline 1 is trivial (majority intent +
generic reply + always escalate). Baseline 2 shares the production
classifier and escalation policy but uses TF-IDF retrieval and a verbatim
top-1 historical reply instead of dense retrieval + LLM generation
(Decision #23) — deliberately, so the comparison isolates exactly what
dense retrieval and generation add over a strong non-LLM system, not just
"our system beats a strawman."

## 10. Results vs. Baselines

**PENDING** — requires the real dataset and a completed golden-set
labeling session. Once both exist:
```
python scripts/preprocess.py
python scripts/train_classifier.py
python scripts/build_index.py
python scripts/build_golden_set.py sample
python scripts/build_golden_set.py label      # interactive, real human
python scripts/build_golden_set.py report
python scripts/evaluate.py run
python scripts/evaluate.py human-review-template
# a human fills in outputs/metrics/human_review_template.csv,
# saves as outputs/metrics/human_review_ratings.csv
python scripts/evaluate.py agreement
```
produces `outputs/metrics/comparison_table.csv` (below, columns filled in
once real) and `outputs/metrics/judge_agreement_report.json`:

| system | intent_accuracy | intent_macro_f1 | escalation_accuracy | escalation_recall | false_auto_handle_rate | retrieval_recall_at_5 |
|---|---|---|---|---|---|---|
| production | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| baseline1_trivial | PENDING | PENDING | PENDING | PENDING | PENDING | — |
| baseline2_tfidf | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

No number in this table will be hand-edited into this file without the
corresponding `outputs/metrics/` artifact existing on disk.

## 11. Failure Analysis

**PENDING real examples** — this section is generated from
`outputs/metrics/evaluation_report.json` and `outputs/metrics/
judge_scores.csv` once Section 10 has run, then manually reviewed. The
methodology: sort golden examples by (a) intent misclassification, (b)
false-auto-handle escalation errors, (c) lowest judge groundedness score,
and (d) largest human-vs-judge disagreement; select the top 5 distinct
failure modes across those axes (not 5 examples of the same failure), and
for each report: the example, expected result, actual result, a
hypothesis for why, and a possible fix. No failure mode is invented ahead
of that run.

## 12. What is misleading about my headline number?

Even once Section 10 has real numbers, treat them with these limitations
in mind:

- **Small golden set (~200 examples).** Per-intent metrics for any intent
  with few golden examples (plausible for `account_access_issue` or
  `payment_or_billing_issue` if they're genuinely rare in the data) will
  have wide, unreported confidence intervals — a single misclassification
  can swing that intent's F1 by a large margin.
- **Sampling bias from cluster stratification.** The golden set is
  stratified by an unsupervised clustering of the eval pool (Decision
  #12), which improves coverage of rare topics but means the sample is
  *not* a simple random sample — headline "accuracy" is not a direct
  estimate of "accuracy on a random incoming message" without reweighting
  by true intent frequency.
- **Class imbalance.** Accuracy alone can look good by being right on the
  most common intent(s) alone; macro-F1 is reported specifically to
  surface this, but a system that's excellent at delivery_delay and poor
  at rare intents can still post a deceptively strong accuracy number.
- **Judge bias.** The LLM judge and the reply-generation LLM may share
  systematic biases (e.g. both trained similarly, both prefer confident-
  sounding text) that a human rater wouldn't share — measured, not
  assumed, via the agreement report, but a *positive* agreement number
  doesn't rule out both agreeing on a *shared* blind spot.
- **Retrieval leakage risk, mitigated but not provably zero.** Thread-
  level splitting (Decision #3) prevents a golden example's own resolution
  from being indexed, but two *different* threads about the same
  boilerplate issue (e.g. a mass shipping delay affecting many customers)
  can be near-duplicates across the split — retrieval "succeeding" there
  is not the same as generalizing to a genuinely novel issue.
- **Correlation between similar examples.** If the golden set contains
  several near-duplicate messages (common in real support data — many
  customers describe the same mass incident similarly), per-example
  metrics are not independent samples, inflating apparent precision on
  whatever pattern those duplicates share.
- **Historical support responses may themselves be imperfect.** Retrieval
  evidence is *actual* AmazonHelp replies, not verified-correct ones; a
  reply "grounded" in a historically weak resolution is grounded, not
  necessarily good (Decision #14's tradeoff).
- **Offline evaluation ≠ production performance.** No rate limits, no
  adversarial/ambiguous phrasing beyond what's in the golden set, no
  multi-turn follow-up handling, no real customer account context.
- **The LLM judge is not ground truth**, only a scalable proxy correlated
  (to whatever degree the agreement report shows) with human judgment on
  this specific rubric and this specific sample size.

## 13. What I Would Do With One More Week

1. Run the full pipeline on real data, then **re-derive the intent
   taxonomy from real cluster evidence** (Section 4) rather than shipping
   a domain-knowledge draft — likely merging/splitting 2-3 intents.
2. Expand `config/intents.yaml`'s `examples` per intent from 2 to ~10 real
   messages (pulled from the discovery report) to fix the weak-label
   classifier's noisy bootstrap (Decision #20's documented limitation) —
   this alone probably moves the headline intent-accuracy number more than
   any other single change.
3. Replace the keyword-bucket `has_conflicting_evidence` heuristic
   (Decision #22) with a measured approach once real evidence exists to
   validate against — e.g. embedding-based reply clustering with a
   threshold tuned against the golden set's actual escalation labels.
4. Expand the human-vs-judge agreement sample past the default 40 and, if
   agreement is weak on any specific rubric dimension, revise the judge
   prompt for that dimension specifically and re-measure.
5. Add a second embedding model (e.g. a larger sentence-transformer) as an
   ablation to see whether retrieval quality is embedding-model-bound or
   corpus-bound.
6. Build a small held-out *second* golden slice reserved for threshold
   tuning (the current thresholds are principled defaults, not yet fit to
   real precision/recall tradeoffs on real escalation labels).

## 14. Limitations

- No real dataset was available while building this system; every
  dataset-derived number in this report is marked PENDING rather than
  estimated or fabricated.
- The intent taxonomy is a validated-pending draft (Section 4).
- The production classifier is trained on weak, taxonomy-seed-similarity
  labels, not human-verified ground truth (Decision #20) — its true
  accuracy is only known once Section 10 runs.
- `has_conflicting_evidence` is a simple keyword heuristic, not semantic
  analysis (Decision #22).
- Retrieval's relevance metric is a proxy (Decision #24), not true
  Recall@K.
- This system handles a single message in isolation; it does not model
  multi-turn follow-up beyond using historical multi-turn threads as
  training/retrieval evidence.
