# Golden Evaluation Set: Methodology

**Status: NOT YET CREATED.** This file documents the *process* by which the
golden set must be built. `golden_set_candidates.csv` and `golden_set.csv`
do not exist in this repository yet because they require the real
`data/raw/twcs.csv` (not available in the environment that built this
pipeline) and a real human labeling session (see "How labels were
assigned" below — this is explicitly not something the pipeline
auto-generates and calls "hand-labeled"). Once created, both files belong
in version control (they are small, and are a core deliverable) —
unlike the raw dataset and generated caches, they are NOT gitignored.

## How to produce it

```
python scripts/preprocess.py          # once, caches data/processed/*.parquet
python scripts/build_golden_set.py sample     # writes golden_set_candidates.csv
python scripts/build_golden_set.py label      # interactive: a human labels each one
python scripts/build_golden_set.py report     # writes label_distribution.json
```

## How examples were sampled

`scripts/build_golden_set.py sample`:

1. Starts from the `golden_eval` split of `{brand}_resolution_pairs.parquet`
   — i.e. only conversations that were held out of the retrieval/training
   corpus at the **thread level** (see `DECISIONS.md` #3). This guarantees
   no golden example's resolution is available to the retrieval system as
   evidence, so groundedness/retrieval metrics measured on it are
   meaningful rather than trivially inflated.
2. Deduplicates to one row per customer tweet (a customer message can
   technically pair with more than one direct brand reply in noisy data;
   we sample the *message*, not the pair).
3. Runs an unsupervised TF-IDF + MiniBatchKMeans clustering over the
   candidate pool (same technique as `src/intents/discover.py`, but this
   run's only purpose is a sampling stratum, not the taxonomy itself) and
   allocates the target sample size **proportionally across clusters**,
   with a floor of 1 example per cluster.

## Why the sample is representative

A pure uniform-random sample of 200 out of tens of thousands of golden-eval
messages is very likely, by chance, to under-represent any issue type that
makes up a small fraction of overall traffic (e.g. account-security
escalations are probably rare relative to delivery questions). Stratifying
by an *unsupervised* cluster — computed without knowing the final intent
taxonomy — ensures every distinct topical grouping the data actually
contains gets at least minimal representation in the golden set, while
still weighting larger clusters proportionally more (so the set isn't
artificially balanced either — it should still reflect real intent
frequency, just not lose the tail entirely).

If the golden_eval pool ends up smaller than the target size (unlikely at
AmazonHelp's scale, but the script checks), the fallback is "take
everything," documented in the script's log output rather than silently
producing a smaller-than-expected set.

## How labels were assigned

`scripts/build_golden_set.py label` is an interactive terminal CLI. For
each candidate it shows:
- the customer's message text,
- the brand's actual historical reply to that message (for the labeler's
  context only — it is *not* shown as a suggested label),
- the full intent taxonomy (from `config/intents.yaml`) as a numbered
  menu.

A human types the intent number, then answers a direct yes/no question:
"Should this have been ESCALATED to a human?" (their judgment of what the
*correct* handling would have been, independent of what AmazonHelp's own
agent actually did), plus optional free-text notes. Every answer is
written to `golden_set.csv` immediately (not batched), so the session can
be interrupted (Ctrl-C or `quit`) and resumed later without losing
progress or re-labeling completed rows.

**This stage requires an actual person at the keyboard.** No part of this
pipeline auto-assigns these labels and no commit claims a labeling session
happened unless `golden_set.csv` exists with real `labeled_at` timestamps.

## How ambiguous cases are handled

- The labeler may type `skip` to leave a genuinely unclear example for a
  second pass (e.g. to discuss with another reviewer) rather than forcing
  a low-confidence label immediately.
- `other_unclear` is a first-class intent in the taxonomy specifically so
  "this doesn't fit" has a real, analyzable label instead of being forced
  into the nearest imperfect category. A large `other_unclear` share in
  the final distribution is itself a finding — see `report/report.md`
  Section 12 ("What is misleading about my headline number?").
- The `notes` field is free text specifically for recording *why* a case
  was hard, e.g. "reads like both damaged_or_defective_item and
  return_or_refund_request; picked the former because no explicit return
  is requested." These notes are the raw material for the failure-analysis
  section of the report.

## Label distribution

**PENDING** — populated by `python scripts/build_golden_set.py report`
into `label_distribution.json` once real labeling is complete. Do not fill
this in by hand; do not report numbers here until that file exists.
