# Engineering Decision Log

Non-obvious decisions made while building this project, with alternatives
considered and the tradeoff accepted. Entries are added as each phase is
built; earlier entries are not renumbered.

---

## 1. Two-pass streaming instead of one (structural index, then text)

**Decision:** Preprocessing reads the CSV twice: pass 1 streams only the
structural columns (`tweet_id`, `author_id`, `inbound`, `created_at`,
`response_tweet_id`, `in_response_to_tweet_id`) to build an in-memory index
and reconstruct threads; pass 2 streams full rows (including `text`) but
only materializes the subset of `tweet_id`s already known to belong to the
target brand's conversations.

**Alternatives considered:**
- Single pass holding everything (including `text`) in memory: simplest
  code, but `text` is the majority of the file's bytes — for 2.8M rows this
  risks multi-GB memory use for no benefit, since thread reconstruction
  never needs `text`.
- Single pass, chunked, discarding non-brand rows immediately: avoids a
  second file read, but conversation membership can only be determined
  *after* the full reply graph is known (a customer's first message doesn't
  mention the brand at all), so a brand-only single pass would miss the
  root message of every thread.

**Why chosen:** The structural index (6 short columns) is small enough to
hold entirely in memory even at 2.8M rows, which makes thread
reconstruction (a graph problem) tractable with a simple in-memory
union-find. The second pass then only pays the I/O/parsing cost of `text`
for a small fraction of the file.

**Tradeoff:** Two full sequential scans of the CSV instead of one — slower
wall-clock time, in exchange for bounded, predictable memory use and much
simpler code than a single-pass streaming graph algorithm.

---

## 2. Conversations are reply-graph connected components, not time windows

**Decision:** A "conversation" (`conversation_id`) is a connected component
of the graph formed by `in_response_to_tweet_id` / `response_tweet_id`
edges (via union-find), not tweets grouped by a fixed time window per
customer.

**Alternatives considered:** Group by `(author_id, day)` or a sliding time
window — simpler, no graph algorithm needed.

**Why chosen:** The reply pointers are ground truth for "which messages
are part of the same support interaction"; a time-window heuristic would
both merge unrelated same-day tweets from a chatty customer and split a
single conversation that happens to span midnight.

**Tradeoff:** Union-find adds code complexity and a full-index memory
requirement (accepted per decision #1).

---

## 3. Thread-level train/golden split, not row-level

**Decision:** `assign_thread_split` splits **conversation_ids** (not
individual rows or resolution pairs) into `train_retrieval` /
`golden_eval`, via a seeded permutation of the unique conversation-id list.

**Why:** A row-level random split would let one side of a conversation
(e.g. the customer's message) land in the golden evaluation set while the
brand's actual reply to that exact message sits in the retrieval corpus —
retrieval would then trivially "find" the answer, inflating every
retrieval and groundedness metric. Splitting whole threads is the only way
to guarantee a golden example's resolution isn't sitting in the evidence
index used to grade it.

**Tradeoff:** Slightly less precise control over the exact train/golden
row-count ratio (a conversation's size varies), acceptable given the
alternative is leakage.

---

## 4. `resolution_pairs` counts only direct reply pairs

**Decision:** A "resolution pair" is a brand outbound tweet whose
`in_response_to_tweet_id` points **directly** at a customer inbound tweet —
not any customer/brand tweet pair that merely share a `conversation_id`.

**Why:** Using conversation co-membership would pair a customer's *second*
message with the brand's *first* reply, an implied context reversal, and
would explode the pair count with combinatorially many indirect pairs
within longer threads. Direct pairs are unambiguous, real evidence of "the
brand replied to this specific issue this specific way."

**Tradeoff:** Undercounts the true amount of resolution evidence in
multi-turn threads (a 4-message thread yields 2 direct pairs, not more),
which is conservative but correct — Phase 5 (retrieval) can still use the
full conversation as *context* around a pair without treating indirect
combinations as ground truth.

---

## 5. `@mentions` are kept, URLs are replaced with a placeholder

**Decision:** `clean_text` keeps `@handle` mentions by default (the
brand's authentic reply voice starts with `@customer`) but replaces URLs
with a literal `[link]` token.

**Why:** Stripping mentions would make retrieved evidence and generated
replies look unlike the brand's real style, which the agent is explicitly
meant to imitate. URLs are almost always dead/anonymized tracking links in
this dataset and add no retrievable semantic value, but their *presence*
is a weak signal worth preserving as a token rather than deleting outright.

**Tradeoff:** None significant; `strip_mentions=True` is available as an
opt-in for any downstream use (e.g. TF-IDF vocabulary) that wants mention
noise removed.

---

## 6. Candidate-brand scoring is a documented weighted average, not a single metric

**Decision:** `score = mean(volume_score, pairs_score, structure_score)`,
each a clipped [0,1] sub-score (log-scaled volume, resolution pairs
relative to `golden_target * 5`, multi-turn fraction relative to 0.5).

**Alternatives considered:** Rank by raw outbound-message count alone
(simplest, but ignores whether there's enough *paired* evidence or
conversational depth); a learned/weighted model (overkill and unjustifiable
without labeled ground truth for "good brand for this task").

**Why chosen:** Each sub-score maps to a concrete requirement from the
assignment (enough volume; enough resolution pairs to support a golden set
*and* a retrieval corpus without starving either; conversational depth
representative of real support interactions), and the formula is fully
transparent in `candidate_brands.md` rather than a black box.

**Tradeoff:** Equal weighting of the three sub-scores is a judgment call,
not derived from data; documented here rather than hidden.

---

## 7. Duplicate `tweet_id` rows: drop, keep first, before all statistics

**Decision:** `clean.drop_duplicate_tweet_ids` runs before thread
reconstruction and before any reported statistic.

**Why:** A duplicate `tweet_id` would silently corrupt the union-find graph
(two "different" rows claiming the same graph node) and double-count in
every volume statistic. This should not occur in a well-formed export, but
the fixture test (`tests/fixtures/sample_twcs.csv`) seeds exactly one
duplicate to prove the guard works.

**Tradeoff:** "Keep first occurrence" is arbitrary when a genuine duplicate
has different content; the duplicate count is still reported in
`dataset_summary.json["duplicates"]` before dropping, so this choice is
visible rather than silent.

---

## 8. Central `src/config.py` instead of scattered constants

**Decision:** Every cross-module tunable (brand, sample size, seed,
embedding model, retrieval `top_k`, escalation thresholds, LLM
model/timeouts, split fractions) lives in one dataclass-based config file,
sourced from environment variables with defaults.

**Why:** The assignment is explicitly reviewed and modified live; a single
place to find/change any threshold is far more defensible in that setting
than constants buried across a dozen files.

**Tradeoff:** One more layer of indirection versus hardcoding values inline
where used; accepted for reviewability.

---

## 9. pandas pinned to `<3.0`

**Decision:** `requirements.txt` pins `pandas>=2.2,<3.0` rather than
letting pip install the (newly released, at the time of writing) pandas
3.0.

**Why:** pandas 3.0 changes default behaviors (string dtype backend,
copy-on-write semantics) that interact with this project's chunked
`pd.concat` / `groupby` / `merge`-heavy pipeline in ways not yet validated.
Pinning to the mature 2.x line keeps behavior predictable for a reviewer
running this on a different machine/date.

**Tradeoff:** Misses pandas 3.x's performance improvements; revisit once
the pipeline has been validated against it.

---

## 10. Intent taxonomy shipped as a labeled DRAFT, not fabricated cluster output

**Decision:** `config/intents.yaml` ships with `status:
DRAFT_PENDING_DATA_VALIDATION`, seeded from domain knowledge of what
`@AmazonHelp` actually handles (order/delivery, refunds, account, Prime,
device support) rather than from real clustering output, because the real
dataset was not available in the environment that built this taxonomy.
`src/intents/discover.py` is the real, tested tool that produces the
evidence (TF-IDF + MiniBatchKMeans clusters, top terms, real sample
messages) a human then uses to validate/revise this file.

**Alternatives considered:** Wait to write any taxonomy until the real
data is available (blocks every downstream phase — golden set, retrieval,
agent, evaluation — on one manual step); silently run discovery on a tiny
placeholder and present its output as if it were real (would fabricate
results, explicitly forbidden).

**Why chosen:** A clearly-labeled seed taxonomy lets every downstream
phase (golden set schema, agent intent list, escalation rules, baseline
classifier) be built and tested now against a real, reviewable set of
intents, while making unmistakable — in the file itself — that intent
*names and boundaries* are pending validation, and that the discovery tool
is what validates them, not domain guesswork alone.

**Tradeoff:** Anyone reading only the taxonomy without noticing the status
field could mistake it for data-validated; mitigated by putting the status
and required-next-step instructions at the very top of the file, in the
README, and here.

---

## 11. The "simple baseline" classifier IS the production classifier

**Decision:** `src/intents/classifier.py`'s TF-IDF + LogisticRegression
`IntentClassifier` is used both as the agent's actual (Phase 7) intent
classifier and, unmodified, as "Baseline 2" (Phase 8's required
simple/non-LLM baseline).

**Alternatives considered:** Build two separate classifiers — a
deliberately weaker one to serve as "the baseline" and a fancier one (e.g.
an LLM-based classifier) as "the real system."

**Why chosen:** The assignment explicitly says to prefer deterministic/
local methods over the LLM where they're strong enough, and a TF-IDF
classifier is strong enough for a well-scoped, ~10-class intent problem.
Using the *same* model for both roles means the "baseline comparison" is
honest — it's not rigged by comparing a strawman to the real system — and
the real system doesn't burn an LLM call per incoming message just to
pick one of ~10 labels.

**Tradeoff:** Baseline 2 and the production classifier will always score
identically on intent metrics; the comparison that remains meaningful is
Baseline 1 (trivial) vs. this classifier, and — separately — this
classifier's retrieval+reply+escalation pipeline vs. Baseline 2's
simpler TF-IDF-nearest-neighbor reply/rule-based escalation (Phase 8).

---

## 12. Golden-set sampling is stratified by *unsupervised* clusters, not by intent

**Decision:** `scripts/build_golden_set.py sample` stratifies the 200-ish
golden-set sample by a fresh TF-IDF+KMeans clustering over the
golden_eval pool itself, not by the (still-draft) intent taxonomy or by
the discovery clusters computed in Phase 3.

**Why:** Using the taxonomy to stratify would bias the sample toward
whatever intent boundaries were guessed in Phase 3, potentially hiding a
real category the taxonomy missed. An independent unsupervised clustering
run on the eval pool itself surfaces the pool's actual structure without
assuming the taxonomy is already correct, and a human still assigns the
real intent label per example during `label`.

**Tradeoff:** Two separate clustering runs exist in the codebase (Phase 3
discovery vs. this sampling stratification) with similar code — accepted
because they serve different purposes (taxonomy evidence vs. sampling
diversity) and conflating them would make it unclear which run any given
number came from.

---

## 13. The labeling CLI writes to disk after every single example

**Decision:** `cmd_label` calls `df.to_csv(...)` after each labeled row,
not once at the end of the session.

**Why:** A 200-example hand-labeling session will not finish in one
sitting. Losing partially-completed work to a closed terminal, a crash, or
an accidental Ctrl-C would be a serious cost for a task that's
irreducibly manual. The per-row save cost (a small CSV rewrite) is
negligible next to that risk.

**Tradeoff:** None meaningful at this scale (hundreds of rows, not
millions).

---

## 14. Retrieval indexes the customer-message side, evidence is the paired brand reply

**Decision:** The FAISS index embeds `customer_text` (historical customer
messages); a search returns those messages' nearest neighbors, and the
*paired* `brand_text` (the brand's actual historical reply to that
specific message) is what's surfaced as grounding evidence.

**Why:** The agent's query at inference time is also a customer message,
so customer-to-customer semantic similarity is the correct search space.
Embedding brand replies instead (or in addition) would search "which past
replies sound like a good reply", which has no connection to whether that
reply actually addressed a similar problem.

**Tradeoff:** If a historical brand reply doesn't actually resolve the
paired customer message well (real support data is imperfect — see report
Section 12), that flawed reply is exactly what gets retrieved and could be
what the agent grounds on. This is a known limitation, not fixed here; the
agent's prompt/generation step (Phase 7) is expected to synthesize across
multiple retrieved pairs rather than copy any single one verbatim.

---

## 15. `Retriever.build` hard-fails on non-`train_retrieval` input

**Decision:** Building the index raises `LeakageError` if any row's
`split` isn't `train_retrieval`, unless the caller explicitly passes
`allow_any_split=True` (used only by tests).

**Why:** Given how much of this assignment's grading rests on "prove
retrieval doesn't leak," a runtime assertion that fails loudly is worth
more than a code comment saying "remember to filter to train_retrieval
before calling this." `scripts/build_index.py` still filters explicitly
too (defense in depth), but the library function itself refuses to be
misused silently.

**Tradeoff:** Slightly more ceremony for legitimate exploratory use (e.g.
"what would retrieval find if I included everything") — solved by the
explicit opt-out flag, which is greppable and impossible to do by
accident.

---

## 16. Embeddings are normalized; the FAISS index is `IndexFlatIP`

**Decision:** `SentenceTransformerEmbedder.embed` L2-normalizes every
vector (`normalize_embeddings=True`), and the index is a flat
inner-product index (`faiss.IndexFlatIP`), so inner product == cosine
similarity.

**Alternatives considered:** `IndexFlatL2` (Euclidean distance) requires a
separate mental model for "lower is better" vs. the intuitive "higher
similarity is better," and doesn't directly correspond to cosine
similarity without also normalizing — same normalization requirement, less
intuitive output.

**Why chosen:** Cosine similarity is the standard, interpretable
similarity measure for sentence embeddings; normalizing once at embed time
means every downstream consumer of a similarity score (escalation
thresholds, evaluation Recall@K, the report) can treat it as a plain
[-1, 1] cosine value.

**Tradeoff:** `IndexFlatIP` is exact (brute-force) search, O(n) per query —
fine at AmazonHelp's realistic corpus size (tens of thousands of pairs)
on CPU, but would need an approximate index (e.g. `IndexIVFFlat`) at much
larger scale. Documented here rather than prematurely engineered now.

---

## 17. LLM access goes through a provider-agnostic interface, never `openai`/NVIDIA directly

**Decision:** `src/llm/base.py` defines `LLMProvider` (an ABC with
`is_configured()` / `generate()`); `src/llm/nvidia.py`'s
`NvidiaLLMProvider` is the only implementation, but the agent, judge, and
any future caller depend on `LLMProvider`/`LLMMessage`/`LLMResponse`, never
on `openai.OpenAI` or NVIDIA-specific request/response shapes directly.

**Why:** The assignment explicitly asks for this so the provider can be
swapped later; NVIDIA's endpoint happening to be OpenAI-compatible today
doesn't guarantee every future provider will be.

**Tradeoff:** One extra layer of indirection for a project with exactly
one real provider today — accepted because it's a small, contained cost
(~60 lines) for a requirement stated directly in the brief.

---

## 18. Missing/placeholder API key raises before any network call, not on first failure

**Decision:** `NvidiaLLMProvider._get_client()` checks
`config.is_configured()` (real key present, not the literal placeholder
`"YOUR_NVIDIA_API_KEY_HERE"`) and raises `LLMNotConfiguredError`
immediately — before constructing an `OpenAI` client or entering the
retry loop.

**Why:** Without this check, an unset/placeholder key would still attempt
3 real HTTP calls with exponential backoff (per decision #19) against
NVIDIA's endpoint, wait several seconds, and then fail with a generic
auth error — burning time and producing a confusing error for exactly the
"I haven't set up my API key yet" case the assignment says must "fail
gracefully with a clear message."

**Tradeoff:** None — this is strictly better than the alternative in every
case; documented because it's easy to accidentally regress (e.g. moving
the check after client construction) without a test, which is why
`test_generate_raises_not_configured_without_calling_network` asserts
`OpenAI()` is never even constructed in this path.

---

## 19. Retries use `tenacity` with exponential backoff, configurable retry count

**Decision:** `NvidiaLLMProvider.generate` retries via
`tenacity.Retrying` with `stop_after_attempt(config.max_retries)` and
`wait_exponential(multiplier=1, min=1, max=10)`, re-raising the last
exception (wrapped as `LLMProviderError`) if all attempts fail.

**Alternatives considered:** Hand-rolled retry loop with `time.sleep` —
avoids one small dependency, but `tenacity` is a well-tested, ~single
purpose library and hand-rolling backoff/jitter correctly is exactly the
kind of thing worth not reinventing.

**Why chosen:** Transient network/rate-limit errors are expected against
any real API; failing after one attempt would make the agent needlessly
fragile, while retrying forever would hang. Exponential backoff bounded at
10s keeps total added latency bounded even at `max_retries=3`.

**Tradeoff:** `retry_if_exception_type(Exception)` retries on *any*
exception, including e.g. a malformed-request error that will never
succeed on retry. Accepted for now (the OpenAI SDK's error hierarchy would
let us retry only on `RateLimitError`/`APIConnectionError` specifically)
because the current failure mode of "retry a few times, then escalate to
human" is safe either way — a human seeing "the agent tried and gave up"
is an acceptable outcome for both transient and permanent LLM failures.

---

*(Further decisions for the agent pipeline, escalation policy, baselines,
and evaluation are appended as those phases are built.)*
