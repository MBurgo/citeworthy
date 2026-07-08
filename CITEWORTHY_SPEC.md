# Citeworthy — AI Citation Optimizer
## Build specification for Claude Code

**Owner:** Matt Burgess, Head of Marketing, The Motley Fool Australia
**Purpose:** Measure and improve how often fool.com.au content is selected as a source by AI answer engines (Google AI Mode/Overviews, Gemini, Perplexity, ChatGPT search) for high-value investing queries.
**Inspiration:** DEJAN's "Bayesian Content Optimizer" — this is a lightweight, validation-first internal version.

---

## 1. Concept

AI answer engines don't select sources the way classic ranking does. They retrieve a candidate set, then choose passages to ground on based on relevance, specificity, evidence density, clarity, and extractability. A page can rank #1 organically and never be cited.

Citeworthy has three subsystems, built in three phases:

1. **Diagnostic ranker (v1):** For a target query, assemble the competitive passage set from the live SERP, then run a pairwise LLM-judge tournament to estimate the probability each passage would be selected as a source. Output: where our passage ranks, and *why* it loses.
2. **Edit loop (v2):** Generate rewrite variants targeting the diagnosed weaknesses, re-run the tournament, keep winners. Guided hill-climbing against the ranker, with a compliance gate on every accepted edit.
3. **Ground-truth citation tracker (v3):** Scheduled runs of target queries against real AI surfaces that expose their citations programmatically (Perplexity API; Gemini API with Google Search grounding). Compute longitudinal citation share per domain. This validates that ranker gains translate to real citation gains — and stands alone as an "AI visibility" KPI.

**Build order rationale:** v1 and v3 are independent; v3 should start collecting baseline data as early as possible because before/after claims need a pre-optimisation baseline. Recommended sequence: v1 → v3 (start scheduler) → v2.

## 2. Goals and non-goals

**Goals**
- Answer, per query: "Would an AI engine cite us, and if not, why not?"
- Produce concrete, compliance-safe rewrite recommendations for specific passages.
- Track real citation share over time for a defined query set, with enough samples to distinguish signal from stochastic noise.
- Cheap to run (target: < AUD $5 per full diagnostic run of 20 queries; < AUD $30/month for weekly tracking of 20 queries).

**Non-goals**
- No attempt to reverse-engineer any single engine's exact retrieval algorithm. The ranker is an explicit proxy; the tracker is the truth.
- No CMS integration or auto-publishing. Output is recommendations and reports; humans apply edits.
- No scraping of ChatGPT or Google AI Mode UIs in v1–v3 (fragile, ToS-risky). Perplexity + Gemini grounding are the programmatic ground truth. UI-based sampling can be a manual monthly sanity check.

## 3. Tech stack

- **Language:** Python 3.11+, managed with `uv`. Type hints throughout; `pydantic` v2 for all data models.
- **LLM judge:** Anthropic API, `claude-haiku-4-5-20251001` (cheap, fast, sufficient for pairwise judgments).
- **LLM editor:** Anthropic API, `claude-sonnet-4-6` (rewrite quality matters more here).
- **Ranking model:** Bradley–Terry via the `choix` library (fallback: simple Elo if choix fitting is unstable with sparse data).
- **SERP data:** pluggable provider interface. Implement `SerpApiProvider` first (serpapi.com, `google` engine, `gl=au`, `hl=en`). Design the interface so DataForSEO can be added later. Do NOT scrape Google directly.
- **Content extraction:** `trafilatura` for main-content extraction; `httpx` for fetching with realistic headers, timeouts, and per-domain politeness delays.
- **Ground truth:** Perplexity API (`sonar` model family — verify current model names in their docs at build time; responses include a citations/sources list) and Gemini API with the Google Search grounding tool (read `groundingMetadata.groundingChunks` for cited URIs). Verify exact response field names against current docs at build time — do not trust this spec's field names blindly.
- **Storage:** SQLite via `sqlmodel` or plain `sqlite3`. Single file `citeworthy.db`. (A later port to Cloudflare D1/Workers can reuse the schema; do not block on it.)
- **Reporting:** Markdown reports written to `reports/`, plus a simple `rich`-formatted terminal summary.
- **Scheduling (v3):** plain cron invoking the CLI (`citeworthy track run`). Keep the tool cron-friendly: idempotent, exit codes, logs to file.
- **Config:** `config.yaml` (validated by pydantic) + `.env` for keys.

## 4. Repository structure

```
citeworthy/
  pyproject.toml
  .env.example
  config.yaml
  CLAUDE.md
  src/citeworthy/
    __init__.py
    cli.py               # typer CLI entrypoint
    config.py            # pydantic config models + loader
    models.py            # core data models (Query, Passage, Matchup, RankResult, ...)
    db.py                # SQLite schema + repository functions
    serp/
      base.py            # SerpProvider protocol
      serpapi.py
    extract.py           # fetch + trafilatura + chunking
    ranker/
      judge.py           # pairwise judge (Anthropic API)
      tournament.py      # pair scheduling, position swap, sampling
      bradley_terry.py   # choix fitting, uncertainty
      diagnose.py        # loss-reason aggregation
    editor/
      variants.py        # rewrite variant generation
      compliance.py      # compliance gate (see §9)
      loop.py            # hill-climbing controller
    truth/
      perplexity.py
      gemini.py
      share.py           # citation share computation + stats
    report.py            # markdown + terminal reports
  tests/
    fixtures/            # canned SERP JSON, HTML pages, judge responses
    test_extract.py
    test_tournament.py
    test_bradley_terry.py
    test_share.py
  reports/               # generated output (gitignored)
```

## 5. Core data models

```python
class TargetQuery(BaseModel):
    id: str                    # slug, e.g. "best-asx-dividend-shares"
    text: str                  # "best ASX dividend shares to buy"
    our_url: HttpUrl | None    # page we're optimising (None = tracker-only query)
    tier: Literal["money", "supporting"]

class Passage(BaseModel):
    id: str                    # hash of (url, chunk_index)
    url: HttpUrl
    domain: str
    title: str
    chunk_index: int
    text: str                  # 200–400 words
    is_ours: bool
    variant_of: str | None     # set for edit-loop variants
    variant_label: str | None  # e.g. "specificity", "front-load-answer"

class Matchup(BaseModel):
    query_id: str
    passage_a: str
    passage_b: str
    winner: str                # passage id
    reason_code: str           # from taxonomy, §7
    reason_text: str
    judge_model: str
    position_order: Literal["ab", "ba"]
    sample_index: int

class RankResult(BaseModel):
    query_id: str
    run_id: str
    passage_id: str
    bt_strength: float         # Bradley–Terry log-strength
    selection_prob: float      # softmax over strengths within the candidate set
    rank: int
    ci_low: float
    ci_high: float             # bootstrap CI on selection_prob

class CitationObservation(BaseModel):
    run_id: str
    query_id: str
    engine: Literal["perplexity", "gemini_grounded"]
    sample_index: int
    cited_urls: list[str]
    answer_text_hash: str
    observed_at: datetime
```

SQLite tables mirror these models, plus `runs` (run_id, kind, started_at, config_hash, git_sha) so every result is reproducible.

## 6. Component specs

### 6.1 Competitive set builder
1. Query the SERP provider for the target query (Australian locale). Take the top 10 organic URLs. Also capture People Also Ask questions if the provider returns them (stored for future query-set expansion, not used in v1 ranking).
2. Add `our_url` if it isn't already in the top 10.
3. Fetch each URL (httpx, 15s timeout, 2 retries, realistic UA, respect robots.txt via `urllib.robotparser`; skip disallowed URLs and log it). Cache raw HTML in `cache/` keyed by URL hash + date, so re-runs within 7 days don't re-fetch.
4. Extract main content with trafilatura. If extraction returns < 200 words, mark the source `extraction_failed` and exclude with a warning.
5. Chunk into passages: split on headings/paragraphs, merge to 200–400 word windows, keep heading context prepended (e.g. "## Best dividend picks for 2026 — " + body). Cap at 6 chunks per URL, selected by a cheap relevance pre-filter (embed-free heuristic in v1: keyword overlap + position; the judge is too expensive to run on every chunk of every page).
6. For our page, keep ALL chunks (we want to find our best passage, not assume it).
7. Candidate set per query: aim for 12–20 passages total. If over budget, keep each competitor's single best pre-filtered chunk.

### 6.2 Ranker — pairwise tournament
- **Pair scheduling:** full round-robin when candidate set ≤ 14 passages (≤ 91 pairs). Above that, Swiss-style: 3 seeded rounds using pre-filter scores, then top-8 round-robin.
- **Debiasing:** every pair is judged twice, once in each position order (A/B and B/A). Disagreement between orders is recorded; pairs with persistent disagreement are effectively ties.
- **Sampling:** temperature 0.3, 2 samples per position order (4 judgments per pair). Configurable.
- **Concurrency:** async with a semaphore (default 8 concurrent requests). Exponential backoff on 429/529. Every judge call and response is logged to the DB for audit.
- **Fitting:** convert judgments to a win matrix; fit Bradley–Terry with `choix.ilsr_pairwise` (regularisation alpha=0.01 for sparse data). Compute `selection_prob` as softmax over strengths. Bootstrap the judgment set (200 resamples) for CIs.
- **Output:** ranked table with probabilities and CIs; flag whether our best passage's CI overlaps with the leader's.

### 6.3 Judge prompt (v1 draft — iterate freely, but keep the JSON contract)

System prompt for the judge:

```
You are the source-selection component of an AI answer engine. Given a user
query and two candidate passages retrieved from the web, decide which single
passage you would ground your answer on (i.e., cite as a source).

Judge ONLY on the passage text provided. Weigh, in rough priority order:
1. Direct relevance: does it actually answer the query as asked?
2. Specificity and evidence: concrete facts, figures, dates, named entities,
   data — not generalities.
3. Extractability: is the answer stated plainly and early, in self-contained
   sentences an engine could quote or paraphrase cleanly?
4. Clarity and structure: unambiguous, well-organised, low fluff.
5. Freshness signals IN THE TEXT (explicit dates, current figures). Do not
   guess publication dates.
Ignore: brand familiarity, domain reputation, writing flair, length for its
own sake. Longer is not better.

Respond with ONLY a JSON object, no markdown fences:
{
  "winner": "A" | "B",
  "confidence": "clear" | "moderate" | "slight",
  "reason_code": one of ["off_topic","too_generic","buried_answer",
    "no_evidence","stale_signals","hedgy","poor_structure",
    "promotional","thin","other"],
  "reason": "<one sentence: why the loser lost>"
}
The reason_code describes the LOSING passage's primary weakness.
```

User message template:

```
Query: {query_text}

Passage A (source: {domain_a}):
"""
{passage_a_text}
"""

Passage B (source: {domain_b}):
"""
{passage_b_text}
"""
```

Note: including the domain is a deliberate v1 choice so the judge context resembles real retrieval; run an ablation later with domains masked to measure brand bias in the judge. Make domain visibility a config flag (`judge.show_domains: true`).

### 6.4 Diagnostics
Aggregate `reason_code` across every matchup our passages lost. Report: loss-reason distribution, the top 3 weaknesses, and 2–3 verbatim `reason_text` examples per weakness. Also report our best chunk vs. our page's first chunk — the "buried answer" gap is expected to be the most common finding.

### 6.5 Edit loop (v2)
1. Take our best passage + top 3 diagnosed weaknesses.
2. Generate 3–4 variants with `claude-sonnet-4-6`, each targeting one weakness, each labelled. Constraints in the editor prompt: preserve all factual claims exactly (no new facts, no new numbers, no invented data); preserve meaning; 200–400 words; plain declarative answer in the first two sentences; keep The Motley Fool AU's plain-English voice.
3. **Compliance gate (hard requirement):** every variant passes through a checker before entering the tournament. Implement as a pluggable hook `compliance.check(text) -> ComplianceResult`. v2 ships with a prompt-based checker enforcing: no guarantees or promises of returns, no unqualified performance claims, no advice-like imperatives ("you should buy"), past performance disclaimers preserved if present in the original, no fabricated statistics. Matt has an existing ASIC compliance checker — build the hook so it can be swapped in. Variants that fail are logged and discarded.
4. Insert surviving variants into the candidate set; re-run the tournament (variants + original + top 5 competitors is sufficient — no need to re-judge the full field).
5. Accept a variant only if its selection_prob beats the original's with non-overlapping bootstrap CIs. Otherwise keep the original ("no significant improvement" is a valid, reportable outcome).
6. Max 3 iterations. Output: final recommended passage, diff against original, per-iteration score trajectory.

### 6.6 Ground-truth tracker (v3)
- For each tracked query, per weekly run: 10 samples against Perplexity API, 10 against Gemini with Google Search grounding (temperature default; the stochasticity is the point). Collect cited/grounded URLs per sample.
- Normalise URLs (strip tracking params, resolve to canonical where cheap) and aggregate to domain.
- **Citation share** (per query, engine, run) = fraction of samples in which the domain appears in the citation list at least once. Also record mean citation position where available.
- Report weekly: our citation share per query and engine, trend vs. baseline, and which domains dominate each query (the real competitive set — feed this back into 6.1's candidate list over time).
- **Statistics discipline:** with 10 samples, a single week's movement is noise. Report 3-week rolling share with Wilson score intervals; only flag changes where intervals don't overlap the baseline's. The report must say "insufficient data" rather than imply trends from one run.
- Store everything; never overwrite. Baseline = the first 2–3 weeks before any v2-driven edits ship.

### 6.7 Reporting
`citeworthy report <query_id>` renders a markdown report per query: ranked passage table (domain, rank, selection prob ± CI, ours highlighted), loss-reason diagnosis, recommended actions, and — once v3 data exists — the citation-share trend. Plus `citeworthy report --all` for a portfolio summary sorted by (query tier, gap to leader). Keep reports plain markdown so they can be pasted into Slack/docs.

## 7. CLI interface (typer)

```
citeworthy init                          # create config.yaml, .env.example, db
citeworthy queries add|list|rm           # manage target query set
citeworthy rank <query_id> [--all]       # v1: build set, run tournament, report
citeworthy optimize <query_id>           # v2: edit loop on a ranked query
citeworthy track run [--engine ...]      # v3: one tracking pass (cron target)
citeworthy report <query_id> | --all
citeworthy costs                         # token spend summary from logged calls
```

## 8. Configuration (config.yaml sketch)

```yaml
locale: { gl: au, hl: en, location: "Australia" }
our_domain: fool.com.au
serp: { provider: serpapi, top_n: 10 }
chunking: { min_words: 200, max_words: 400, max_chunks_per_url: 6 }
judge:
  model: claude-haiku-4-5-20251001
  temperature: 0.3
  samples_per_order: 2
  show_domains: true
  max_concurrency: 8
editor: { model: claude-sonnet-4-6, variants: 4, max_iterations: 3 }
tournament: { round_robin_max: 14, bootstrap_resamples: 200 }
truth:
  samples_per_engine: 10
  engines: [perplexity, gemini_grounded]
budget:
  max_usd_per_rank_run: 2.00        # hard stop; abort run if exceeded
  max_usd_per_track_run: 3.00
```

Budget enforcement: track input/output tokens per call, estimate cost from a price table in config (editable — API prices change), halt with a clear error when the cap is hit.

## 9. Environment / keys

`.env`: `ANTHROPIC_API_KEY`, `SERPAPI_API_KEY`, `PERPLEXITY_API_KEY`, `GEMINI_API_KEY`. All optional except Anthropic; components degrade gracefully with a clear message when a key is missing (e.g., tracker runs Perplexity-only if no Gemini key).

## 10. Testing and acceptance criteria

**Testing approach:** no live API calls in tests. Fixtures for SERP JSON, HTML pages, and judge JSON responses; the judge client takes an injectable transport so tests use canned responses. Property tests for chunker (no chunk outside word bounds, no dropped text) and Bradley–Terry (transitive dominance in a synthetic win matrix recovers the true order).

**v1 acceptance:**
- `citeworthy rank best-asx-dividend-shares` completes end-to-end against live APIs in < 10 minutes and < $2.
- Position-swap disagreement rate is reported; if > 30%, the run warns that judge signal is weak.
- Sanity eval: on 3 hand-built test sets where one passage is objectively superior (planted specific data vs. generic fluff), the superior passage ranks #1 in ≥ 2 of 3.
- Report renders with CIs and a coherent diagnosis.

**v2 acceptance:**
- At least one variant beats the original with non-overlapping CIs on a real query, OR the loop correctly reports no significant improvement.
- Zero variants bypass the compliance gate (gate failures are logged, discarded, visible in the report).
- Diff output is clean and paste-ready.

**v3 acceptance:**
- Cron-driven weekly run populates observations for 20 queries across both engines unattended for 2 consecutive weeks.
- Report shows per-query citation share with Wilson intervals and refuses to claim trends from a single run.

## 11. Cost model (estimate — verify against current pricing)

- Rank run, one query: ~91 pairs × 4 judgments × ~1,400 input / 120 output tokens ≈ 0.5M input + 45K output tokens on Haiku → well under $1. Twenty queries ≈ a few dollars.
- Track run, weekly: 20 queries × 20 samples ≈ 400 answer-engine calls; Perplexity and Gemini costs dominate — check their current pricing and reflect it in `citeworthy costs`.

## 12. Risks and open decisions (surface these during the build, don't silently choose)

1. **Judge–reality correlation is unproven.** Mitigation is the whole v3 design; additionally, once 4+ weeks of tracker data exist, compute the rank correlation between ranker selection_prob and observed citation share across queries and print it in the portfolio report. If correlation is weak, the judge prompt (not the architecture) is the first thing to iterate.
2. **SERP ≠ AI retrieval candidate set.** Google SERP is a proxy for what engines retrieve. v3's observed-competitor data partially corrects this: feed frequently-cited domains back into the candidate builder.
3. **Perplexity/Gemini API response schemas change.** Isolate parsing in `truth/`, fail loudly with the raw payload logged.
4. **trafilatura failures on JS-heavy competitor pages.** Log extraction failures per domain; if a major competitor consistently fails, that's a known blind spot to report, not silently drop.
5. **Judge self-preference:** an Anthropic judge may have systematic tastes. The debate-orchestrator pattern (multi-model judging) is a natural v4; design `judge.py` so the model is injectable.

## 13. Suggested CLAUDE.md for the repo

```
# Citeworthy
Internal tool: measures and improves fool.com.au citation share in AI answers.
- Python 3.11+, uv, pydantic v2, typer, SQLite. Run tests with `uv run pytest`.
- NEVER make live API calls in tests; use fixtures in tests/fixtures/.
- All LLM calls go through src/citeworthy/ranker/judge.py or editor/ — never
  inline API calls elsewhere. Every call is logged to the db with token counts.
- Respect the budget caps in config.yaml; abort, don't overspend.
- The judge prompt JSON contract in §6.3 of CITEWORTHY_SPEC.md is load-bearing;
  changing fields requires updating models.py, db.py, and diagnose.py together.
- Compliance gate (editor/compliance.py) must run on every generated variant.
  No exceptions, including in tests of the loop (use a stub checker).
- Verify current model names/pricing at https://docs.claude.com before
  hardcoding; same for Perplexity and Gemini response schemas.
```

## 14. Build sequence for Claude Code

1. Scaffold repo, config, models, db, CLI skeleton (all commands stubbed).
2. Competitive set builder + chunker, with fixtures and tests.
3. Judge client + tournament + Bradley–Terry + bootstrap, tested on synthetic matrices.
4. `rank` end-to-end + report. **Stop here for human review of a real run.**
5. Tracker (`truth/` + `track run` + share stats). Start cron immediately after review.
6. Edit loop + compliance gate.
7. Portfolio report + judge-vs-reality correlation.

Milestone 4 is the checkpoint: Matt reviews a real ranked report for one Share Advisor money query before any further build.
