"""Core data models for Citeworthy.

These pydantic v2 models are the canonical shape of the domain. The SQLite
schema in db.py mirrors them. See §5 of CITEWORTHY_SPEC.md.

The judge JSON contract (§6.3) is load-bearing: changing Matchup fields requires
updating db.py and ranker/diagnose.py together.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl

# Reason-code taxonomy for the loser of a matchup (§6.3). Kept here so models.py,
# db.py, and ranker/diagnose.py share one source of truth.
REASON_CODES: tuple[str, ...] = (
    "off_topic",
    "too_generic",
    "buried_answer",
    "no_evidence",
    "stale_signals",
    "hedgy",
    "poor_structure",
    "promotional",
    "thin",
    "other",
)

RunKind = Literal["rank", "optimize", "track"]


class TargetQuery(BaseModel):
    id: str  # slug, e.g. "best-asx-dividend-shares"
    text: str  # "best ASX dividend shares to buy"
    our_url: HttpUrl | None = None  # page we're optimising (None = tracker-only query)
    tier: Literal["money", "supporting"]


class Passage(BaseModel):
    id: str  # hash of (url, chunk_index)
    url: HttpUrl
    domain: str
    title: str
    chunk_index: int
    text: str  # 200–400 words
    is_ours: bool = False
    variant_of: str | None = None  # set for edit-loop variants
    variant_label: str | None = None  # e.g. "specificity", "front-load-answer"


class Matchup(BaseModel):
    query_id: str
    passage_a: str
    passage_b: str
    winner: str  # passage id
    reason_code: str  # from taxonomy, §6.3 / REASON_CODES
    reason_text: str
    judge_model: str
    position_order: Literal["ab", "ba"]
    sample_index: int


class RankResult(BaseModel):
    query_id: str
    run_id: str
    passage_id: str
    bt_strength: float  # Bradley–Terry log-strength
    selection_prob: float  # softmax over strengths within the candidate set
    rank: int
    ci_low: float
    ci_high: float  # bootstrap CI on selection_prob


class CitationObservation(BaseModel):
    run_id: str
    query_id: str
    engine: Literal["perplexity", "gemini_grounded"]
    sample_index: int
    cited_urls: list[str]
    answer_text_hash: str
    observed_at: datetime


class Run(BaseModel):
    """A single execution, so every result is reproducible (§5)."""

    run_id: str
    kind: RunKind
    started_at: datetime
    config_hash: str
    git_sha: str | None = None
