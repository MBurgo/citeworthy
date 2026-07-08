"""Pair scheduling, position swap, sampling, and tournament execution.

Full round-robin when the candidate set <= round_robin_max (<= 91 pairs); above
that, a static Swiss-style schedule seeded by pre-filter order (§6.2). Every pair
is judged twice (A/B and B/A) × samples_per_order; position-swap disagreement is
recorded. Judging runs concurrently under a semaphore (max_concurrency). Every
judge call is logged to the DB with token counts, and the run aborts if the
budget cap is exceeded (§6.2, CLAUDE.md).
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations

from .. import db
from ..config import Config
from ..models import Matchup, Passage
from .bradley_terry import Comparison
from .judge import Judge


class BudgetExceeded(Exception):
    """Raised when a run's estimated cost exceeds the configured cap (§8)."""


# --- Pair scheduling -------------------------------------------------------


def _round_robin(ids: list[str]) -> list[tuple[str, str]]:
    return list(combinations(ids, 2))


def _swiss_pairs(n: int, rounds: int = 3, top_k: int = 8) -> list[tuple[int, int]]:
    """Static Swiss-style schedule over `n` seeds (best-first): `rounds` rotation
    rounds (each seed meets a few nearby-strength opponents) plus a full
    round-robin over the top `top_k` seeds. Returns index pairs.
    """
    pairs: set[frozenset[int]] = set()

    # Circle method: rotate all but the first position to generate rounds.
    arr: list[int | None] = list(range(n))
    if n % 2 == 1:
        arr.append(None)  # bye slot
    m = len(arr)
    for _ in range(min(rounds, m - 1)):
        for i in range(m // 2):
            a, b = arr[i], arr[m - 1 - i]
            if a is not None and b is not None:
                pairs.add(frozenset((a, b)))
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]  # rotate, first fixed

    # Full round-robin over the top seeds.
    for a, b in combinations(range(min(top_k, n)), 2):
        pairs.add(frozenset((a, b)))

    ordered: list[tuple[int, int]] = []
    for fs in pairs:
        i, j = sorted(fs)
        ordered.append((i, j))
    ordered.sort()
    return ordered


def schedule_pairs(
    passages: list[Passage], round_robin_max: int
) -> list[tuple[str, str]]:
    """Return the (passage_a_id, passage_b_id) pairs to judge (§6.2).

    `passages` should be in pre-filter (best-first) order so the Swiss seeding is
    meaningful for larger candidate sets."""
    ids = [p.id for p in passages]
    if len(ids) <= round_robin_max:
        return _round_robin(ids)
    return [(ids[i], ids[j]) for i, j in _swiss_pairs(len(ids))]


# --- Task fan-out ----------------------------------------------------------


@dataclass(frozen=True)
class _Task:
    """One judgment: canonical pair (a,b) shown in `order`, sample `sample_index`."""

    a_id: str  # canonical passage a
    b_id: str  # canonical passage b
    order: str  # "ab" (A=a,B=b) or "ba" (A=b,B=a)
    sample_index: int

    @property
    def shown_a(self) -> str:
        return self.a_id if self.order == "ab" else self.b_id

    @property
    def shown_b(self) -> str:
        return self.b_id if self.order == "ab" else self.a_id


def _build_tasks(
    pairs: list[tuple[str, str]], samples_per_order: int
) -> list[_Task]:
    tasks: list[_Task] = []
    for a_id, b_id in pairs:
        for order in ("ab", "ba"):
            for s in range(samples_per_order):
                tasks.append(_Task(a_id=a_id, b_id=b_id, order=order, sample_index=s))
    return tasks


# --- Result ----------------------------------------------------------------


@dataclass
class TournamentResult:
    query_id: str
    passage_ids: list[str]
    matchups: list[Matchup] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    disagreement_rate: float = 0.0
    total_cost_usd: float = 0.0
    n_calls: int = 0
    weak_signal: bool = False  # True if disagreement rate > 30% (§10 acceptance)


DISAGREEMENT_WARN_THRESHOLD = 0.30


def run_tournament(
    query_id: str,
    query_text: str,
    passages: list[Passage],
    judge: Judge,
    config: Config,
    *,
    conn: sqlite3.Connection | None = None,
    run_id: str | None = None,
    budget_cap: float | None = None,
) -> TournamentResult:
    """Schedule pairs, judge each in both orders × samples, and collect the win
    matrix. Logs every call (if `conn`/`run_id` given) and aborts on budget (§6.2).
    """
    passage_ids = [p.id for p in passages]
    text_by_id = {p.id: p.text for p in passages}
    domain_by_id = {p.id: p.domain for p in passages}
    index_of = {pid: i for i, pid in enumerate(passage_ids)}

    pairs = schedule_pairs(passages, config.tournament.round_robin_max)
    tasks = _build_tasks(pairs, config.judge.samples_per_order)

    def _run(task: _Task):
        verdict = judge.judge_pair(
            query_text=query_text,
            passage_a_text=text_by_id[task.shown_a],
            passage_b_text=text_by_id[task.shown_b],
            domain_a=domain_by_id[task.shown_a],
            domain_b=domain_by_id[task.shown_b],
        )
        return task, verdict

    result = TournamentResult(query_id=query_id, passage_ids=passage_ids)
    # winner id per (canonical_pair, order, sample) for disagreement analysis.
    winners: dict[tuple[str, str, str, int], str] = {}

    max_workers = max(1, config.judge.max_concurrency)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for task, verdict in pool.map(_run, tasks):
            winner_id = task.shown_a if verdict.winner == "A" else task.shown_b
            loser_id = task.shown_b if verdict.winner == "A" else task.shown_a

            result.matchups.append(
                Matchup(
                    query_id=query_id,
                    passage_a=task.a_id,
                    passage_b=task.b_id,
                    winner=winner_id,
                    reason_code=verdict.reason_code,
                    reason_text=verdict.reason,
                    judge_model=judge.model,
                    position_order=task.order,  # type: ignore[arg-type]
                    sample_index=task.sample_index,
                )
            )
            result.comparisons.append((index_of[winner_id], index_of[loser_id]))
            winners[(task.a_id, task.b_id, task.order, task.sample_index)] = winner_id

            cost = config.estimate_cost(
                judge.model, verdict.input_tokens, verdict.output_tokens
            )
            result.total_cost_usd += cost
            result.n_calls += 1

            if conn is not None and run_id is not None:
                db.log_judge_call(
                    conn,
                    run_id=run_id,
                    model=judge.model,
                    input_tokens=verdict.input_tokens,
                    output_tokens=verdict.output_tokens,
                    est_usd=cost,
                    created_at=datetime.now(timezone.utc),
                )

            if budget_cap is not None and result.total_cost_usd > budget_cap:
                raise BudgetExceeded(
                    f"run cost ${result.total_cost_usd:.4f} exceeds cap "
                    f"${budget_cap:.2f} after {result.n_calls} judge calls"
                )

    result.disagreement_rate = _disagreement_rate(pairs, config.judge.samples_per_order, winners)
    result.weak_signal = result.disagreement_rate > DISAGREEMENT_WARN_THRESHOLD
    return result


def _disagreement_rate(
    pairs: list[tuple[str, str]],
    samples_per_order: int,
    winners: dict[tuple[str, str, str, int], str],
) -> float:
    """Fraction of pairs where the A/B-order and B/A-order majority winners
    disagree (position bias, §6.2)."""
    if not pairs:
        return 0.0
    disagreements = 0
    for a_id, b_id in pairs:
        ab = [winners[(a_id, b_id, "ab", s)] for s in range(samples_per_order)
              if (a_id, b_id, "ab", s) in winners]
        ba = [winners[(a_id, b_id, "ba", s)] for s in range(samples_per_order)
              if (a_id, b_id, "ba", s) in winners]
        if not ab or not ba:
            continue
        if _majority(ab) != _majority(ba):
            disagreements += 1
    return disagreements / len(pairs)


def _majority(ids: list[str]) -> str:
    counts: dict[str, int] = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    return max(counts, key=lambda k: counts[k])
