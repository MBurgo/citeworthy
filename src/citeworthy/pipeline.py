"""Rank pipeline orchestration (Milestone 4).

Wires the pieces built in Milestones 2–3 into the end-to-end `rank` flow:
competitive-set builder → tournament → Bradley–Terry → persistence → report.

Dependencies (SERP provider, judge, httpx client) are injected so the flow is
testable against fixtures with no live API calls (§10). The CLI supplies the live
implementations.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import db, report
from .config import Config
from .extract import DEFAULT_CACHE_DIR, CandidateSet, build_candidate_set
from .models import RankResult, TargetQuery
from .ranker.judge import Judge
from .ranker.tournament import TournamentResult, run_tournament
from .serp.base import SerpProvider


@dataclass
class RankOutput:
    query: TargetQuery
    run_id: str
    candidate_set: CandidateSet
    tournament: TournamentResult
    rank_results: list[RankResult] = field(default_factory=list)
    report_md: str = ""
    report_path: Path | None = None

    @property
    def total_cost_usd(self) -> float:
        return self.tournament.total_cost_usd


def _to_rank_results(bt_results, query_id: str, run_id: str) -> list[RankResult]:
    return [
        RankResult(
            query_id=query_id,
            run_id=run_id,
            passage_id=b.passage_id,
            bt_strength=b.bt_strength,
            selection_prob=b.selection_prob,
            rank=b.rank,
            ci_low=b.ci_low,
            ci_high=b.ci_high,
        )
        for b in bt_results
    ]


def rank_query(
    query: TargetQuery,
    config: Config,
    *,
    provider: SerpProvider,
    judge: Judge,
    client: httpx.Client,
    run_id: str,
    conn: sqlite3.Connection | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    respect_robots: bool = True,
    budget_cap: float | None = None,
    write_report_file: bool = True,
) -> RankOutput:
    """Run one query end-to-end: build the set, judge, fit, persist, report.

    Raises BudgetExceeded (from the tournament) if the run exceeds its cap.
    """
    from .ranker.bradley_terry import fit_bradley_terry

    if budget_cap is None:
        budget_cap = config.budget.max_usd_per_rank_run

    candidate_set = build_candidate_set(
        query, provider, config, client=client, cache_dir=cache_dir,
        respect_robots=respect_robots,
    )
    passages = candidate_set.passages
    passages_by_id = {p.id: p for p in passages}

    tournament = run_tournament(
        query.id, query.text, passages, judge, config,
        conn=conn, run_id=run_id, budget_cap=budget_cap,
    )

    bt_results = fit_bradley_terry(
        tournament.passage_ids,
        tournament.comparisons,
        bootstrap_resamples=config.tournament.bootstrap_resamples,
    )
    rank_results = _to_rank_results(bt_results, query.id, run_id)

    # Persist passages, matchups, and rank results under this run.
    if conn is not None:
        for p in passages:
            db.save_passage(conn, p)
        for m in tournament.matchups:
            db.save_matchup(conn, run_id, m)
        for r in rank_results:
            db.save_rank_result(conn, r)

    md = report.render_query_report_md(
        query, rank_results, passages_by_id, tournament.matchups,
        run_id=run_id, our_domain=config.our_domain,
        disagreement_rate=tournament.disagreement_rate,
    )
    path = report.write_report(query.id, md) if write_report_file else None

    return RankOutput(
        query=query,
        run_id=run_id,
        candidate_set=candidate_set,
        tournament=tournament,
        rank_results=rank_results,
        report_md=md,
        report_path=path,
    )
