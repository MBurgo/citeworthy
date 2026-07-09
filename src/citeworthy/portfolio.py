"""Portfolio assembly + judge-vs-reality correlation (Milestone 7).

Builds the cross-query portfolio (§6.7 `report --all`) and, once enough tracker
data exists, the Spearman rank correlation between the ranker's selection_prob and
the observed citation share per query (§12.1). A weak correlation is a signal to
iterate the judge prompt, not the architecture.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from . import db
from .config import Config
from .truth.share import domain_share, rolling_window, to_domain

# §12.1: trust the correlation only once ~4 weeks of tracker data exist.
MIN_RUNS_FOR_CORRELATION = 4
MIN_QUERIES_FOR_CORRELATION = 3
WEAK_CORRELATION = 0.3  # |rho| below this → iterate the judge prompt first


@dataclass
class PortfolioRow:
    query_id: str
    text: str
    tier: str
    our_rank: int | None = None
    our_prob: float | None = None
    leader_prob: float | None = None
    gap: float | None = None  # leader_prob - our_prob (0 if we lead)
    n_passages: int = 0
    our_share: dict[str, float | None] = field(default_factory=dict)  # engine -> rolling share


@dataclass
class Correlation:
    engine: str
    n_queries: int
    min_runs: int
    rho: float | None = None
    insufficient: bool = True

    @property
    def weak(self) -> bool:
        return self.rho is not None and abs(self.rho) < WEAK_CORRELATION


# --- rank correlation (Spearman) -------------------------------------------


def _rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(len(arr), dtype=int)
    inv[sorter] = np.arange(len(arr))
    arr_sorted = arr[sorter]
    ranks = np.arange(1, len(arr) + 1, dtype=float)
    i = 0
    while i < len(arr):  # average tied ranks
        j = i
        while j + 1 < len(arr) and arr_sorted[j + 1] == arr_sorted[i]:
            j += 1
        if j > i:
            ranks[i : j + 1] = ranks[i : j + 1].mean()
        i = j + 1
    return ranks[inv]


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0:
        return None  # no variance → correlation undefined
    return float((xm * ym).sum() / denom)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation (Pearson on ranks). None if undefined."""
    if len(xs) != len(ys) or len(xs) < MIN_QUERIES_FOR_CORRELATION:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


# --- assembly --------------------------------------------------------------


def _rank_snapshot(conn: sqlite3.Connection, query, our_domain: str):
    """(our_rank, our_prob, leader_prob, gap, n_passages) from the latest rank run."""
    run_id = db.latest_rank_run_id(conn, query.id)
    if run_id is None:
        return None, None, None, None, 0
    results = db.get_rank_results(conn, run_id, query.id)
    if not results:
        return None, None, None, None, 0
    passages = db.get_passages(conn, [r.passage_id for r in results])
    leader = min(results, key=lambda r: r.rank)
    ours = [r for r in results if passages[r.passage_id].is_ours]
    if not ours:
        return None, None, leader.selection_prob, None, len(results)
    best = min(ours, key=lambda r: r.rank)
    gap = 0.0 if best.passage_id == leader.passage_id else leader.selection_prob - best.selection_prob
    return best.rank, best.selection_prob, leader.selection_prob, gap, len(results)


def _rolling_share(conn: sqlite3.Connection, query_id: str, engine: str, our_domain: str):
    """(rolling share of our domain, n_runs) for an engine, or (None, 0)."""
    runs = db.observations_by_run(conn, query_id, engine)
    if not runs:
        return None, 0
    stat = domain_share(rolling_window(runs), our_domain)
    return stat.share, len(runs)


_TIER_ORDER = {"money": 0, "supporting": 1}


def build_portfolio(
    conn: sqlite3.Connection, config: Config
) -> tuple[list[PortfolioRow], list[Correlation]]:
    """Assemble portfolio rows (sorted by tier, then gap-to-leader) and the
    per-engine judge-vs-reality correlations (§6.7, §12.1)."""
    our_domain = to_domain(config.our_domain)
    engines = list(config.truth.engines)

    rows: list[PortfolioRow] = []
    # engine -> list of (our_prob, our_share, n_runs) for correlation
    corr_data: dict[str, list[tuple[float, float, int]]] = {e: [] for e in engines}

    for query in db.list_queries(conn):
        rank, prob, leader_prob, gap, n = _rank_snapshot(conn, query, our_domain)
        shares: dict[str, float | None] = {}
        for engine in engines:
            share, n_runs = _rolling_share(conn, query.id, engine, our_domain)
            shares[engine] = share
            if prob is not None and share is not None and n_runs > 0:
                corr_data[engine].append((prob, share, n_runs))
        rows.append(PortfolioRow(
            query_id=query.id, text=query.text, tier=query.tier,
            our_rank=rank, our_prob=prob, leader_prob=leader_prob, gap=gap,
            n_passages=n, our_share=shares,
        ))

    # Sort: tier (money first), then biggest gap first; unranked queries last.
    rows.sort(key=lambda r: (
        _TIER_ORDER.get(r.tier, 99),
        r.our_rank is None,  # ranked before unranked
        -(r.gap if r.gap is not None else -1.0),
    ))

    correlations: list[Correlation] = []
    for engine in engines:
        data = corr_data[engine]
        n_queries = len(data)
        min_runs = min((d[2] for d in data), default=0)
        insufficient = n_queries < MIN_QUERIES_FOR_CORRELATION or min_runs < MIN_RUNS_FOR_CORRELATION
        rho = None if insufficient else spearman([d[0] for d in data], [d[1] for d in data])
        correlations.append(Correlation(
            engine=engine, n_queries=n_queries, min_runs=min_runs,
            rho=rho, insufficient=insufficient,
        ))
    return rows, correlations
