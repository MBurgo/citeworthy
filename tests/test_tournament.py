"""Tournament scheduling + execution on synthetic data (no live calls)."""

from __future__ import annotations

import re
from itertools import combinations

import pytest

from citeworthy import db
from citeworthy.config import load_config
from citeworthy.models import Passage
from citeworthy.ranker.bradley_terry import fit_bradley_terry
from citeworthy.ranker.judge import Judge, TransportResponse
from citeworthy.ranker.tournament import (
    BudgetExceeded,
    run_tournament,
    schedule_pairs,
)

STR_RE = re.compile(r"STR=(\d+)")


def _passages(strengths: list[int]) -> list[Passage]:
    return [
        Passage(
            id=f"p{i}",
            url=f"https://d{i}.com/page",
            domain=f"d{i}.com",
            title=f"Title {i}",
            chunk_index=0,
            text=f"STR={s} — passage {i} body text.",
        )
        for i, s in enumerate(strengths)
    ]


class StrengthTransport:
    """Deterministic judge: picks whichever passage has the higher planted STR,
    regardless of A/B position (so no position bias)."""

    def complete(self, *, system, user, model, temperature, max_tokens):
        a_str, b_str = (int(x) for x in STR_RE.findall(user)[:2])
        winner = "A" if a_str >= b_str else "B"
        text = f'{{"winner":"{winner}","confidence":"clear","reason_code":"too_generic","reason":"weaker"}}'
        return TransportResponse(text, input_tokens=1400, output_tokens=120)


class PositionBiasTransport:
    """Always picks whichever passage is shown in slot A (pure position bias)."""

    def complete(self, *, system, user, model, temperature, max_tokens):
        return TransportResponse(
            '{"winner":"A","confidence":"slight","reason_code":"other","reason":"slot A"}',
            input_tokens=1000,
            output_tokens=100,
        )


def _judge(transport):
    return Judge(transport, model="claude-haiku-4-5-20251001", temperature=0.3)


# --- scheduling ------------------------------------------------------------


def test_round_robin_pair_count():
    passages = _passages([1] * 6)
    pairs = schedule_pairs(passages, round_robin_max=14)
    assert len(pairs) == len(list(combinations(range(6), 2)))  # C(6,2) = 15
    assert all(a != b for a, b in pairs)


def test_swiss_for_large_sets():
    passages = _passages(list(range(18, 0, -1)))  # 18 passages, best-first
    pairs = schedule_pairs(passages, round_robin_max=14)
    # Fewer than a full round-robin (C(18,2)=153), no self-pairs, all unique.
    assert 0 < len(pairs) < len(list(combinations(range(18), 2)))
    assert all(a != b for a, b in pairs)
    assert len({frozenset(p) for p in pairs}) == len(pairs)
    # Top-8 seeds are fully connected (round-robin among them).
    ids = [p.id for p in passages]
    top8 = set(ids[:8])
    pairset = {frozenset(p) for p in pairs}
    for a, b in combinations(ids[:8], 2):
        assert frozenset((a, b)) in pairset
    assert top8  # sanity


# --- execution -------------------------------------------------------------


def test_recovers_planted_order():
    strengths = [50, 40, 30, 20, 10]
    passages = _passages(strengths)
    cfg = load_config("config.yaml")
    result = run_tournament("q", "the query", passages, _judge(StrengthTransport()), cfg)

    # C(5,2)=10 pairs * 2 orders * samples_per_order(2) = 40 calls.
    assert result.n_calls == 40
    assert result.disagreement_rate == 0.0
    assert result.weak_signal is False
    assert result.total_cost_usd > 0

    bt = {r.passage_id: r for r in fit_bradley_terry(result.passage_ids, result.comparisons, bootstrap_resamples=0)}
    assert [bt[f"p{i}"].rank for i in range(5)] == [1, 2, 3, 4, 5]


def test_position_bias_flagged():
    passages = _passages([10, 20, 30])
    cfg = load_config("config.yaml")
    result = run_tournament("q", "the query", passages, _judge(PositionBiasTransport()), cfg)
    # Every pair disagrees between A/B and B/A orders.
    assert result.disagreement_rate == 1.0
    assert result.weak_signal is True


def test_budget_abort():
    passages = _passages([10, 20, 30, 40])
    cfg = load_config("config.yaml")
    with pytest.raises(BudgetExceeded):
        run_tournament(
            "q", "the query", passages, _judge(StrengthTransport()), cfg, budget_cap=0.0001
        )


def test_logs_calls_to_db(tmp_path):
    passages = _passages([30, 20, 10])
    cfg = load_config("config.yaml")
    db_path = tmp_path / "t.db"
    db.init_db(db_path)
    conn = db.connect(db_path)
    try:
        from datetime import datetime

        from citeworthy.models import Run

        db.create_run(conn, Run(run_id="r1", kind="rank", started_at=datetime(2026, 7, 8), config_hash="h"))
        result = run_tournament(
            "q", "the query", passages, _judge(StrengthTransport()), cfg, conn=conn, run_id="r1"
        )
        summary = db.cost_summary(conn)
        assert summary and summary[0]["calls"] == result.n_calls
        assert summary[0]["model"] == "claude-haiku-4-5-20251001"
    finally:
        conn.close()
