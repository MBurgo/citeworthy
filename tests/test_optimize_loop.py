"""Edit-loop end-to-end with injected transports + stub checker (§10, CLAUDE.md)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from citeworthy import db
from citeworthy.config import load_config
from citeworthy.editor.compliance import ComplianceResult
from citeworthy.editor.loop import OptimizeError, run_optimize
from citeworthy.models import Matchup, Passage, RankResult, Run, TargetQuery
from citeworthy.ranker.judge import Judge, TransportResponse

QUERY = TargetQuery(id="asx-200", text="ASX 200", tier="money",
                    our_url="https://www.fool.com.au/asx/")

MARKER = "STRONGER"
IMPROVED = f"The ASX 200 is Australia's benchmark of 200 companies. {MARKER} lead answer with 200 firms."


# --- stub compliance checkers (per CLAUDE.md: loop tests use a stub) --------


class PassAll:
    def check(self, text, *, original=None):
        return ComplianceResult(passed=True)


class FailAll:
    def check(self, text, *, original=None):
        return ComplianceResult(passed=False, violations=["stubbed rejection"])


# --- canned transports -----------------------------------------------------


class EditorTransport:
    def __init__(self, text=IMPROVED):
        self.text = text

    def complete(self, *, system, user, model, temperature, max_tokens):
        return TransportResponse(self.text, input_tokens=500, output_tokens=200)


class MarkerJudge:
    """Prefers the passage containing MARKER; ties break to slot A."""

    def complete(self, *, system, user, model, temperature, max_tokens):
        idx = user.find("Passage B (source:")
        a_has = MARKER in user[:idx]
        b_has = MARKER in user[idx:]
        winner = "A" if a_has and not b_has else "B" if b_has and not a_has else "A"
        return TransportResponse(
            f'{{"winner":"{winner}","confidence":"clear","reason_code":"buried_answer","reason":"x"}}',
            1400, 120,
        )


class PositionJudge:
    """Always picks slot A — pure position bias, no real signal."""

    def complete(self, *, system, user, model, temperature, max_tokens):
        return TransportResponse(
            '{"winner":"A","confidence":"slight","reason_code":"other","reason":"slot A"}',
            1000, 100,
        )


def _judge(transport):
    return Judge(transport, model="claude-haiku-4-5-20251001", temperature=0.3)


# --- fixtures --------------------------------------------------------------


def _seed_rank_run(conn):
    db.upsert_query(conn, QUERY)
    db.create_run(conn, Run(run_id="rank1", kind="rank",
                            started_at=datetime(2026, 7, 1), config_hash="h"))
    orig = Passage(id="ours0", url="https://www.fool.com.au/asx/", domain="fool.com.au",
                   title="Fool ASX", chunk_index=0,
                   text="The ASX 200 is an index. It has 200 companies.", is_ours=True)
    c1 = Passage(id="c1", url="https://marketindex.com.au/asx200", domain="marketindex.com.au",
                 title="MI", chunk_index=0, text="Market Index covers the ASX 200 with 200 firms.", is_ours=False)
    c2 = Passage(id="c2", url="https://commsec.com.au/asx", domain="commsec.com.au",
                 title="CS", chunk_index=0, text="CommSec explains the ASX 200.", is_ours=False)
    for p in (orig, c1, c2):
        db.save_passage(conn, p)
    def _rr(pid, strength, prob, rank, lo, hi):
        return RankResult(query_id="asx-200", run_id="rank1", passage_id=pid,
                          bt_strength=strength, selection_prob=prob, rank=rank,
                          ci_low=lo, ci_high=hi)

    db.save_rank_result(conn, _rr("c1", 2.0, 0.5, 1, 0.4, 0.6))
    db.save_rank_result(conn, _rr("c2", 1.0, 0.3, 2, 0.2, 0.4))
    db.save_rank_result(conn, _rr("ours0", -1.0, 0.2, 3, 0.1, 0.3))
    for i, rc in enumerate(["buried_answer", "buried_answer", "too_generic", "no_evidence"]):
        db.save_matchup(conn, "rank1", Matchup(
            query_id="asx-200", passage_a="ours0", passage_b="c1", winner="c1",
            reason_code=rc, reason_text="lacks specifics", judge_model="m",
            position_order="ab", sample_index=i))
    return orig


def _conn(tmp_path):
    db.init_db(tmp_path / "cw.db")
    return db.connect(tmp_path / "cw.db")


def _optimize(conn, judge_transport, checker):
    cfg = load_config("config.yaml")
    db.create_run(conn, Run(run_id="opt1", kind="optimize",
                            started_at=datetime(2026, 7, 8), config_hash="h"))
    return run_optimize(
        QUERY, cfg, judge=_judge(judge_transport), editor_transport=EditorTransport(),
        checker=checker, conn=conn, run_id="opt1", editor_model="claude-sonnet-4-6",
    )


# --- tests -----------------------------------------------------------------


def test_accepts_improved_variant(tmp_path):
    conn = _conn(tmp_path)
    try:
        _seed_rank_run(conn)
        out = _optimize(conn, MarkerJudge(), PassAll())

        assert out.accepted is True
        assert MARKER in out.final.text
        assert out.final.variant_label in {"buried_answer", "too_generic", "no_evidence"}
        assert out.iterations[0].accepted is True
        assert out.diff  # non-empty diff vs original

        # Report structure (§6.5 output).
        for section in ("## Outcome", "## Score trajectory", "## Compliance gate",
                        "## Recommended passage", "Accepted a variant"):
            assert section in out.report_md

        # Variant passages persisted under the optimize run.
        results = db.get_rank_results(conn, "opt1", "asx-200")
        passages = db.get_passages(conn, [r.passage_id for r in results])
        assert any(p.variant_of == "ours0" for p in passages.values())
        # Editor spend logged for `costs`.
        assert any(row["model"] == "claude-sonnet-4-6" for row in db.cost_summary(conn))
    finally:
        conn.close()


def test_compliance_gate_discards_all(tmp_path):
    conn = _conn(tmp_path)
    try:
        _seed_rank_run(conn)
        out = _optimize(conn, MarkerJudge(), FailAll())

        assert out.accepted is False
        assert out.final.id == "ours0"  # unchanged
        assert len(out.failed_variants) >= 1
        assert "discarded" in out.report_md
        # No variant reached the tournament (nothing persisted for the opt run).
        assert db.get_rank_results(conn, "opt1", "asx-200") == []
    finally:
        conn.close()


def test_no_significant_improvement_keeps_original(tmp_path):
    conn = _conn(tmp_path)
    try:
        _seed_rank_run(conn)
        # PositionJudge gives no real signal, so no variant beats the original with
        # non-overlapping CIs.
        out = _optimize(conn, PositionJudge(), PassAll())
        assert out.accepted is False
        assert out.final.id == "ours0"
        assert "No significant improvement" in out.report_md
    finally:
        conn.close()


def test_requires_prior_rank_run(tmp_path):
    conn = _conn(tmp_path)
    try:
        db.upsert_query(conn, QUERY)
        db.create_run(conn, Run(run_id="opt1", kind="optimize",
                                started_at=datetime(2026, 7, 8), config_hash="h"))
        cfg = load_config("config.yaml")
        with pytest.raises(OptimizeError):
            run_optimize(QUERY, cfg, judge=_judge(MarkerJudge()),
                         editor_transport=EditorTransport(), checker=PassAll(),
                         conn=conn, run_id="opt1")
    finally:
        conn.close()
