"""Tracker run end-to-end with injected engine clients (no live calls)."""

from __future__ import annotations

from datetime import datetime

import pytest

from citeworthy import db, report
from citeworthy.config import load_config
from citeworthy.models import Run, TargetQuery
from citeworthy.tracker import TrackBudgetExceeded, run_track
from citeworthy.truth.base import EngineSample


class FakeEngine:
    """Cycles through canned samples so citation share is deterministic."""

    def __init__(self, engine: str, samples: list[list[str]]):
        self.engine = engine
        self._samples = samples
        self._i = 0

    def sample_once(self, query_text: str) -> EngineSample:
        urls = self._samples[self._i % len(self._samples)]
        self._i += 1
        return EngineSample(cited_urls=list(urls), answer_text=f"answer {self._i}")


def _query():
    return TargetQuery(id="asx-200", text="ASX 200", tier="money",
                       our_url="https://www.fool.com.au/asx-200-explained/")


def _conn(tmp_path):
    db.init_db(tmp_path / "cw.db")
    return db.connect(tmp_path / "cw.db")


def test_run_track_persists_and_summarises(tmp_path):
    cfg = load_config("config.yaml")
    conn = _conn(tmp_path)
    try:
        db.upsert_query(conn, _query())
        db.create_run(conn, Run(run_id="t1", kind="track",
                                started_at=datetime(2026, 7, 8), config_hash="h"))
        # fool cited in 2 of every 4 samples on perplexity.
        engine = FakeEngine("perplexity", [
            ["https://www.fool.com.au/x", "https://marketindex.com.au/asx200"],
            ["https://marketindex.com.au/asx200"],
            ["https://www.fool.com.au/y"],
            ["https://commsec.com.au/z"],
        ])
        result = run_track([_query()], [engine], cfg, run_id="t1", conn=conn, samples=8)

        assert result.n_samples == 8
        assert result.n_failures == 0
        assert result.total_cost_usd == pytest.approx(8 * cfg.truth_pricing["perplexity"])

        stat = result.stats[0]
        assert stat.our_share.successes == 4  # fool in 4 of 8 samples
        assert abs(stat.our_share.share - 0.5) < 1e-9

        # Observations were persisted (never overwrite — §6.6).
        stored = db.get_observations(conn, query_id="asx-200", engine="perplexity")
        assert len(stored) == 8
        groups = db.observations_by_run(conn, "asx-200", "perplexity")
        assert len(groups) == 1 and len(groups[0]) == 8
    finally:
        conn.close()


def test_run_track_survives_flaky_calls(tmp_path):
    cfg = load_config("config.yaml")

    class FlakyEngine:
        engine = "perplexity"

        def __init__(self):
            self.n = 0

        def sample_once(self, q):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("boom")
            return EngineSample(["https://www.fool.com.au/x"], "a")

    result = run_track([_query()], [FlakyEngine()], cfg, run_id="t", samples=3)
    assert result.n_failures == 1
    assert result.n_samples == 2
    assert any("boom" in w for w in result.warnings)


def test_run_track_budget_abort(tmp_path):
    cfg = load_config("config.yaml")
    engine = FakeEngine("perplexity", [["https://fool.com.au/x"]])
    with pytest.raises(TrackBudgetExceeded):
        run_track([_query()], [engine], cfg, run_id="t", samples=100, budget_cap=0.001)


def test_report_includes_citation_share(tmp_path, monkeypatch):
    cfg = load_config("config.yaml")
    conn = _conn(tmp_path)
    try:
        db.upsert_query(conn, _query())
        # A rank run so the report has a ranked table to attach to.
        db.create_run(conn, Run(run_id="rank1", kind="rank",
                                started_at=datetime(2026, 7, 1), config_hash="h"))
        from citeworthy.models import Passage, RankResult
        p = Passage(id="p1", url="https://www.fool.com.au/asx-200-explained/",
                    domain="fool.com.au", title="T", chunk_index=0, text="x", is_ours=True)
        db.save_passage(conn, p)
        db.save_rank_result(conn, RankResult(query_id="asx-200", run_id="rank1", passage_id="p1",
                                             bt_strength=0.0, selection_prob=1.0, rank=1,
                                             ci_low=1.0, ci_high=1.0))
        # A track run with observations.
        db.create_run(conn, Run(run_id="t1", kind="track",
                                started_at=datetime(2026, 7, 8), config_hash="h"))
        engine = FakeEngine("perplexity", [["https://www.fool.com.au/x"], ["https://rival.com/y"]])
        run_track([_query()], [engine], cfg, run_id="t1", conn=conn, samples=6)

        monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
        md, _ = report.render_query_report(conn, "asx-200")
        assert "## Citation share (v3)" in md
        assert "perplexity" in md
        assert "our share" in md
        # single run -> insufficient-data guard, not a trend claim.
        assert "Insufficient data" in md
    finally:
        conn.close()
