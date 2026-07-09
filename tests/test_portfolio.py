"""Portfolio assembly + judge-vs-reality correlation (§6.7, §12.1)."""

from __future__ import annotations

from datetime import datetime

from citeworthy import db, report
from citeworthy.config import load_config
from citeworthy.models import CitationObservation, Passage, RankResult, Run, TargetQuery
from citeworthy.portfolio import build_portfolio, spearman


def _conn(tmp_path):
    db.init_db(tmp_path / "cw.db")
    return db.connect(tmp_path / "cw.db")


def _seed_query(conn, qid, tier, our_prob, leader_prob, *, engine=None, share=0.0, n_track=0):
    q = TargetQuery(id=qid, text=qid.upper(), tier=tier,
                    our_url=f"https://www.fool.com.au/{qid}/")
    db.upsert_query(conn, q)
    rr = f"rank-{qid}"
    db.create_run(conn, Run(run_id=rr, kind="rank", started_at=datetime(2026, 6, 1), config_hash="h"))
    db.save_passage(conn, Passage(id=f"{qid}-o", url=f"https://www.fool.com.au/{qid}/",
                                  domain="fool.com.au", title="o", chunk_index=0, text="x", is_ours=True))
    db.save_passage(conn, Passage(id=f"{qid}-c", url="https://rival.com/x",
                                  domain="rival.com", title="c", chunk_index=0, text="y", is_ours=False))
    db.save_rank_result(conn, RankResult(query_id=qid, run_id=rr, passage_id=f"{qid}-c",
                                         bt_strength=1.0, selection_prob=leader_prob, rank=1,
                                         ci_low=leader_prob - 0.05, ci_high=leader_prob + 0.05))
    db.save_rank_result(conn, RankResult(query_id=qid, run_id=rr, passage_id=f"{qid}-o",
                                         bt_strength=0.0, selection_prob=our_prob, rank=2,
                                         ci_low=our_prob - 0.05, ci_high=our_prob + 0.05))
    for w in range(n_track):
        trid = f"trk-{qid}-{w}"
        db.create_run(conn, Run(run_id=trid, kind="track",
                                started_at=datetime(2026, 6, 8 + w * 7), config_hash="h"))
        cited = round(share * 10)
        for s in range(10):
            urls = ["https://www.fool.com.au/x"] if s < cited else ["https://rival.com/y"]
            db.save_observation(conn, trid, CitationObservation(
                run_id=trid, query_id=qid, engine=engine, sample_index=s,
                cited_urls=urls, answer_text_hash="h",
                observed_at=datetime(2026, 6, 8 + w * 7)))


# --- spearman --------------------------------------------------------------


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 2], [1, 2]) is None  # too few points
    assert spearman([1, 1, 1], [2, 3, 4]) is None  # no variance -> undefined


# --- portfolio assembly ----------------------------------------------------


def test_portfolio_sorts_and_computes_gap(tmp_path):
    conn = _conn(tmp_path)
    try:
        _seed_query(conn, "money-a", "money", our_prob=0.10, leader_prob=0.80)  # gap 0.70
        _seed_query(conn, "money-b", "money", our_prob=0.50, leader_prob=0.80)  # gap 0.30
        _seed_query(conn, "supp-a", "supporting", our_prob=0.20, leader_prob=0.80)
        rows, _ = build_portfolio(conn, load_config("config.yaml"))

        # money tier first, biggest gap first within tier; supporting last.
        assert [r.query_id for r in rows] == ["money-a", "money-b", "supp-a"]
        assert abs(rows[0].gap - 0.70) < 1e-9
        assert rows[0].our_rank == 2 and abs(rows[0].leader_prob - 0.80) < 1e-9
    finally:
        conn.close()


def test_correlation_insufficient_without_enough_runs(tmp_path):
    conn = _conn(tmp_path)
    try:
        # 3 queries but only 3 track runs each -> below the 4-week threshold.
        _seed_query(conn, "q1", "money", 0.1, 0.8, engine="perplexity", share=0.1, n_track=3)
        _seed_query(conn, "q2", "money", 0.4, 0.8, engine="perplexity", share=0.4, n_track=3)
        _seed_query(conn, "q3", "money", 0.7, 0.8, engine="perplexity", share=0.7, n_track=3)
        _, corrs = build_portfolio(conn, load_config("config.yaml"))
        ppx = next(c for c in corrs if c.engine == "perplexity")
        assert ppx.insufficient is True and ppx.rho is None
    finally:
        conn.close()


def test_correlation_computed_with_enough_data(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    conn = _conn(tmp_path)
    try:
        # selection_prob and observed share both rise together -> strong positive rho.
        _seed_query(conn, "q1", "money", 0.1, 0.8, engine="perplexity", share=0.1, n_track=4)
        _seed_query(conn, "q2", "money", 0.4, 0.8, engine="perplexity", share=0.4, n_track=4)
        _seed_query(conn, "q3", "supporting", 0.7, 0.8, engine="perplexity", share=0.7, n_track=4)
        rows, corrs = build_portfolio(conn, load_config("config.yaml"))
        ppx = next(c for c in corrs if c.engine == "perplexity")
        assert ppx.insufficient is False
        assert ppx.rho == 1.0  # monotone increasing
        assert ppx.min_runs == 4 and ppx.n_queries == 3
        # gemini has no data -> insufficient.
        gem = next(c for c in corrs if c.engine == "gemini_grounded")
        assert gem.insufficient is True

        md, path = report.render_portfolio_report(conn, load_config("config.yaml"))
        assert "# Citeworthy — Portfolio" in md
        assert "## Judge vs reality" in md
        assert "perplexity" in md
    finally:
        conn.close()
