"""Milestone 1 smoke tests: config loads, db initialises, query CRUD works.

No live API calls (CLAUDE.md). Later milestones add fixtures and property tests
for the chunker, tournament, Bradley–Terry, and share stats (§10).
"""

from __future__ import annotations

from datetime import datetime

from citeworthy import db
from citeworthy.config import Config, load_config
from citeworthy.models import Run, TargetQuery


def test_config_defaults_and_hash():
    cfg = Config()
    assert cfg.our_domain == "fool.com.au"
    assert cfg.judge.model == "claude-haiku-4-5-20251001"
    # config_hash is stable and short.
    assert cfg.config_hash() == cfg.config_hash()
    assert len(cfg.config_hash()) == 16


def test_load_repo_config():
    cfg = load_config("config.yaml")
    assert cfg.serp.provider == "serpapi"
    assert cfg.tournament.round_robin_max == 14
    assert "claude-haiku-4-5-20251001" in cfg.pricing


def test_db_init_and_query_crud(tmp_path):
    db_path = tmp_path / "citeworthy.db"
    db.init_db(db_path)
    conn = db.connect(db_path)
    try:
        q = TargetQuery(
            id="best-asx-dividend-shares",
            text="best ASX dividend shares to buy",
            tier="money",
            our_url="https://www.fool.com.au/best-dividend-shares/",
        )
        db.upsert_query(conn, q)
        rows = db.list_queries(conn)
        assert len(rows) == 1
        assert rows[0].id == "best-asx-dividend-shares"

        # Upsert is idempotent on id.
        db.upsert_query(conn, q)
        assert len(db.list_queries(conn)) == 1

        assert db.remove_query(conn, "best-asx-dividend-shares") is True
        assert db.list_queries(conn) == []
        assert db.remove_query(conn, "nope") is False
    finally:
        conn.close()


def test_run_and_cost_logging(tmp_path):
    db_path = tmp_path / "citeworthy.db"
    db.init_db(db_path)
    conn = db.connect(db_path)
    try:
        run = Run(
            run_id="r1",
            kind="rank",
            started_at=datetime(2026, 7, 8, 12, 0, 0),
            config_hash="abc123",
            git_sha="deadbeef",
        )
        db.create_run(conn, run)
        db.log_judge_call(
            conn,
            run_id="r1",
            model="claude-haiku-4-5-20251001",
            input_tokens=1400,
            output_tokens=120,
            est_usd=0.0016,
            created_at=datetime(2026, 7, 8, 12, 0, 1),
        )
        summary = db.cost_summary(conn)
        assert len(summary) == 1
        assert summary[0]["calls"] == 1
        assert summary[0]["input_tokens"] == 1400
    finally:
        conn.close()
