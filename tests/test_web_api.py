"""FastAPI endpoints via TestClient, backed by a temp SQLite db (no live calls)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from citeworthy import db
from citeworthy.models import Passage, RankResult, Run


@pytest.fixture
def client(tmp_path, monkeypatch):
    dbfile = tmp_path / "web.db"
    # Any non-postgres value routes db.connect() to this SQLite file.
    monkeypatch.setenv("CITEWORTHY_DATABASE_URL", str(dbfile))
    from citeworthy.web.app import app

    with TestClient(app) as c:
        yield c, dbfile


def test_dashboard_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "Citeworthy" in r.text


def test_query_crud(client):
    c, _ = client
    r = c.post("/api/queries", json={"id": "asx-200", "text": "ASX 200", "tier": "money"})
    assert r.status_code == 201
    assert any(q["id"] == "asx-200" for q in c.get("/api/queries").json())
    assert c.delete("/api/queries/asx-200").status_code == 200
    assert c.get("/api/queries").json() == []
    assert c.delete("/api/queries/asx-200").status_code == 404


def test_job_enqueue_validation(client):
    c, _ = client
    # rank needs a query_id
    assert c.post("/api/jobs", json={"kind": "rank"}).status_code == 400
    # unknown query
    assert c.post("/api/jobs", json={"kind": "rank", "query_id": "nope"}).status_code == 404

    c.post("/api/queries", json={"id": "asx-200", "text": "ASX 200", "tier": "money"})
    r = c.post("/api/jobs", json={"kind": "rank", "query_id": "asx-200"})
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "queued" and job["kind"] == "rank"

    assert c.get(f"/api/jobs/{job['job_id']}").json()["status"] == "queued"
    assert c.get("/api/jobs/missing").status_code == 404
    # track needs no query_id
    assert c.post("/api/jobs", json={"kind": "track"}).status_code == 201


def test_report_and_portfolio_and_costs(client):
    c, dbfile = client
    # 404 before any rank run.
    c.post("/api/queries", json={"id": "asx-200", "text": "ASX 200", "tier": "money",
                                 "our_url": "https://www.fool.com.au/asx/"})
    assert c.get("/api/report/asx-200").status_code == 404

    # Seed a rank run directly, then the report renders.
    store = db.connect(str(dbfile))
    try:
        db.create_run(store, Run(run_id="r1", kind="rank",
                                 started_at=datetime(2026, 7, 1, tzinfo=timezone.utc), config_hash="h"))
        db.save_passage(store, Passage(id="p1", url="https://www.fool.com.au/asx/",
                                       domain="fool.com.au", title="T", chunk_index=0,
                                       text="x", is_ours=True))
        db.save_rank_result(store, RankResult(query_id="asx-200", run_id="r1", passage_id="p1",
                                              bt_strength=0.0, selection_prob=1.0, rank=1,
                                              ci_low=1.0, ci_high=1.0))
    finally:
        store.close()

    rep = c.get("/api/report/asx-200")
    assert rep.status_code == 200
    assert "# Citeworthy — ASX 200" in rep.json()["markdown"]
    assert "<table>" in rep.json()["html"]  # markdown rendered to HTML

    assert c.get("/api/portfolio").status_code == 200
    assert "Portfolio" in c.get("/api/portfolio").json()["markdown"]

    costs = c.get("/api/costs").json()
    assert "by_model" in costs and "total_usd" in costs
