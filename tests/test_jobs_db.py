"""Job-queue repository functions (SQLite backend)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from citeworthy import db


def _store(tmp_path):
    return db.init_db(tmp_path / "j.db") and db.connect(tmp_path / "j.db")


def test_enqueue_get_list(tmp_path):
    store = _store(tmp_path)
    try:
        t0 = datetime(2026, 7, 8, tzinfo=timezone.utc)
        db.enqueue_job(store, "j1", "rank", "asx-200", t0)
        db.enqueue_job(store, "j2", "track", None, t0 + timedelta(seconds=1))

        job = db.get_job(store, "j1")
        assert job["status"] == "queued" and job["kind"] == "rank" and job["query_id"] == "asx-200"
        assert db.get_job(store, "nope") is None

        jobs = db.list_jobs(store)
        assert [j["job_id"] for j in jobs] == ["j2", "j1"]  # newest first
    finally:
        store.close()


def test_claim_and_finish(tmp_path):
    store = _store(tmp_path)
    try:
        t0 = datetime(2026, 7, 8, tzinfo=timezone.utc)
        db.enqueue_job(store, "j1", "rank", "asx-200", t0)
        db.enqueue_job(store, "j2", "rank", "other", t0 + timedelta(seconds=1))

        first = db.claim_next_job(store, t0 + timedelta(seconds=2))
        assert first["job_id"] == "j1" and first["status"] == "running"

        second = db.claim_next_job(store, t0 + timedelta(seconds=3))
        assert second["job_id"] == "j2"  # oldest remaining queued

        assert db.claim_next_job(store, t0 + timedelta(seconds=4)) is None  # queue empty

        db.finish_job(store, "j1", "done", t0 + timedelta(seconds=5),
                      run_id="r1", report_query_id="asx-200")
        done = db.get_job(store, "j1")
        assert done["status"] == "done" and done["run_id"] == "r1"
        assert done["report_query_id"] == "asx-200"

        db.finish_job(store, "j2", "error", t0 + timedelta(seconds=6), error="boom")
        assert db.get_job(store, "j2")["error"] == "boom"
    finally:
        store.close()
