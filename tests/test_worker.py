"""Worker job execution with stub runners (no live API calls)."""

from __future__ import annotations

from datetime import datetime, timezone

from citeworthy import db
from citeworthy.config import load_config
from citeworthy.models import Passage, RankResult
from citeworthy.web.runners import JobRunners, execute_job
from citeworthy.web.worker import process_one


def _store(tmp_path):
    db.init_db(tmp_path / "w.db")
    return db.connect(tmp_path / "w.db")


def _stub_runners(recorder):
    def rank(store, config, run_id, query_id):
        # A runner writes real rows; assert it receives the created run_id.
        recorder["rank_run_id"] = run_id
        db.save_passage(store, Passage(id=f"{run_id}-p", url="https://www.fool.com.au/x",
                                       domain="fool.com.au", title="t", chunk_index=0,
                                       text="x", is_ours=True))
        db.save_rank_result(store, RankResult(query_id=query_id, run_id=run_id,
                                              passage_id=f"{run_id}-p", bt_strength=0.0,
                                              selection_prob=1.0, rank=1, ci_low=1.0, ci_high=1.0))
        return query_id

    def optimize(store, config, run_id, query_id):
        return query_id

    def track(store, config, run_id, query_id):
        recorder["track_ran"] = True
        return None

    return JobRunners(rank=rank, optimize=optimize, track=track)


def test_execute_rank_job_success(tmp_path):
    store = _store(tmp_path)
    try:
        db.enqueue_job(store, "j1", "rank", "asx-200", datetime.now(timezone.utc))
        job = db.claim_next_job(store, datetime.now(timezone.utc))
        rec = {}
        execute_job(store, job, load_config("config.yaml"), _stub_runners(rec))

        done = db.get_job(store, "j1")
        assert done["status"] == "done"
        assert done["report_query_id"] == "asx-200"
        assert done["run_id"] == rec["rank_run_id"]
        # The run row and rank result were persisted under the job's run.
        assert db.get_rank_results(store, done["run_id"], "asx-200")
    finally:
        store.close()


def test_execute_job_records_error(tmp_path):
    store = _store(tmp_path)
    try:
        db.enqueue_job(store, "j1", "rank", "asx-200", datetime.now(timezone.utc))
        job = db.claim_next_job(store, datetime.now(timezone.utc))

        def boom(store, config, run_id, query_id):
            raise RuntimeError("kaboom")

        runners = JobRunners(rank=boom, optimize=boom, track=boom)
        execute_job(store, job, load_config("config.yaml"), runners)

        failed = db.get_job(store, "j1")
        assert failed["status"] == "error" and "kaboom" in failed["error"]
    finally:
        store.close()


def test_process_one_claims_and_runs(tmp_path):
    store = _store(tmp_path)
    try:
        db.enqueue_job(store, "j1", "track", None, datetime.now(timezone.utc))
        rec = {}
        assert process_one(store, load_config("config.yaml"), _stub_runners(rec)) is True
        assert rec.get("track_ran") is True
        assert db.get_job(store, "j1")["status"] == "done"
        # Empty queue -> nothing processed.
        assert process_one(store, load_config("config.yaml"), _stub_runners(rec)) is False
    finally:
        store.close()
