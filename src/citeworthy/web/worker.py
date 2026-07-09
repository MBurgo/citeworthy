"""Background worker: claims queued jobs and runs the engine (Milestone: web).

Runs on a persistent host (Render/Fly/Cloud Run) — NOT a Vercel serverless
function, whose timeout can't accommodate a multi-minute rank. Point it at the
same Postgres database as the API via CITEWORTHY_DATABASE_URL.

    python -m citeworthy.web.worker
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .. import db
from ..config import load_config
from ..db import Store
from .runners import JobRunners, execute_job, live_runners

log = logging.getLogger("citeworthy.worker")

POLL_SECONDS = 5.0


def process_one(store: Store, config, runners: JobRunners) -> bool:
    """Claim and run a single job. Returns True if one was processed."""
    job = db.claim_next_job(store, datetime.now(timezone.utc))
    if job is None:
        return False
    log.info("running job %s (%s, query=%s)", job["job_id"], job["kind"], job.get("query_id"))
    execute_job(store, job, config, runners)
    fresh = db.get_job(store, job["job_id"])
    log.info("job %s -> %s", job["job_id"], fresh["status"] if fresh else "?")
    return True


def run_forever(poll_seconds: float = POLL_SECONDS) -> None:  # pragma: no cover - loop
    from dotenv import load_dotenv

    load_dotenv()  # pick up API keys from a local .env (no-op if absent)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    runners = live_runners()
    store = db.connect()
    db.ensure_schema(store)
    log.info("worker started (dialect=%s); polling every %.0fs", store.dialect, poll_seconds)
    try:
        while True:
            try:
                if not process_one(store, config, runners):
                    time.sleep(poll_seconds)
            except Exception:  # noqa: BLE001 - never let the loop die on one bad job
                log.exception("worker loop error")
                time.sleep(poll_seconds)
    finally:
        store.close()


if __name__ == "__main__":  # pragma: no cover
    run_forever()
