"""Job execution — the bridge between a queued job and the engine (Milestone: web).

`execute_job` owns the run row + job status transitions; the actual engine work is
delegated to three runner callables so the worker loop is testable with stubs that
make no live API calls. `live_runners()` wires the real SerpApi/Anthropic/
Perplexity/Gemini clients from environment keys.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .. import db
from ..config import Config
from ..db import Store
from ..models import Run

# (store, config, run_id, query_id) -> report_query_id | None
Runner = Callable[[Store, Config, str, str | None], str | None]


@dataclass
class JobRunners:
    rank: Runner
    optimize: Runner
    track: Runner


def _now() -> datetime:
    return datetime.now(timezone.utc)


def execute_job(
    store: Store,
    job: dict,
    config: Config,
    runners: JobRunners,
    *,
    now: datetime | None = None,
) -> None:
    """Run one claimed job: create its run row, dispatch to the engine, and record
    the outcome on the job. Never raises — failures are recorded as status=error."""
    now = now or _now()
    job_id = job["job_id"]
    kind = job["kind"]
    query_id = job.get("query_id")
    run_id = uuid.uuid4().hex[:12]

    db.create_run(store, Run(
        run_id=run_id, kind=kind, started_at=now, config_hash=config.config_hash(),
    ))
    try:
        if kind == "rank":
            report_qid = runners.rank(store, config, run_id, query_id)
        elif kind == "optimize":
            report_qid = runners.optimize(store, config, run_id, query_id)
        elif kind == "track":
            report_qid = runners.track(store, config, run_id, query_id)
        else:
            raise ValueError(f"unknown job kind: {kind}")
        db.finish_job(store, job_id, "done", _now(),
                      run_id=run_id, report_query_id=report_qid)
    except Exception as exc:  # noqa: BLE001 - surface any failure on the job, not the worker
        db.finish_job(store, job_id, "error", _now(), run_id=run_id, error=str(exc))


# --- live runners ----------------------------------------------------------


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def _live_rank(store: Store, config: Config, run_id: str, query_id: str | None) -> str:
    import httpx

    from ..extract import DEFAULT_UA
    from ..pipeline import rank_query
    from ..ranker.judge import AnthropicTransport, Judge
    from ..serp.serpapi import SerpApiProvider

    query = db.get_query(store, query_id)
    if query is None:
        raise RuntimeError(f"no such query: {query_id}")

    provider = SerpApiProvider(_require("SERPAPI_API_KEY"))
    judge = Judge(
        AnthropicTransport(_require("ANTHROPIC_API_KEY")),
        model=config.judge.model, temperature=config.judge.temperature,
        show_domains=config.judge.show_domains,
    )
    client = httpx.Client(timeout=15.0, headers={"User-Agent": DEFAULT_UA}, follow_redirects=True)
    try:
        rank_query(query, config, provider=provider, judge=judge, client=client,
                   run_id=run_id, conn=store, write_report_file=False)
    finally:
        client.close()
    return query_id


def _live_optimize(store: Store, config: Config, run_id: str, query_id: str | None) -> str:
    from ..editor.compliance import RuleComplianceChecker
    from ..editor.loop import run_optimize
    from ..ranker.judge import AnthropicTransport, Judge

    query = db.get_query(store, query_id)
    if query is None:
        raise RuntimeError(f"no such query: {query_id}")
    transport = AnthropicTransport(_require("ANTHROPIC_API_KEY"))
    judge = Judge(transport, model=config.judge.model,
                  temperature=config.judge.temperature, show_domains=config.judge.show_domains)
    run_optimize(query, config, judge=judge, editor_transport=transport,
                 checker=RuleComplianceChecker(), conn=store, run_id=run_id,
                 editor_model=config.editor.model)
    return query_id


def _live_track(store: Store, config: Config, run_id: str, query_id: str | None) -> None:
    from ..tracker import run_track
    from ..truth.gemini import GeminiClient
    from ..truth.perplexity import PerplexityClient

    queries = db.list_queries(store)
    clients = []
    if os.environ.get("PERPLEXITY_API_KEY"):
        clients.append(PerplexityClient(os.environ["PERPLEXITY_API_KEY"],
                                        model=config.truth.perplexity_model))
    if os.environ.get("GEMINI_API_KEY"):
        clients.append(GeminiClient(os.environ["GEMINI_API_KEY"],
                                    model=config.truth.gemini_model))
    if not clients:
        raise RuntimeError("no tracker engines available (set PERPLEXITY_API_KEY / GEMINI_API_KEY)")
    run_track(queries, clients, config, run_id=run_id, conn=store,
              budget_cap=config.budget.max_usd_per_track_run)
    return None


def live_runners() -> JobRunners:
    return JobRunners(rank=_live_rank, optimize=_live_optimize, track=_live_track)
