"""Ground-truth tracker orchestration (Milestone 5).

One tracking pass: for each tracked query, sample each engine N times, store every
observation (never overwrite — §6.6), and summarise this run's citation shares.
Engine clients are injected so the flow is testable with no live calls (§10). The
run aborts if the per-run budget cap is exceeded (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import db
from .config import Config
from .models import CitationObservation, TargetQuery
from .truth.base import EngineClient
from .truth.share import ShareStat, compute_shares, domain_share

log = logging.getLogger("citeworthy.tracker")


class TrackBudgetExceeded(Exception):
    """Raised when a track run's estimated cost exceeds the cap (§8)."""


@dataclass
class EngineRunStat:
    query_id: str
    engine: str
    n_samples: int
    our_share: ShareStat
    top_domains: list[ShareStat]
    failures: int


@dataclass
class TrackResult:
    run_id: str
    stats: list[EngineRunStat] = field(default_factory=list)
    total_cost_usd: float = 0.0
    n_samples: int = 0
    n_failures: int = 0
    warnings: list[str] = field(default_factory=list)


def _hash_answer(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_track(
    queries: list[TargetQuery],
    clients: list[EngineClient],
    config: Config,
    *,
    run_id: str,
    conn: sqlite3.Connection | None = None,
    samples: int | None = None,
    budget_cap: float | None = None,
    our_domain: str | None = None,
) -> TrackResult:
    """Sample every (query, engine) and persist observations (§6.6)."""
    samples = samples if samples is not None else config.truth.samples_per_engine
    our_domain = our_domain or config.our_domain
    result = TrackResult(run_id=run_id)

    for query in queries:
        for client in clients:
            engine = client.engine
            per_run: list[CitationObservation] = []
            failures = 0
            for i in range(samples):
                cost = config.truth_pricing.get(engine, 0.0)
                result.total_cost_usd += cost
                if budget_cap is not None and result.total_cost_usd > budget_cap:
                    raise TrackBudgetExceeded(
                        f"track cost ${result.total_cost_usd:.4f} exceeds cap "
                        f"${budget_cap:.2f}"
                    )
                try:
                    sample = client.sample_once(query.text)
                except Exception as exc:  # noqa: BLE001 - one flaky call shouldn't kill a cron run
                    failures += 1
                    result.n_failures += 1
                    result.warnings.append(f"{engine}/{query.id} sample {i}: {exc}")
                    log.warning("track sample failed: %s/%s #%d: %s", engine, query.id, i, exc)
                    continue

                obs = CitationObservation(
                    run_id=run_id,
                    query_id=query.id,
                    engine=engine,  # type: ignore[arg-type]
                    sample_index=i,
                    cited_urls=sample.cited_urls,
                    answer_text_hash=_hash_answer(sample.answer_text),
                    observed_at=datetime.now(timezone.utc),
                )
                per_run.append(obs)
                result.n_samples += 1
                if conn is not None:
                    db.save_observation(conn, run_id, obs)

            if samples > 0 and failures == samples:
                result.warnings.append(
                    f"[!] {engine}/{query.id}: all {samples} samples failed — "
                    "check the API key or a response-schema change."
                )

            result.stats.append(
                EngineRunStat(
                    query_id=query.id,
                    engine=engine,
                    n_samples=len(per_run),
                    our_share=domain_share(per_run, our_domain),
                    top_domains=compute_shares(per_run)[:5],
                    failures=failures,
                )
            )
    return result
