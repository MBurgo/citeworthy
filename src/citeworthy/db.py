"""SQLite schema and repository functions.

Single file `citeworthy.db`. Tables mirror the pydantic models in models.py,
plus `runs` (reproducibility) and `judge_calls` (token/cost audit — every LLM
call is logged, §6.2 / CLAUDE.md). See §4, §5 of CITEWORTHY_SPEC.md.

Plain `sqlite3` is used to keep the schema portable to Cloudflare D1 later.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .models import (
    CitationObservation,
    Matchup,
    Passage,
    RankResult,
    Run,
    TargetQuery,
)

DEFAULT_DB_PATH = Path("citeworthy.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id        TEXT PRIMARY KEY,
    text      TEXT NOT NULL,
    our_url   TEXT,
    tier      TEXT NOT NULL CHECK (tier IN ('money', 'supporting'))
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('rank', 'optimize', 'track')),
    started_at  TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    git_sha     TEXT
);

CREATE TABLE IF NOT EXISTS passages (
    id            TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    domain        TEXT NOT NULL,
    title         TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    is_ours       INTEGER NOT NULL DEFAULT 0,
    variant_of    TEXT,
    variant_label TEXT
);

CREATE TABLE IF NOT EXISTS matchups (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    query_id       TEXT NOT NULL,
    passage_a      TEXT NOT NULL,
    passage_b      TEXT NOT NULL,
    winner         TEXT NOT NULL,
    reason_code    TEXT NOT NULL,
    reason_text    TEXT NOT NULL,
    judge_model    TEXT NOT NULL,
    position_order TEXT NOT NULL CHECK (position_order IN ('ab', 'ba')),
    sample_index   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS rank_results (
    run_id         TEXT NOT NULL,
    query_id       TEXT NOT NULL,
    passage_id     TEXT NOT NULL,
    bt_strength    REAL NOT NULL,
    selection_prob REAL NOT NULL,
    rank           INTEGER NOT NULL,
    ci_low         REAL NOT NULL,
    ci_high        REAL NOT NULL,
    PRIMARY KEY (run_id, passage_id)
);

CREATE TABLE IF NOT EXISTS observations (
    run_id           TEXT NOT NULL,
    query_id         TEXT NOT NULL,
    engine           TEXT NOT NULL CHECK (engine IN ('perplexity', 'gemini_grounded')),
    sample_index     INTEGER NOT NULL,
    cited_urls       TEXT NOT NULL,   -- JSON array
    answer_text_hash TEXT NOT NULL,
    observed_at      TEXT NOT NULL
);

-- Token/cost audit for every LLM call, for `citeworthy costs` and budget caps.
CREATE TABLE IF NOT EXISTS judge_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    est_usd       REAL NOT NULL,
    created_at    TEXT NOT NULL
);
"""


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Create the database file and all tables if they don't exist."""
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return Path(path)


# --- Repository functions --------------------------------------------------
# Milestone 1 covers query management (used by `queries add|list|rm`) and run
# creation. Persistence for passages/matchups/results/observations is scaffolded
# and lands with the milestone that produces that data.


def upsert_query(conn: sqlite3.Connection, q: TargetQuery) -> None:
    conn.execute(
        """
        INSERT INTO queries (id, text, our_url, tier)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            text = excluded.text,
            our_url = excluded.our_url,
            tier = excluded.tier
        """,
        (q.id, q.text, str(q.our_url) if q.our_url else None, q.tier),
    )
    conn.commit()


def list_queries(conn: sqlite3.Connection) -> list[TargetQuery]:
    rows = conn.execute("SELECT id, text, our_url, tier FROM queries ORDER BY id").fetchall()
    return [
        TargetQuery(
            id=r["id"],
            text=r["text"],
            our_url=r["our_url"],
            tier=r["tier"],
        )
        for r in rows
    ]


def get_query(conn: sqlite3.Connection, query_id: str) -> TargetQuery | None:
    r = conn.execute(
        "SELECT id, text, our_url, tier FROM queries WHERE id = ?", (query_id,)
    ).fetchone()
    if r is None:
        return None
    return TargetQuery(id=r["id"], text=r["text"], our_url=r["our_url"], tier=r["tier"])


def remove_query(conn: sqlite3.Connection, query_id: str) -> bool:
    cur = conn.execute("DELETE FROM queries WHERE id = ?", (query_id,))
    conn.commit()
    return cur.rowcount > 0


def create_run(conn: sqlite3.Connection, run: Run) -> None:
    conn.execute(
        """
        INSERT INTO runs (run_id, kind, started_at, config_hash, git_sha)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            run.run_id,
            run.kind,
            run.started_at.isoformat(),
            run.config_hash,
            run.git_sha,
        ),
    )
    conn.commit()


def log_judge_call(
    conn: sqlite3.Connection,
    run_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    est_usd: float,
    created_at: datetime,
) -> None:
    conn.execute(
        """
        INSERT INTO judge_calls
            (run_id, model, input_tokens, output_tokens, est_usd, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, model, input_tokens, output_tokens, est_usd, created_at.isoformat()),
    )
    conn.commit()


def cost_summary(conn: sqlite3.Connection) -> list[dict]:
    """Aggregate logged token spend per model, for `citeworthy costs`."""
    rows = conn.execute(
        """
        SELECT model,
               COUNT(*)            AS calls,
               SUM(input_tokens)   AS input_tokens,
               SUM(output_tokens)  AS output_tokens,
               SUM(est_usd)        AS est_usd
        FROM judge_calls
        GROUP BY model
        ORDER BY est_usd DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


# --- Rank persistence (Milestone 4) ----------------------------------------


def save_passage(conn: sqlite3.Connection, passage: Passage) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO passages
            (id, url, domain, title, chunk_index, text, is_ours, variant_of, variant_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            passage.id,
            str(passage.url),
            passage.domain,
            passage.title,
            passage.chunk_index,
            passage.text,
            int(passage.is_ours),
            passage.variant_of,
            passage.variant_label,
        ),
    )
    conn.commit()


def save_matchup(conn: sqlite3.Connection, run_id: str, matchup: Matchup) -> None:
    conn.execute(
        """
        INSERT INTO matchups
            (run_id, query_id, passage_a, passage_b, winner, reason_code,
             reason_text, judge_model, position_order, sample_index)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            matchup.query_id,
            matchup.passage_a,
            matchup.passage_b,
            matchup.winner,
            matchup.reason_code,
            matchup.reason_text,
            matchup.judge_model,
            matchup.position_order,
            matchup.sample_index,
        ),
    )
    conn.commit()


def save_rank_result(conn: sqlite3.Connection, result: RankResult) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO rank_results
            (run_id, query_id, passage_id, bt_strength, selection_prob, rank, ci_low, ci_high)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.run_id,
            result.query_id,
            result.passage_id,
            result.bt_strength,
            result.selection_prob,
            result.rank,
            result.ci_low,
            result.ci_high,
        ),
    )
    conn.commit()


def latest_rank_run_id(conn: sqlite3.Connection, query_id: str) -> str | None:
    """The run_id of the most recent rank run that produced results for a query."""
    row = conn.execute(
        """
        SELECT rr.run_id
        FROM rank_results rr
        JOIN runs r ON rr.run_id = r.run_id
        WHERE rr.query_id = ?
        ORDER BY r.started_at DESC
        LIMIT 1
        """,
        (query_id,),
    ).fetchone()
    return row["run_id"] if row else None


def get_rank_results(
    conn: sqlite3.Connection, run_id: str, query_id: str
) -> list[RankResult]:
    rows = conn.execute(
        """
        SELECT run_id, query_id, passage_id, bt_strength, selection_prob, rank, ci_low, ci_high
        FROM rank_results
        WHERE run_id = ? AND query_id = ?
        ORDER BY rank
        """,
        (run_id, query_id),
    ).fetchall()
    return [RankResult(**dict(r)) for r in rows]


def get_matchups(
    conn: sqlite3.Connection, run_id: str, query_id: str
) -> list[Matchup]:
    rows = conn.execute(
        """
        SELECT query_id, passage_a, passage_b, winner, reason_code, reason_text,
               judge_model, position_order, sample_index
        FROM matchups
        WHERE run_id = ? AND query_id = ?
        """,
        (run_id, query_id),
    ).fetchall()
    return [Matchup(**dict(r)) for r in rows]


def get_passages(conn: sqlite3.Connection, ids: list[str]) -> dict[str, Passage]:
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""
        SELECT id, url, domain, title, chunk_index, text, is_ours, variant_of, variant_label
        FROM passages WHERE id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    result: dict[str, Passage] = {}
    for r in rows:
        result[r["id"]] = Passage(
            id=r["id"],
            url=r["url"],
            domain=r["domain"],
            title=r["title"],
            chunk_index=r["chunk_index"],
            text=r["text"],
            is_ours=bool(r["is_ours"]),
            variant_of=r["variant_of"],
            variant_label=r["variant_label"],
        )
    return result


# --- Tracker persistence (Milestone 5) -------------------------------------


def save_observation(conn: sqlite3.Connection, run_id: str, obs: CitationObservation) -> None:
    conn.execute(
        """
        INSERT INTO observations
            (run_id, query_id, engine, sample_index, cited_urls, answer_text_hash, observed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            obs.query_id,
            obs.engine,
            obs.sample_index,
            json.dumps(obs.cited_urls),
            obs.answer_text_hash,
            obs.observed_at.isoformat(),
        ),
    )
    conn.commit()


def _row_to_observation(r: sqlite3.Row) -> tuple[str, CitationObservation]:
    """Returns (run_id, observation)."""
    return r["run_id"], CitationObservation(
        run_id=r["run_id"],
        query_id=r["query_id"],
        engine=r["engine"],
        sample_index=r["sample_index"],
        cited_urls=json.loads(r["cited_urls"]),
        answer_text_hash=r["answer_text_hash"],
        observed_at=datetime.fromisoformat(r["observed_at"]),
    )


def get_observations(
    conn: sqlite3.Connection,
    *,
    query_id: str | None = None,
    engine: str | None = None,
    run_id: str | None = None,
) -> list[CitationObservation]:
    """Fetch observations (optionally filtered), oldest run first."""
    clauses, params = [], []
    if query_id is not None:
        clauses.append("o.query_id = ?")
        params.append(query_id)
    if engine is not None:
        clauses.append("o.engine = ?")
        params.append(engine)
    if run_id is not None:
        clauses.append("o.run_id = ?")
        params.append(run_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT o.run_id, o.query_id, o.engine, o.sample_index, o.cited_urls,
               o.answer_text_hash, o.observed_at
        FROM observations o
        JOIN runs r ON o.run_id = r.run_id
        {where}
        ORDER BY r.started_at ASC, o.sample_index ASC
        """,
        params,
    ).fetchall()
    return [_row_to_observation(r)[1] for r in rows]


def observations_by_run(
    conn: sqlite3.Connection, query_id: str, engine: str
) -> list[list[CitationObservation]]:
    """Observations for a (query, engine), grouped by run in chronological order
    — the shape the rolling-window / trend stats expect (§6.6)."""
    rows = conn.execute(
        """
        SELECT o.run_id, o.query_id, o.engine, o.sample_index, o.cited_urls,
               o.answer_text_hash, o.observed_at, r.started_at
        FROM observations o
        JOIN runs r ON o.run_id = r.run_id
        WHERE o.query_id = ? AND o.engine = ?
        ORDER BY r.started_at ASC, o.sample_index ASC
        """,
        (query_id, engine),
    ).fetchall()
    groups: list[list[CitationObservation]] = []
    order: list[str] = []
    by_run: dict[str, list[CitationObservation]] = {}
    for r in rows:
        run_id, obs = _row_to_observation(r)
        if run_id not in by_run:
            by_run[run_id] = []
            order.append(run_id)
        by_run[run_id].append(obs)
    for run_id in order:
        groups.append(by_run[run_id])
    return groups


def _dump_urls(urls: list[str]) -> str:
    return json.dumps(urls)
