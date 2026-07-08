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


# --- Scaffolded persistence (implemented with their milestone) -------------


def save_passage(conn: sqlite3.Connection, passage: Passage) -> None:
    raise NotImplementedError("Passage persistence lands with Milestone 2.")


def save_matchup(conn: sqlite3.Connection, run_id: str, matchup: Matchup) -> None:
    raise NotImplementedError("Matchup persistence lands with Milestone 3.")


def save_rank_result(conn: sqlite3.Connection, result: RankResult) -> None:
    raise NotImplementedError("RankResult persistence lands with Milestone 3.")


def save_observation(conn: sqlite3.Connection, obs: CitationObservation) -> None:
    raise NotImplementedError("Observation persistence lands with Milestone 5.")


def _dump_urls(urls: list[str]) -> str:
    return json.dumps(urls)
