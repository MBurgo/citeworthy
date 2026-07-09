# Citeworthy — web app

An optional web layer over the Citeworthy engine: a FastAPI API + interactive
dashboard for managing queries, triggering rank/optimize/track jobs, and viewing
reports and citation-share trends. The CLI still works unchanged — the web layer
reuses the same engine functions and database schema.

## Architecture

The heavy jobs (a `rank` run is hundreds of LLM calls over several minutes) can't
run inside a Vercel serverless function's timeout, so the work is split:

```
  Browser ──► Vercel (FastAPI serverless)          Postgres (Neon/Supabase)
              • read: queries, reports,        ◄──►  • queries, runs, passages,
                portfolio, costs                       matchups, rank_results,
              • enqueue jobs  ─────────────────┐       observations, judge_calls
                                               │       • jobs (the work queue)
                                               ▼
                              Worker (Render / Fly / Cloud Run)
                              • claims queued jobs
                              • runs rank / optimize / track
                              • writes results back to Postgres
```

- **API (Vercel):** fast endpoints only — DB reads, report rendering, and
  enqueuing jobs. `src/citeworthy/web/app.py`, served via `api/index.py`.
- **Worker (persistent host):** polls the `jobs` table and runs the engine.
  `src/citeworthy/web/worker.py` (`python -m citeworthy.web.worker`).
- **Database:** Postgres, shared by both. `db.py` is dual-backend — the CLI/tests
  use SQLite, the deployed app/worker use Postgres via `CITEWORTHY_DATABASE_URL`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard (HTML) |
| GET/POST/DELETE | `/api/queries` | Manage the target query set |
| POST | `/api/jobs` | Enqueue a `rank` / `optimize` / `track` job |
| GET | `/api/jobs`, `/api/jobs/{id}` | Job list / status |
| GET | `/api/report/{query_id}` | Per-query report (markdown + HTML) |
| GET | `/api/portfolio` | Portfolio summary + judge-vs-reality correlation |
| GET | `/api/costs` | Token-spend summary |

## Run locally

```bash
uv pip install -e ".[web]"
# API (SQLite by default — no DATABASE_URL needed locally):
uv run uvicorn citeworthy.web.app:app --reload
# Worker (separate terminal; needs API keys in the environment):
uv run python -m citeworthy.web.worker
```

Open http://localhost:8000. Add a query, click **Rank**, and watch the job move
`queued → running → done`; then **Report**.

## Deploy

### 1. Database — Neon or Supabase

Create a Postgres database and copy its connection string. The schema is created
automatically on first API/worker start (`ensure_schema`).

### 2. API — Vercel

- `vercel.json` routes all requests to `api/index.py` (the ASGI app).
- `requirements.txt` pins the **API-only** deps (no trafilatura/choix/anthropic —
  those are the worker's, kept out to stay under Vercel's function size limit).
- Set project env vars: `CITEWORTHY_DATABASE_URL` (the Postgres string). The API
  never calls the LLM providers, so it needs no API keys.

### 3. Worker — Render / Fly / Cloud Run

Build `Dockerfile.worker` and run it with these env vars:

| Var | Needed for |
|---|---|
| `CITEWORTHY_DATABASE_URL` | always (same Postgres as the API) |
| `ANTHROPIC_API_KEY` | rank, optimize |
| `SERPAPI_API_KEY` | rank |
| `PERPLEXITY_API_KEY`, `GEMINI_API_KEY` | track (each optional; degrades) |

### 4. Weekly tracker

Point Vercel Cron (or any scheduler) at a small endpoint / script that POSTs
`{"kind":"track"}` to `/api/jobs` weekly. The worker runs it; the judge-vs-reality
correlation becomes meaningful after ~4 weeks (§12.1).

## Caveats / next steps

- **Job claiming** uses a simple `UPDATE ... WHERE status='queued'`. It's safe for
  one worker; for multiple concurrent workers on Postgres, switch the claim to
  `SELECT ... FOR UPDATE SKIP LOCKED`.
- **Budget caps** still apply inside each run (rank/track abort on the cap).
- **Auth:** the dashboard is unauthenticated. Put it behind Vercel's access
  protection (or an auth proxy) before exposing it beyond the team.
