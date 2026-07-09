"""Web layer: FastAPI API (Vercel serverless) + background worker.

The API is fast (DB reads + enqueue); the worker claims queued jobs and runs the
long rank/optimize/track flows. Both share a Postgres database (Neon/Supabase) via
CITEWORTHY_DATABASE_URL; locally they fall back to SQLite.
"""
