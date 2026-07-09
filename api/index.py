"""Vercel serverless entry point.

Vercel's Python runtime serves the ASGI `app` exported here. We add ../src to the
import path so the `citeworthy` package (src layout) is importable without an
install step. Third-party deps come from the root requirements.txt.

Set CITEWORTHY_DATABASE_URL (Postgres) in the Vercel project env. The heavy
rank/optimize/track jobs run in the separate worker, not here.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from citeworthy.web.app import app  # noqa: E402  (path setup must run first)

__all__ = ["app"]
