"""Markdown + terminal reports.

Per-query: ranked passage table (domain, rank, selection prob ± CI, ours
highlighted), loss-reason diagnosis, recommended actions, and — once v3 data
exists — the citation-share trend. Plus `--all` portfolio summary. Plain
markdown so it pastes into Slack/docs (§6.7). Scaffolded in Milestone 1;
implemented in Milestone 4.
"""

from __future__ import annotations

from pathlib import Path

REPORTS_DIR = Path("reports")


def render_query_report(query_id: str) -> str:
    """Render the markdown report for one query and write it to reports/ (§6.7)."""
    raise NotImplementedError("render_query_report lands with Milestone 4.")


def render_portfolio_report() -> str:
    """Render the `--all` portfolio summary (§6.7)."""
    raise NotImplementedError("render_portfolio_report lands with Milestone 7.")
