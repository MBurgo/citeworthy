"""Markdown reports (§6.7).

Per-query report: ranked passage table (domain, rank, selection prob ± CI, ours
highlighted), where-we-stand summary (best passage, leader, CI overlap,
buried-answer gap), loss-reason diagnosis, and recommended actions. Plain
markdown so it pastes into Slack/docs. The citation-share trend is appended once
v3 tracking data exists (Milestone 5). Portfolio `--all` is Milestone 7.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import db
from .models import Matchup, Passage, RankResult, TargetQuery
from .ranker.diagnose import Diagnosis, diagnose_losses

REPORTS_DIR = Path("reports")

# Recommended action per §6.3 loss-reason (used in the report's actions block).
ACTIONS: dict[str, str] = {
    "off_topic": "Make the passage answer the query directly in its opening line.",
    "too_generic": "Replace generalities with specific facts, figures, and named entities.",
    "buried_answer": "Front-load the direct answer into the first two sentences.",
    "no_evidence": "Add concrete evidence — data, dates, numbers, named sources.",
    "stale_signals": "Add explicit current dates and up-to-date figures in the text.",
    "hedgy": "State the answer plainly; remove hedging and qualifiers.",
    "poor_structure": "Tighten structure — clear headings, short self-contained sentences.",
    "promotional": "Cut promotional language; lead with informational substance.",
    "thin": "Expand the passage with substantive, on-topic detail.",
    "other": "Review the passage against the judge's verbatim feedback below.",
}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _disagreement_from_matchups(matchups: list[Matchup]) -> float:
    """Position-swap disagreement rate, recomputed from stored matchups (§6.2)."""
    ab: dict[tuple[str, str], list[str]] = {}
    ba: dict[tuple[str, str], list[str]] = {}
    for m in matchups:
        key = (m.passage_a, m.passage_b)
        (ab if m.position_order == "ab" else ba).setdefault(key, []).append(m.winner)
    pairs = set(ab) | set(ba)
    if not pairs:
        return 0.0

    def majority(xs: list[str]) -> str:
        return max(set(xs), key=xs.count)

    disagreements = 0
    for key in pairs:
        a, b = ab.get(key), ba.get(key)
        if a and b and majority(a) != majority(b):
            disagreements += 1
    return disagreements / len(pairs)


@dataclass
class OurPosition:
    has_passages: bool
    best: RankResult | None = None
    best_passage: Passage | None = None
    leader: RankResult | None = None
    leader_passage: Passage | None = None
    ci_overlaps_leader: bool = False
    first_chunk: RankResult | None = None  # our page's chunk_index == 0


def _our_position(
    rank_results: list[RankResult], passages: dict[str, Passage]
) -> OurPosition:
    by_rank = sorted(rank_results, key=lambda r: r.rank)
    leader = by_rank[0] if by_rank else None
    ours = [r for r in rank_results if passages[r.passage_id].is_ours]
    if not ours:
        return OurPosition(has_passages=False, leader=leader,
                           leader_passage=passages[leader.passage_id] if leader else None)

    best = min(ours, key=lambda r: r.rank)
    first_chunk = next(
        (r for r in ours if passages[r.passage_id].chunk_index == 0), None
    )
    overlaps = bool(
        leader
        and best.passage_id != leader.passage_id
        and best.ci_high >= leader.ci_low
    )
    return OurPosition(
        has_passages=True,
        best=best,
        best_passage=passages[best.passage_id],
        leader=leader,
        leader_passage=passages[leader.passage_id] if leader else None,
        ci_overlaps_leader=overlaps,
        first_chunk=first_chunk,
    )


def render_query_report_md(
    query: TargetQuery,
    rank_results: list[RankResult],
    passages: dict[str, Passage],
    matchups: list[Matchup],
    *,
    run_id: str = "",
    our_domain: str = "fool.com.au",
    disagreement_rate: float | None = None,
) -> str:
    """Render the per-query markdown report from in-memory data (§6.7)."""
    our_ids = {pid for pid, p in passages.items() if p.is_ours}
    diagnosis = diagnose_losses(matchups, our_ids)
    pos = _our_position(rank_results, passages)
    if disagreement_rate is None:
        disagreement_rate = _disagreement_from_matchups(matchups)

    lines: list[str] = []
    lines.append(f"# Citeworthy — {query.text}")
    lines.append("")
    meta = f"`{query.id}` · tier **{query.tier}** · our domain `{our_domain}`"
    if run_id:
        meta += f" · run `{run_id[:8]}`"
    lines.append(meta)
    lines.append("")

    # Ranked passages table.
    lines.append(f"## Ranked passages ({len(rank_results)})")
    lines.append("")
    lines.append("| Rank | Domain | Ours | Sel. prob | 95% CI |")
    lines.append("|-----:|--------|:----:|----------:|--------|")
    for r in sorted(rank_results, key=lambda r: r.rank):
        p = passages[r.passage_id]
        ours = "✓" if p.is_ours else ""
        domain = f"**{p.domain}**" if p.is_ours else p.domain
        ci = f"{_pct(r.ci_low)}–{_pct(r.ci_high)}"
        lines.append(f"| {r.rank} | {domain} | {ours} | {_pct(r.selection_prob)} | {ci} |")
    lines.append("")

    # Where we stand.
    lines.append("## Where we stand")
    lines.append("")
    if not pos.has_passages:
        lines.append("- We have **no passages** in this candidate set. "
                     "Check `our_url` and extraction warnings.")
    else:
        b, bp = pos.best, pos.best_passage
        assert b and bp
        lines.append(
            f"- **Our best passage:** rank {b.rank}/{len(rank_results)} — "
            f"`{bp.domain}` (chunk {bp.chunk_index}) — "
            f"sel. prob {_pct(b.selection_prob)} (CI {_pct(b.ci_low)}–{_pct(b.ci_high)})"
        )
        if pos.leader and pos.leader_passage:
            lead_is_ours = pos.leader.passage_id == b.passage_id
            lead_note = " (that's us! 🎉)" if lead_is_ours else ""
            lines.append(
                f"- **Leader:** `{pos.leader_passage.domain}` (rank 1) — "
                f"sel. prob {_pct(pos.leader.selection_prob)}{lead_note}"
            )
            if not lead_is_ours:
                if pos.ci_overlaps_leader:
                    lines.append("- **CI vs leader:** overlaps — the gap is **not** "
                                 "statistically clear on this run.")
                else:
                    lines.append("- **CI vs leader:** does **not** overlap — the leader "
                                 "is clearly ahead.")
        if pos.first_chunk:
            gap = pos.first_chunk.rank - b.rank
            if gap > 0:
                lines.append(
                    f"- **Buried-answer gap:** our page's first chunk ranks "
                    f"{pos.first_chunk.rank}, but our best chunk ranks {b.rank} "
                    f"(gap of {gap}). The strong material is not at the top of the page."
                )
            else:
                lines.append("- **Buried-answer gap:** our first chunk is already our "
                             "best — the answer is well placed.")
    lines.append("")

    # Loss diagnosis.
    lines.append("## Loss diagnosis")
    lines.append("")
    if diagnosis.total_losses == 0:
        lines.append("Our passages lost no matchups on this run.")
    else:
        lines.append(f"Our passages lost **{diagnosis.total_losses}** matchups. "
                     "Top weaknesses:")
        lines.append("")
        for i, w in enumerate(diagnosis.top_weaknesses, start=1):
            action = ACTIONS.get(w.reason_code, ACTIONS["other"])
            lines.append(f"{i}. **{w.reason_code}** ({w.count} losses) — {action}")
            for ex in w.examples:
                lines.append(f"   - _{ex}_")
    lines.append("")

    # Recommended actions.
    if diagnosis.top_weaknesses:
        lines.append("## Recommended actions")
        lines.append("")
        for w in diagnosis.top_weaknesses:
            lines.append(f"- {ACTIONS.get(w.reason_code, ACTIONS['other'])}")
        lines.append("")

    # Footer.
    lines.append("---")
    warn = ""
    if disagreement_rate > 0.30:
        warn = " ⚠️ Above 30% — judge signal is weak; treat this ranking with caution."
    lines.append(f"*Judge position-swap disagreement: {_pct(disagreement_rate)}.{warn}*")
    lines.append("")
    lines.append("*Citation-share trend appears here once v3 tracking data exists "
                 "(Milestone 5).*")
    lines.append("")
    return "\n".join(lines)


def write_report(query_id: str, markdown: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{query_id}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def render_query_report(
    conn: sqlite3.Connection, query_id: str, *, our_domain: str = "fool.com.au"
) -> tuple[str, Path]:
    """Render and write the report for a query's most recent rank run (§6.7)."""
    query = db.get_query(conn, query_id)
    if query is None:
        raise KeyError(f"no such query: {query_id}")
    run_id = db.latest_rank_run_id(conn, query_id)
    if run_id is None:
        raise KeyError(f"no rank run found for {query_id} — run `citeworthy rank {query_id}` first")

    rank_results = db.get_rank_results(conn, run_id, query_id)
    matchups = db.get_matchups(conn, run_id, query_id)
    passages = db.get_passages(conn, [r.passage_id for r in rank_results])

    md = render_query_report_md(
        query, rank_results, passages, matchups, run_id=run_id, our_domain=our_domain
    )
    return md, write_report(query_id, md)


def render_portfolio_report() -> str:
    """The `--all` portfolio summary (§6.7) — implemented in Milestone 7."""
    raise NotImplementedError("Portfolio report lands with Milestone 7.")
