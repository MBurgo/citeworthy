"""Hill-climbing controller for the edit loop (v2, §6.5).

Reads a query's most recent rank run, then iterates:
  generate variants of our best passage → compliance-gate every variant →
  re-run the tournament on (current best + surviving variants + top-5 competitors)
  → accept a variant only if its selection_prob beats the original's with
  non-overlapping bootstrap CIs. Max 3 iterations. "No significant improvement"
  is a valid, reported outcome.

Judge, editor transport, and compliance checker are injected so the whole loop is
testable with no live calls and a stub checker (§10, CLAUDE.md).
"""

from __future__ import annotations

import difflib
import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .. import db
from ..config import Config
from ..models import Passage, RankResult, TargetQuery
from ..ranker.bradley_terry import fit_bradley_terry
from ..ranker.diagnose import diagnose_losses
from ..ranker.judge import Judge, JudgeTransport
from ..ranker.tournament import BudgetExceeded, run_tournament
from .compliance import ComplianceChecker
from .variants import Variant, generate_variants

TOP_COMPETITORS = 5  # §6.5 step 4: original + variants + top 5 competitors


class OptimizeError(Exception):
    """Raised when a query can't be optimised (e.g. never ranked)."""


@dataclass
class FailedVariant:
    label: str
    violations: list[str]


@dataclass
class Iteration:
    index: int
    variants_generated: int
    variants_passed: int
    failed: list[FailedVariant] = field(default_factory=list)
    best_label: str | None = None
    best_prob: float = 0.0
    original_prob: float = 0.0
    accepted: bool = False


@dataclass
class OptimizeOutput:
    query: TargetQuery
    run_id: str
    original: Passage
    final: Passage
    accepted: bool = False
    iterations: list[Iteration] = field(default_factory=list)
    total_cost_usd: float = 0.0
    diff: str = ""
    report_md: str = ""
    report_path: object = None

    @property
    def failed_variants(self) -> list[FailedVariant]:
        return [f for it in self.iterations for f in it.failed]


def _variant_passage(original: Passage, variant: Variant, iteration: int, idx: int) -> Passage:
    seed = f"{original.id}|{variant.label}|{iteration}|{idx}|{variant.text[:64]}"
    vid = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return Passage(
        id=vid,
        url=original.url,
        domain=original.domain,
        title=original.title,
        chunk_index=original.chunk_index,
        text=variant.text,
        is_ours=True,
        variant_of=original.id,
        variant_label=variant.label,
    )


def _load_ranked_context(conn: sqlite3.Connection, query: TargetQuery):
    """Fetch the latest rank run's best-our-passage, competitors, and weaknesses."""
    run_id = db.latest_rank_run_id(conn, query.id)
    if run_id is None:
        raise OptimizeError(
            f"{query.id} has no rank run — run `citeworthy rank {query.id}` first."
        )
    rank_results = db.get_rank_results(conn, run_id, query.id)
    matchups = db.get_matchups(conn, run_id, query.id)
    passages = db.get_passages(conn, [r.passage_id for r in rank_results])

    ours = [r for r in rank_results if passages[r.passage_id].is_ours]
    if not ours:
        raise OptimizeError(f"{query.id} has no passages of ours to optimise.")
    original = passages[min(ours, key=lambda r: r.rank).passage_id]

    competitors = [
        passages[r.passage_id]
        for r in sorted(rank_results, key=lambda r: r.rank)
        if not passages[r.passage_id].is_ours
    ][:TOP_COMPETITORS]

    weaknesses = [w.reason_code for w in diagnose_losses(matchups, {original.id}).top_weaknesses]
    return original, competitors, weaknesses


def run_optimize(
    query: TargetQuery,
    config: Config,
    *,
    judge: Judge,
    editor_transport: JudgeTransport,
    checker: ComplianceChecker,
    conn: sqlite3.Connection,
    run_id: str,
    editor_model: str | None = None,
    budget_cap: float | None = None,
) -> OptimizeOutput:
    """Run the guided hill-climb and return the recommended passage (§6.5)."""
    editor_model = editor_model or config.editor.model
    if budget_cap is None:
        budget_cap = config.budget.max_usd_per_rank_run

    original, competitors, weaknesses = _load_ranked_context(conn, query)

    out = OptimizeOutput(query=query, run_id=run_id, original=original, final=original)
    current = original
    current_weaknesses = weaknesses or []
    spent = 0.0

    for i in range(1, config.editor.max_iterations + 1):
        variants = generate_variants(
            current, current_weaknesses,
            transport=editor_transport, model=editor_model,
            n_variants=config.editor.variants,
        )
        # Log/charge every editor call (CLAUDE.md).
        for v in variants:
            cost = config.estimate_cost(editor_model, v.input_tokens, v.output_tokens)
            spent += cost
            db.log_judge_call(
                conn, run_id=run_id, model=editor_model,
                input_tokens=v.input_tokens, output_tokens=v.output_tokens,
                est_usd=cost, created_at=datetime.now(timezone.utc),
            )
            if budget_cap is not None and spent > budget_cap:
                out.total_cost_usd = spent
                raise BudgetExceeded(f"optimize cost ${spent:.4f} exceeds cap ${budget_cap:.2f}")

        # Compliance gate — EVERY variant, no exceptions (§6.5, CLAUDE.md).
        passed: list[Variant] = []
        failed: list[FailedVariant] = []
        for v in variants:
            result = checker.check(v.text, original=original.text)
            if result.passed:
                passed.append(v)
            else:
                failed.append(FailedVariant(label=v.label, violations=result.violations))

        iteration = Iteration(
            index=i,
            variants_generated=len(variants),
            variants_passed=len(passed),
            failed=failed,
        )

        if not passed:
            out.iterations.append(iteration)
            break  # nothing survived the gate this round

        # Reduced field: current best + surviving variants + top-5 competitors.
        variant_passages = [
            _variant_passage(original, v, i, idx) for idx, v in enumerate(passed)
        ]
        field_passages = [current, *variant_passages, *competitors]

        remaining = None if budget_cap is None else max(0.0, budget_cap - spent)
        tournament = run_tournament(
            query.id, query.text, field_passages, judge, config,
            conn=conn, run_id=run_id, budget_cap=remaining,
        )
        spent += tournament.total_cost_usd

        bt = {r.passage_id: r for r in fit_bradley_terry(
            tournament.passage_ids, tournament.comparisons,
            bootstrap_resamples=config.tournament.bootstrap_resamples,
        )}
        # Persist the reduced field for audit.
        for p in variant_passages:
            db.save_passage(conn, p)
        for m in tournament.matchups:
            db.save_matchup(conn, run_id, m)
        for pid, r in bt.items():
            db.save_rank_result(conn, RankResult(
                query_id=query.id, run_id=run_id, passage_id=pid,
                bt_strength=r.bt_strength, selection_prob=r.selection_prob,
                rank=r.rank, ci_low=r.ci_low, ci_high=r.ci_high,
            ))

        original_result = bt[current.id]
        iteration.original_prob = original_result.selection_prob

        best_vp, best_res = max(
            ((vp, bt[vp.id]) for vp in variant_passages),
            key=lambda t: t[1].selection_prob,
        )
        iteration.best_label = best_vp.variant_label
        iteration.best_prob = best_res.selection_prob

        # Accept only on a real, non-overlapping improvement (§6.5 step 5).
        accepted = (
            best_res.selection_prob > original_result.selection_prob
            and best_res.ci_low > original_result.ci_high
        )
        iteration.accepted = accepted
        out.iterations.append(iteration)

        if not accepted:
            break  # no significant improvement — keep the original (a valid outcome)

        # Hill-climb: the accepted variant becomes the new best; re-diagnose.
        current = best_vp
        out.accepted = True
        out.final = current
        re_diag = diagnose_losses(tournament.matchups, {current.id})
        current_weaknesses = [w.reason_code for w in re_diag.top_weaknesses] or current_weaknesses

    out.total_cost_usd = spent
    out.diff = _unified_diff(original.text, out.final.text)

    from ..report import render_optimize_report_md

    out.report_md = render_optimize_report_md(out)
    return out


def _unified_diff(original: str, final: str) -> str:
    """Sentence-level unified diff, readable in a fenced report block."""
    def sentences(t: str) -> list[str]:
        import re
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]

    if original == final:
        return ""
    return "\n".join(
        difflib.unified_diff(
            sentences(original), sentences(final),
            fromfile="original", tofile="recommended", lineterm="",
        )
    )
