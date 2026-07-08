"""Bradley–Terry fitting and uncertainty.

Convert judgments to a win matrix; fit with `choix.ilsr_pairwise` (alpha=0.01);
selection_prob = softmax over strengths; bootstrap (200 resamples) for CIs (§6.2).
Fallback to simple Elo if choix fitting is unstable with sparse data (§3).
Scaffolded in Milestone 1; implemented in Milestone 3.
"""

from __future__ import annotations

from ..models import RankResult


def fit_bradley_terry(
    passage_ids: list[str],
    win_pairs: list[tuple[int, int]],
    *,
    alpha: float = 0.01,
    bootstrap_resamples: int = 200,
) -> list[RankResult]:
    """Fit BT strengths and bootstrap CIs on selection_prob (§6.2)."""
    raise NotImplementedError("fit_bradley_terry lands with Milestone 3.")
