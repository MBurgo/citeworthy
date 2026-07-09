"""Bradley–Terry fitting and uncertainty.

Convert pairwise judgments to a comparison list, fit strengths with
`choix.ilsr_pairwise` (alpha=0.01 regularises sparse data), compute
`selection_prob` as a softmax over strengths, and bootstrap the judgment set for
CIs on selection_prob (§6.2). Falls back to a win-rate estimate if choix fitting
is unstable (§3).
"""

from __future__ import annotations

from dataclasses import dataclass

import choix
import numpy as np

# A comparison is (winner_index, loser_index) into the passage_ids list.
Comparison = tuple[int, int]


@dataclass
class BTResult:
    passage_id: str
    bt_strength: float  # Bradley–Terry log-strength
    selection_prob: float  # softmax over strengths within the candidate set
    rank: int  # 1 = most likely to be selected
    ci_low: float
    ci_high: float  # bootstrap CI on selection_prob


def _softmax(strengths: np.ndarray) -> np.ndarray:
    shifted = strengths - strengths.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


def _winrate_strengths(n: int, comparisons: list[Comparison]) -> np.ndarray:
    """Fallback strengths from smoothed win rates when choix fitting fails."""
    wins = np.zeros(n)
    games = np.zeros(n)
    for w, loser in comparisons:
        wins[w] += 1
        games[w] += 1
        games[loser] += 1
    rate = (wins + 0.5) / (games + 1.0)
    return np.log(rate / (1.0 - rate))


def _fit_strengths(n: int, comparisons: list[Comparison], alpha: float) -> np.ndarray:
    """Fit BT strengths; fall back to win-rate log-odds if choix is unstable."""
    if not comparisons:
        return np.zeros(n)
    try:
        return choix.ilsr_pairwise(n, comparisons, alpha=alpha)
    except Exception:  # noqa: BLE001 - choix can fail on degenerate data (§3 fallback)
        return _winrate_strengths(n, comparisons)


def fit_bradley_terry(
    passage_ids: list[str],
    comparisons: list[Comparison],
    *,
    alpha: float = 0.01,
    bootstrap_resamples: int = 200,
    rng_seed: int = 0,
) -> list[BTResult]:
    """Fit BT strengths and bootstrap CIs on selection_prob (§6.2).

    Ranking is by strength (desc). CIs come from resampling the comparison list
    with replacement and recomputing selection_prob. rng_seed keeps runs
    reproducible (§5).
    """
    n = len(passage_ids)
    if n == 0:
        return []

    strengths = _fit_strengths(n, comparisons, alpha)
    probs = _softmax(strengths)

    # Bootstrap CIs on selection_prob.
    if comparisons and bootstrap_resamples > 0:
        rng = np.random.default_rng(rng_seed)
        m = len(comparisons)
        samples: list[np.ndarray] = []
        for _ in range(bootstrap_resamples):
            idx = rng.integers(0, m, size=m)
            resampled = [comparisons[i] for i in idx]
            s = _fit_strengths(n, resampled, alpha)
            samples.append(_softmax(s))
        stacked = np.vstack(samples)
        ci_low = np.percentile(stacked, 2.5, axis=0)
        ci_high = np.percentile(stacked, 97.5, axis=0)
    else:
        ci_low = probs.copy()
        ci_high = probs.copy()

    # Rank by strength, descending (1-based).
    order = np.argsort(-strengths)
    rank_of = {int(i): r + 1 for r, i in enumerate(order)}

    return [
        BTResult(
            passage_id=passage_ids[i],
            bt_strength=float(strengths[i]),
            selection_prob=float(probs[i]),
            rank=rank_of[i],
            ci_low=float(ci_low[i]),
            ci_high=float(ci_high[i]),
        )
        for i in range(n)
    ]
