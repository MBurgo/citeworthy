"""Bradley–Terry fitting + bootstrap on synthetic matrices (§10)."""

from __future__ import annotations

from itertools import combinations

from hypothesis import given, settings
from hypothesis import strategies as st

from citeworthy.ranker.bradley_terry import fit_bradley_terry


def _transitive_comparisons(n: int, repeats: int = 3):
    """Lower index always beats higher index (perfect transitive dominance)."""
    comps = []
    for i, j in combinations(range(n), 2):
        for _ in range(repeats):
            comps.append((i, j))  # i (stronger) beats j
    return comps


def test_recovers_true_order():
    ids = [f"p{i}" for i in range(5)]
    results = fit_bradley_terry(ids, _transitive_comparisons(5), bootstrap_resamples=50)
    by_id = {r.passage_id: r for r in results}
    # p0 is strongest -> rank 1, p4 weakest -> rank 5.
    ranks = [by_id[f"p{i}"].rank for i in range(5)]
    assert ranks == [1, 2, 3, 4, 5]
    # selection_prob is a proper distribution and monotone with strength.
    probs = [by_id[f"p{i}"].selection_prob for i in range(5)]
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs == sorted(probs, reverse=True)


def test_bootstrap_ci_brackets_and_separates():
    ids = [f"p{i}" for i in range(4)]
    results = fit_bradley_terry(ids, _transitive_comparisons(4, repeats=5), bootstrap_resamples=200)
    by_id = {r.passage_id: r for r in results}
    for r in results:
        assert r.ci_low <= r.ci_high
    # Clear leader's lower CI beats the loser's upper CI (non-overlapping).
    assert by_id["p0"].ci_low > by_id["p3"].ci_high


def test_empty_and_single():
    assert fit_bradley_terry([], []) == []
    single = fit_bradley_terry(["only"], [], bootstrap_resamples=10)
    assert len(single) == 1
    assert single[0].rank == 1
    assert abs(single[0].selection_prob - 1.0) < 1e-9


def test_reproducible_with_seed():
    ids = [f"p{i}" for i in range(4)]
    comps = _transitive_comparisons(4)
    a = fit_bradley_terry(ids, comps, rng_seed=42)
    b = fit_bradley_terry(ids, comps, rng_seed=42)
    assert [r.ci_low for r in a] == [r.ci_low for r in b]


@settings(max_examples=50, deadline=None)
@given(perm=st.permutations(range(5)))
def test_dominance_order_recovered_under_permutation(perm):
    # Assign strengths by the permutation: perm[k] is the item that sits at
    # dominance level k (0 = strongest). Stronger item always beats weaker.
    order = list(perm)  # order[level] = item id
    strength_rank = {item: level for level, item in enumerate(order)}  # lower = stronger
    comps = []
    for a, b in combinations(range(5), 2):
        winner, loser = (a, b) if strength_rank[a] < strength_rank[b] else (b, a)
        comps.append((winner, loser))

    ids = [f"p{i}" for i in range(5)]
    results = {int(r.passage_id[1:]): r for r in fit_bradley_terry(ids, comps, bootstrap_resamples=0)}
    # The item at dominance level 0 must have rank 1, level 1 -> rank 2, etc.
    for level, item in enumerate(order):
        assert results[item].rank == level + 1
