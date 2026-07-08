"""Pair scheduling, position swap, sampling.

Full round-robin when candidate set <= 14; Swiss-style above that. Every pair
judged twice (A/B and B/A); disagreement recorded. Async with a semaphore,
exponential backoff on 429/529 (§6.2). Scaffolded in Milestone 1; implemented
in Milestone 3.
"""

from __future__ import annotations

from itertools import combinations

from ..models import Passage


def schedule_pairs(passages: list[Passage], round_robin_max: int) -> list[tuple[str, str]]:
    """Return the list of (passage_a_id, passage_b_id) pairs to judge (§6.2)."""
    raise NotImplementedError("schedule_pairs lands with Milestone 3.")


def _round_robin(ids: list[str]) -> list[tuple[str, str]]:
    return list(combinations(ids, 2))
