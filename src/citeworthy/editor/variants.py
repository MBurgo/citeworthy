"""Rewrite variant generation (claude-sonnet-4-6).

Generate 3–4 labelled variants targeting the top diagnosed weaknesses, each
preserving all factual claims exactly and the plain-English voice (§6.5).
Scaffolded in Milestone 1; implemented in Milestone 6.
"""

from __future__ import annotations

from ..models import Passage


def generate_variants(
    passage: Passage,
    weaknesses: list[str],
    *,
    model: str,
    n_variants: int,
) -> list[Passage]:
    """Produce labelled rewrite variants of a passage (§6.5)."""
    raise NotImplementedError("generate_variants lands with Milestone 6.")
