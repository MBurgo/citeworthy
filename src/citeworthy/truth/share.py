"""Citation share computation + statistics.

Citation share (per query, engine, run) = fraction of samples in which the domain
appears at least once. Report 3-week rolling share with Wilson score intervals;
only flag changes where intervals don't overlap the baseline. Say "insufficient
data" rather than imply trends from one run (§6.6). Scaffolded in Milestone 1;
implemented in Milestone 5.
"""

from __future__ import annotations

from ..models import CitationObservation


def citation_share(observations: list[CitationObservation], domain: str) -> float:
    """Fraction of samples that cite `domain` at least once (§6.6)."""
    raise NotImplementedError("citation_share lands with Milestone 5.")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a citation-share proportion (§6.6)."""
    raise NotImplementedError("wilson_interval lands with Milestone 5.")
