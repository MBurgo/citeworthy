"""Loss-reason aggregation.

Aggregate reason_code across every matchup our passages lost; report the
distribution, top 3 weaknesses, verbatim examples, and the buried-answer gap
(our best chunk vs. our page's first chunk) (§6.4). Scaffolded in Milestone 1;
implemented in Milestone 4.
"""

from __future__ import annotations

from ..models import Matchup


def diagnose_losses(matchups: list[Matchup], our_passage_ids: set[str]) -> dict:
    """Return the loss-reason diagnosis for our passages (§6.4)."""
    raise NotImplementedError("diagnose_losses lands with Milestone 4.")
