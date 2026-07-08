"""Hill-climbing controller (v2).

Generate variants -> compliance gate -> re-run tournament -> accept only if
selection_prob beats the original with non-overlapping CIs. Max 3 iterations
(§6.5). Scaffolded in Milestone 1; implemented in Milestone 6.
"""

from __future__ import annotations

from ..models import Passage


def run_edit_loop(query_id: str, *, max_iterations: int = 3) -> Passage:
    """Run the guided hill-climb and return the final recommended passage (§6.5)."""
    raise NotImplementedError("run_edit_loop lands with Milestone 6.")
