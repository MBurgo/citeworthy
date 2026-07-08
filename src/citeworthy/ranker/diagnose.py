"""Loss-reason aggregation (§6.4).

Aggregate reason_code across every matchup our passages lost, and surface the
top weaknesses with verbatim examples. The buried-answer gap (our best chunk vs
our page's first chunk) is computed in the report layer, where the rank results
are available alongside the matchups.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..models import Matchup


@dataclass
class Weakness:
    reason_code: str
    count: int
    examples: list[str] = field(default_factory=list)  # verbatim reason_text


@dataclass
class Diagnosis:
    total_losses: int = 0
    distribution: dict[str, int] = field(default_factory=dict)  # reason_code -> count
    top_weaknesses: list[Weakness] = field(default_factory=list)


def _loser(m: Matchup) -> str:
    """The passage that lost a matchup (the winner's opponent)."""
    return m.passage_a if m.winner == m.passage_b else m.passage_b


def diagnose_losses(
    matchups: list[Matchup],
    our_passage_ids: set[str],
    *,
    top_n: int = 3,
    examples_per_weakness: int = 3,
) -> Diagnosis:
    """Aggregate loss reasons across matchups our passages lost (§6.4).

    reason_code describes the losing passage's primary weakness (§6.3), so we
    only count matchups where one of our passages is the loser."""
    distribution: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for m in matchups:
        if _loser(m) in our_passage_ids:
            distribution[m.reason_code] += 1
            text = m.reason_text.strip()
            # Keep a few distinct, non-empty verbatim examples per weakness.
            if text and text not in examples[m.reason_code]:
                if len(examples[m.reason_code]) < examples_per_weakness:
                    examples[m.reason_code].append(text)

    top_weaknesses = [
        Weakness(reason_code=code, count=count, examples=examples.get(code, []))
        for code, count in distribution.most_common(top_n)
    ]

    return Diagnosis(
        total_losses=sum(distribution.values()),
        distribution=dict(distribution),
        top_weaknesses=top_weaknesses,
    )
