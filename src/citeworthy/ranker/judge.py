"""Pairwise judge (Anthropic API).

The judge is the source-selection component of an AI answer engine. It takes an
injectable transport so tests use canned responses (§10). The model is injectable
so multi-model judging (§12.5) is a natural extension. JSON contract in §6.3 is
load-bearing. Scaffolded in Milestone 1; implemented in Milestone 3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JudgeVerdict:
    """Parsed judge output (§6.3 JSON contract)."""

    winner: str  # "A" | "B"
    confidence: str  # "clear" | "moderate" | "slight"
    reason_code: str  # from REASON_CODES
    reason: str
    input_tokens: int = 0
    output_tokens: int = 0


def judge_pair(
    query_text: str,
    passage_a_text: str,
    passage_b_text: str,
    *,
    domain_a: str,
    domain_b: str,
    model: str,
    temperature: float,
    show_domains: bool,
) -> JudgeVerdict:
    """Judge which passage an engine would ground on (§6.3)."""
    raise NotImplementedError("judge_pair lands with Milestone 3.")
