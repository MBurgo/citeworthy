"""Compliance gate (hard requirement).

Every generated variant passes through a checker before entering the tournament.
Implemented as a pluggable hook so Matt's existing ASIC compliance checker can be
swapped in (§6.5). v2 ships a prompt-based checker. Scaffolded in Milestone 1;
implemented in Milestone 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ComplianceResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    notes: str = ""


@runtime_checkable
class ComplianceChecker(Protocol):
    def check(self, text: str) -> ComplianceResult:
        """Return whether `text` passes the compliance gate (§6.5)."""
        ...


class PromptComplianceChecker:
    """v2 prompt-based checker (no guarantees, no advice imperatives, etc.)."""

    def check(self, text: str) -> ComplianceResult:
        raise NotImplementedError("PromptComplianceChecker lands with Milestone 6.")
