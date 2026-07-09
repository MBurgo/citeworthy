"""SerpProvider protocol.

Pluggable interface so DataForSEO can be added alongside SerpApi later (§3, §6.1).
Concrete providers implement `search`. Scaffolded in Milestone 1; the first
implementation (serpapi.py) lands with Milestone 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SerpResult:
    """Normalised SERP payload from any provider."""

    query: str
    organic_urls: list[str] = field(default_factory=list)
    people_also_ask: list[str] = field(default_factory=list)
    raw: dict | None = None


@runtime_checkable
class SerpProvider(Protocol):
    def search(self, query: str, *, gl: str, hl: str, location: str, top_n: int) -> SerpResult:
        """Return the top organic URLs (and PAA questions) for a query."""
        ...
