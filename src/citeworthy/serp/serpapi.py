"""SerpApi provider (serpapi.com, `google` engine, gl=au, hl=en).

Scaffolded in Milestone 1; implemented in Milestone 2. Do NOT scrape Google
directly (§3, §6.1).
"""

from __future__ import annotations

from .base import SerpProvider, SerpResult


class SerpApiProvider:
    """SerpProvider backed by serpapi.com."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(
        self, query: str, *, gl: str, hl: str, location: str, top_n: int
    ) -> SerpResult:
        raise NotImplementedError("SerpApiProvider.search lands with Milestone 2.")


# Structural check: SerpApiProvider satisfies the SerpProvider protocol.
_: type[SerpProvider] = SerpApiProvider  # type: ignore[assignment]
