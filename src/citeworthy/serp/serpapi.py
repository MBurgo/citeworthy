"""SerpApi provider (serpapi.com, `google` engine, gl=au, hl=en).

Takes an injectable httpx.Client so tests use canned responses via MockTransport
and never call the live API (§10). Do NOT scrape Google directly (§3, §6.1).
"""

from __future__ import annotations

import httpx

from .base import SerpProvider, SerpResult

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


class SerpApiProvider:
    """SerpProvider backed by serpapi.com."""

    def __init__(self, api_key: str, *, client: httpx.Client | None = None) -> None:
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=30.0)

    def search(
        self, query: str, *, gl: str, hl: str, location: str, top_n: int
    ) -> SerpResult:
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "google_domain": "google.com.au",
            "gl": gl,
            "hl": hl,
            "location": location,
            "num": top_n,
        }
        resp = self._client.get(SERPAPI_ENDPOINT, params=params)
        resp.raise_for_status()
        data = resp.json()

        organic = [
            r["link"]
            for r in data.get("organic_results", [])
            if isinstance(r, dict) and r.get("link")
        ][:top_n]

        # People Also Ask: SerpApi exposes these as `related_questions`.
        paa = [
            q["question"]
            for q in data.get("related_questions", [])
            if isinstance(q, dict) and q.get("question")
        ]

        return SerpResult(
            query=query, organic_urls=organic, people_also_ask=paa, raw=data
        )


# Structural check: SerpApiProvider satisfies the SerpProvider protocol.
_: type[SerpProvider] = SerpApiProvider  # type: ignore[assignment]
