"""Perplexity API citation collection (v3).

Queries the `sonar` model family (OpenAI-compatible /chat/completions). Sources
come back as `search_results[].url` (newer) or a top-level `citations[]` array of
URL strings (older) — we parse both so a schema shift degrades gracefully rather
than silently dropping citations (§3, §12.3). Verify current model names/schema
against the docs at build time.

Takes an injectable httpx.Client so tests use canned responses (§10).
"""

from __future__ import annotations

import httpx

from .base import EngineSample, TruthParseError

ENDPOINT = "https://api.perplexity.ai/chat/completions"


def extract_sources(payload: dict) -> list[str]:
    """Publisher URLs cited by a Perplexity response (§6.6)."""
    search_results = payload.get("search_results")
    if isinstance(search_results, list) and search_results:
        urls = [
            s["url"]
            for s in search_results
            if isinstance(s, dict) and isinstance(s.get("url"), str)
        ]
        if urls:
            return urls

    citations = payload.get("citations")
    if isinstance(citations, list):
        return [c for c in citations if isinstance(c, str)]

    # Neither field present in a shape we recognise: fail loudly with the payload.
    if "choices" not in payload:
        raise TruthParseError("Perplexity response has no choices/citations", payload)
    return []  # a grounded-but-uncited answer is a legitimate empty observation


def extract_answer(payload: dict) -> str:
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


class PerplexityClient:
    engine = "perplexity"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "sonar",
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=60.0)

    def sample_once(self, query_text: str) -> EngineSample:
        """One stochastic Perplexity call (temperature default — §6.6)."""
        resp = self._client.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": query_text}],
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        return EngineSample(
            cited_urls=extract_sources(payload),
            answer_text=extract_answer(payload),
        )
