"""Gemini API citation collection with Google Search grounding (v3).

Enables the `google_search` tool and reads cited sources from
`candidates[0].groundingMetadata.groundingChunks[].web`. The `web.uri` is a
Vertex grounding-redirect URL (not the publisher), while `web.title` usually
holds the publisher domain — so for domain aggregation we prefer `title` when it
looks like a hostname, falling back to the redirect URI. Verify field names
against current docs at build time (§3, §12.3).

Takes an injectable httpx.Client so tests use canned responses (§10).
"""

from __future__ import annotations

import re

import httpx

from .base import EngineSample, TruthParseError

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# A bare hostname like "reuters.com" or "www.afr.com.au" (no scheme, no path).
_DOMAIN_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$", re.IGNORECASE)


def _chunk_source(web: dict) -> str | None:
    """Best source string for a grounding chunk: the publisher domain from
    `title` when available, else the (redirect) `uri`."""
    title = web.get("title")
    if isinstance(title, str) and _DOMAIN_RE.match(title.strip()):
        return title.strip()
    uri = web.get("uri")
    if isinstance(uri, str) and uri:
        return uri
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def extract_sources(payload: dict) -> list[str]:
    """Grounded source domains/URIs for a Gemini response (§6.6)."""
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise TruthParseError("Gemini response has no candidates", payload)
    if not candidates:
        return []  # safety-blocked or empty: a legitimate empty observation

    metadata = candidates[0].get("groundingMetadata") or {}
    chunks = metadata.get("groundingChunks") or []
    sources: list[str] = []
    for chunk in chunks:
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if isinstance(web, dict):
            src = _chunk_source(web)
            if src:
                sources.append(src)
    return sources


def extract_answer(payload: dict) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    except (KeyError, IndexError, TypeError):
        return ""


class GeminiClient:
    engine = "gemini_grounded"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.5-flash",
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=60.0)

    def sample_once(self, query_text: str) -> EngineSample:
        """One stochastic Gemini call with Google Search grounding (§6.6)."""
        resp = self._client.post(
            f"{API_ROOT}/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json={
                "contents": [{"parts": [{"text": query_text}]}],
                "tools": [{"google_search": {}}],
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        return EngineSample(
            cited_urls=extract_sources(payload),
            answer_text=extract_answer(payload),
        )
