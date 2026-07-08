"""Gemini API citation collection with Google Search grounding (v3).

Read cited URIs from `groundingMetadata.groundingChunks`. Verify exact field
names against current docs at build time (§3, §12.3). Isolate parsing here; fail
loudly with the raw payload logged. Scaffolded in Milestone 1; implemented in
Milestone 5.
"""

from __future__ import annotations

from ..models import CitationObservation


def sample_citations(query_text: str, *, api_key: str, samples: int) -> list[CitationObservation]:
    """Collect grounded URLs across N Gemini samples (§6.6)."""
    raise NotImplementedError("Gemini grounding sampling lands with Milestone 5.")
