"""Perplexity API citation collection (v3).

Query the `sonar` model family; responses include a citations/sources list.
Verify current model names and response field names against the docs at build
time — do not trust the spec's field names blindly (§3, §12.3). Isolate parsing
here; fail loudly with the raw payload logged. Scaffolded in Milestone 1;
implemented in Milestone 5.
"""

from __future__ import annotations

from ..models import CitationObservation


def sample_citations(query_text: str, *, api_key: str, samples: int) -> list[CitationObservation]:
    """Collect cited URLs across N Perplexity samples (§6.6)."""
    raise NotImplementedError("Perplexity sampling lands with Milestone 5.")
