"""Fetch + trafilatura extraction + chunking.

Fetch each URL (httpx, realistic UA, robots.txt respected, per-domain politeness,
7-day HTML cache), extract main content with trafilatura, and chunk into 200–400
word passages with heading context prepended (§6.1). Scaffolded in Milestone 1;
implemented in Milestone 2.
"""

from __future__ import annotations

from .config import ChunkingConfig
from .models import Passage


def fetch_html(url: str) -> str:
    """Fetch raw HTML with caching, retries, and robots.txt respect (§6.1)."""
    raise NotImplementedError("fetch_html lands with Milestone 2.")


def extract_main_text(html: str) -> str:
    """Extract main content via trafilatura (§6.1)."""
    raise NotImplementedError("extract_main_text lands with Milestone 2.")


def chunk_passages(
    text: str,
    *,
    url: str,
    domain: str,
    title: str,
    is_ours: bool,
    cfg: ChunkingConfig,
) -> list[Passage]:
    """Split into 200–400 word windows with heading context (§6.1)."""
    raise NotImplementedError("chunk_passages lands with Milestone 2.")
