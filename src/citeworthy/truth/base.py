"""Shared types for the ground-truth tracker engines.

Each engine client turns one query into an EngineSample (the cited sources +
answer text for a single stochastic call). Parsing is isolated per engine and
fails loudly with the raw payload logged, because provider response schemas
change (§3, §12.3).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger("citeworthy.truth")


class TruthParseError(Exception):
    """Raised when an engine response can't be parsed. Carries the raw payload."""

    def __init__(self, message: str, payload: object) -> None:
        # Log the raw payload for audit; never overwrite/discard it silently (§12.3).
        try:
            log.error("%s | raw payload: %s", message, json.dumps(payload)[:4000])
        except (TypeError, ValueError):
            log.error("%s | raw payload (unserialisable): %r", message, payload)
        super().__init__(message)


@dataclass
class EngineSample:
    """One stochastic call's result: cited sources (as URLs or domains) + answer."""

    cited_urls: list[str] = field(default_factory=list)
    answer_text: str = ""


class EngineClient(Protocol):
    """A tracker engine that can sample citations for a query."""

    engine: str  # "perplexity" | "gemini_grounded"

    def sample_once(self, query_text: str) -> EngineSample: ...
