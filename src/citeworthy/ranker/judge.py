"""Pairwise judge (Anthropic API).

The judge is the source-selection component of an AI answer engine (§6.3). It is
built around an injectable transport so tests use canned responses and never make
live calls (§10 / CLAUDE.md). The judge model is injectable so multi-model judging
(§12.5) is a natural extension.

The JSON contract in §6.3 is load-bearing: changing fields requires updating
models.py, db.py, and ranker/diagnose.py together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from ..models import REASON_CODES

# Model verified against https://docs.claude.com at build time (Milestone 3):
# claude-haiku-4-5-20251001 is current; $1.00/MTok input, $5.00/MTok output.

SYSTEM_PROMPT = """\
You are the source-selection component of an AI answer engine. Given a user
query and two candidate passages retrieved from the web, decide which single
passage you would ground your answer on (i.e., cite as a source).

Judge ONLY on the passage text provided. Weigh, in rough priority order:
1. Direct relevance: does it actually answer the query as asked?
2. Specificity and evidence: concrete facts, figures, dates, named entities,
   data — not generalities.
3. Extractability: is the answer stated plainly and early, in self-contained
   sentences an engine could quote or paraphrase cleanly?
4. Clarity and structure: unambiguous, well-organised, low fluff.
5. Freshness signals IN THE TEXT (explicit dates, current figures). Do not
   guess publication dates.
Ignore: brand familiarity, domain reputation, writing flair, length for its
own sake. Longer is not better.

Respond with ONLY a JSON object, no markdown fences:
{
  "winner": "A" | "B",
  "confidence": "clear" | "moderate" | "slight",
  "reason_code": one of ["off_topic","too_generic","buried_answer",
    "no_evidence","stale_signals","hedgy","poor_structure",
    "promotional","thin","other"],
  "reason": "<one sentence: why the loser lost>"
}
The reason_code describes the LOSING passage's primary weakness."""

USER_TEMPLATE = """\
Query: {query_text}

Passage A (source: {domain_a}):
\"\"\"
{passage_a_text}
\"\"\"

Passage B (source: {domain_b}):
\"\"\"
{passage_b_text}
\"\"\""""

WITHHELD = "withheld"
CONFIDENCE_LEVELS = ("clear", "moderate", "slight")


class JudgeError(Exception):
    """Raised when a judge response can't be parsed into the §6.3 contract."""


@dataclass
class TransportResponse:
    """Raw output from a transport: the model's text plus token usage."""

    text: str
    input_tokens: int
    output_tokens: int


class JudgeTransport(Protocol):
    """Anything that can turn a (system, user) prompt into a TransportResponse.

    Production wraps the Anthropic SDK; tests inject canned responses."""

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, max_tokens: int
    ) -> TransportResponse: ...


@dataclass
class JudgeVerdict:
    """Parsed judge output (§6.3 JSON contract) plus token accounting."""

    winner: str  # "A" | "B"
    confidence: str  # "clear" | "moderate" | "slight"
    reason_code: str  # from REASON_CODES
    reason: str
    input_tokens: int = 0
    output_tokens: int = 0


def _parse_verdict(text: str) -> tuple[str, str, str, str]:
    """Extract (winner, confidence, reason_code, reason) from a judge response,
    tolerating stray markdown fences or prose around the JSON object."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JudgeError(f"no JSON object in judge response: {text!r}")
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"invalid JSON in judge response: {exc}") from exc

    winner = str(obj.get("winner", "")).strip().upper()
    if winner not in ("A", "B"):
        raise JudgeError(f"winner must be 'A' or 'B', got {obj.get('winner')!r}")

    confidence = str(obj.get("confidence", "moderate")).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        confidence = "moderate"

    reason_code = str(obj.get("reason_code", "other")).strip().lower()
    if reason_code not in REASON_CODES:
        reason_code = "other"

    reason = str(obj.get("reason", "")).strip()
    return winner, confidence, reason_code, reason


class Judge:
    """Judges a single pair of passages via an injectable transport."""

    def __init__(
        self,
        transport: JudgeTransport,
        *,
        model: str,
        temperature: float = 0.3,
        show_domains: bool = True,
        max_tokens: int = 256,
    ) -> None:
        self.transport = transport
        self.model = model
        self.temperature = temperature
        self.show_domains = show_domains
        self.max_tokens = max_tokens

    def judge_pair(
        self,
        query_text: str,
        passage_a_text: str,
        passage_b_text: str,
        *,
        domain_a: str,
        domain_b: str,
    ) -> JudgeVerdict:
        """Decide which passage an answer engine would ground on (§6.3)."""
        user = USER_TEMPLATE.format(
            query_text=query_text,
            domain_a=domain_a if self.show_domains else WITHHELD,
            domain_b=domain_b if self.show_domains else WITHHELD,
            passage_a_text=passage_a_text,
            passage_b_text=passage_b_text,
        )
        resp = self.transport.complete(
            system=SYSTEM_PROMPT,
            user=user,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        winner, confidence, reason_code, reason = _parse_verdict(resp.text)
        return JudgeVerdict(
            winner=winner,
            confidence=confidence,
            reason_code=reason_code,
            reason=reason,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )


class AnthropicTransport:
    """Production transport backed by the Anthropic SDK.

    The SDK retries 429/5xx with exponential backoff automatically (§6.2); we
    bump max_retries. Not exercised in tests — those inject canned transports."""

    def __init__(self, api_key: str, *, client=None, max_retries: int = 4) -> None:
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, max_tokens: int
    ) -> TransportResponse:
        msg = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return TransportResponse(
            text=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
