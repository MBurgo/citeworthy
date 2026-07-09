"""Compliance gate (hard requirement, §6.5).

Every generated variant passes through a checker before it can enter the
tournament — no exceptions (CLAUDE.md). The checker is a pluggable hook so Matt's
existing ASIC compliance checker can be swapped in via the same interface.

v2 ships two implementations:
- RuleComplianceChecker: deterministic, no LLM call — the default gate. A hard
  compliance gate should not depend on a stochastic model, and it costs nothing.
  Enforces: no guarantees/promises of returns, no unqualified performance claims,
  no advice-like imperatives, no fabricated statistics (numbers absent from the
  original), and preservation of any past-performance disclaimer in the original.
- PromptComplianceChecker: the LLM-based checker from the spec, available for a
  nuanced second pass or when a model judgment is preferred.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..ranker.judge import JudgeTransport


@dataclass
class ComplianceResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    notes: str = ""


@runtime_checkable
class ComplianceChecker(Protocol):
    def check(self, text: str, *, original: str | None = None) -> ComplianceResult:
        """Return whether `text` passes the compliance gate (§6.5).

        `original` (the pre-edit passage) lets rules that are relative to the
        source run — fabricated-number and disclaimer-preservation checks. Matt's
        ASIC checker may ignore it."""
        ...


# --- Rule-based checker (deterministic default) ----------------------------

# Guarantees / promises of returns.
_GUARANTEE = re.compile(
    r"\b(guarantee[sd]?|promise[sd]?|assured?|risk[-\s]?free|can'?t\s+lose|"
    r"sure\s+thing|no\s+risk)\b",
    re.IGNORECASE,
)
# Unqualified performance predictions.
_PERFORMANCE = re.compile(
    r"\bwill\s+(?:definitely\s+|certainly\s+|surely\s+)?"
    r"(rise|soar|surge|double|triple|skyrocket|outperform|beat\s+the\s+market|"
    r"make\s+you\s+(?:money|rich))\b",
    re.IGNORECASE,
)
# Advice-like imperatives.
_IMPERATIVE = re.compile(
    r"\b(you\s+should\s+(buy|sell|invest|purchase)|"
    r"(buy|sell)\s+(this\s+stock\s+)?now|you\s+must\s+(buy|sell)|"
    r"we\s+recommend\s+(you\s+)?(buy|sell))\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_DISCLAIMER = re.compile(r"past\s+performance", re.IGNORECASE)


def _numbers(text: str) -> set[str]:
    # Normalise "1,200" -> "1200" so reformatting isn't flagged as fabrication.
    return set(_NUMBER.findall(text.replace(",", "")))


class RuleComplianceChecker:
    """Deterministic compliance gate (§6.5)."""

    def check(self, text: str, *, original: str | None = None) -> ComplianceResult:
        violations: list[str] = []

        if _GUARANTEE.search(text):
            violations.append("Contains a guarantee/promise of returns or a risk-free claim.")
        if _PERFORMANCE.search(text):
            violations.append("Contains an unqualified performance prediction.")
        if _IMPERATIVE.search(text):
            violations.append("Contains advice-like imperative language (e.g. 'you should buy').")

        if original is not None:
            new_numbers = _numbers(text) - _numbers(original)
            if new_numbers:
                sample = ", ".join(sorted(new_numbers)[:5])
                violations.append(
                    f"Introduces numbers not present in the original (possible "
                    f"fabricated statistics): {sample}."
                )
            if _DISCLAIMER.search(original) and not _DISCLAIMER.search(text):
                violations.append(
                    "Drops a past-performance disclaimer that was in the original."
                )

        return ComplianceResult(passed=not violations, violations=violations)


# --- Prompt-based checker (LLM, §6.5) --------------------------------------

_COMPLIANCE_SYSTEM = """\
You are a financial-content compliance reviewer for The Motley Fool Australia,
applying ASIC-aligned rules. Review the CANDIDATE passage and flag any of:
- guarantees or promises of returns
- unqualified performance claims (predicting a stock will rise/outperform, etc.)
- advice-like imperatives (e.g. "you should buy")
- fabricated statistics: numbers or data NOT present in the ORIGINAL passage
- a past-performance disclaimer present in the ORIGINAL but dropped in the CANDIDATE

Respond with ONLY a JSON object, no markdown fences:
{"passed": true|false, "violations": ["<one short sentence each>"]}"""


class PromptComplianceChecker:
    """LLM-based compliance checker (the spec's prompt-based gate)."""

    def __init__(self, transport: JudgeTransport, *, model: str, max_tokens: int = 512):
        self.transport = transport
        self.model = model
        self.max_tokens = max_tokens
        # Token accounting so the loop can log/charge this call (CLAUDE.md).
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def check(self, text: str, *, original: str | None = None) -> ComplianceResult:
        user = f"CANDIDATE:\n\"\"\"\n{text}\n\"\"\""
        if original is not None:
            user += f"\n\nORIGINAL:\n\"\"\"\n{original}\n\"\"\""
        resp = self.transport.complete(
            system=_COMPLIANCE_SYSTEM, user=user, model=self.model,
            temperature=None, max_tokens=self.max_tokens,
        )
        self.last_input_tokens = resp.input_tokens
        self.last_output_tokens = resp.output_tokens
        return _parse_compliance(resp.text)


def _parse_compliance(text: str) -> ComplianceResult:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        # Fail closed: an unparseable compliance response must not pass the gate.
        return ComplianceResult(passed=False, violations=["Unparseable compliance response."])
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return ComplianceResult(passed=False, violations=["Invalid JSON from compliance checker."])
    violations = [str(v) for v in obj.get("violations", []) if str(v).strip()]
    passed = bool(obj.get("passed", False)) and not violations
    return ComplianceResult(passed=passed, violations=violations)
