"""Compliance gate — the hard requirement (§6.5)."""

from __future__ import annotations

from citeworthy.editor.compliance import (
    ComplianceResult,
    PromptComplianceChecker,
    RuleComplianceChecker,
    _parse_compliance,
)
from citeworthy.ranker.judge import TransportResponse

CLEAN = (
    "The S&P/ASX 200 tracks the 200 largest companies on the ASX. It is "
    "float-adjusted and reviewed quarterly. Past performance is not a reliable "
    "indicator of future results."
)


def _check(text, original=None):
    return RuleComplianceChecker().check(text, original=original)


def test_clean_text_passes():
    assert _check(CLEAN, original=CLEAN).passed


def test_flags_guarantee():
    r = _check("This stock offers guaranteed returns with no risk.")
    assert not r.passed and any("guarantee" in v.lower() for v in r.violations)


def test_flags_performance_prediction():
    r = _check("This share will double next year.")
    assert not r.passed


def test_flags_advice_imperative():
    r = _check("You should buy this stock today.")
    assert not r.passed and any("imperative" in v.lower() for v in r.violations)


def test_flags_fabricated_numbers():
    original = "The ASX 200 holds 200 companies."
    r = _check("The ASX 200 holds 200 companies and returned 12.5% in 2025.", original=original)
    assert not r.passed
    assert any("12.5" in v or "2025" in v for v in r.violations)


def test_reformatted_numbers_pass():
    original = "The index rose 1,200 points."
    r = _check("The index rose 1200 points.", original=original)  # comma reformatting
    assert r.passed


def test_flags_dropped_disclaimer():
    original = "Shares can fall. Past performance is not a reliable indicator."
    r = _check("Shares can fall.", original=original)
    assert not r.passed and any("disclaimer" in v.lower() for v in r.violations)


def test_prompt_checker_parses_and_fails_closed():
    # Valid JSON pass.
    def ok(**kw):
        return TransportResponse('{"passed": true, "violations": []}', 10, 5)

    c = PromptComplianceChecker(type("T", (), {"complete": staticmethod(ok)})(), model="m")
    assert c.check("x", original="x").passed

    # Unparseable -> fails closed (must not pass the gate).
    assert _parse_compliance("the model rambled with no json").passed is False
    # Explicit violations override passed=true.
    assert _parse_compliance('{"passed": true, "violations": ["nope"]}').passed is False
