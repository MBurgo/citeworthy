"""Judge parsing + prompt construction, using a canned transport (no live calls)."""

from __future__ import annotations

import pytest

from citeworthy.ranker.judge import Judge, JudgeError, TransportResponse


class CannedTransport:
    """Returns a fixed text; records the last (system, user) it was given."""

    def __init__(self, text: str, *, input_tokens: int = 1400, output_tokens: int = 120):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.last_user: str | None = None
        self.last_system: str | None = None

    def complete(self, *, system, user, model, temperature, max_tokens):
        self.last_system = system
        self.last_user = user
        return TransportResponse(self.text, self.input_tokens, self.output_tokens)


def _judge(text, **kw):
    transport = CannedTransport(text)
    return Judge(transport, model="claude-haiku-4-5-20251001", **kw), transport


def test_parses_clean_json():
    j, _ = _judge('{"winner":"A","confidence":"clear","reason_code":"buried_answer","reason":"B buries it"}')
    v = j.judge_pair("q", "a", "b", domain_a="x.com", domain_b="y.com")
    assert v.winner == "A"
    assert v.confidence == "clear"
    assert v.reason_code == "buried_answer"
    assert v.reason == "B buries it"
    assert v.input_tokens == 1400 and v.output_tokens == 120


def test_tolerates_markdown_fences_and_prose():
    text = 'Here is my answer:\n```json\n{"winner":"B","confidence":"slight","reason_code":"thin","reason":"too short"}\n```'
    j, _ = _judge(text)
    v = j.judge_pair("q", "a", "b", domain_a="x.com", domain_b="y.com")
    assert v.winner == "B"
    assert v.reason_code == "thin"


def test_invalid_reason_code_falls_back_to_other():
    j, _ = _judge('{"winner":"A","confidence":"moderate","reason_code":"made_up","reason":"x"}')
    v = j.judge_pair("q", "a", "b", domain_a="x.com", domain_b="y.com")
    assert v.reason_code == "other"


def test_bad_winner_raises():
    j, _ = _judge('{"winner":"C","confidence":"clear","reason_code":"thin","reason":"x"}')
    with pytest.raises(JudgeError):
        j.judge_pair("q", "a", "b", domain_a="x.com", domain_b="y.com")


def test_no_json_raises():
    j, _ = _judge("I cannot decide.")
    with pytest.raises(JudgeError):
        j.judge_pair("q", "a", "b", domain_a="x.com", domain_b="y.com")


def test_show_domains_toggles_prompt():
    text = '{"winner":"A","confidence":"clear","reason_code":"thin","reason":"x"}'
    j, t = _judge(text, show_domains=True)
    j.judge_pair("best ASX shares", "AA", "BB", domain_a="fool.com.au", domain_b="rival.com")
    assert "fool.com.au" in t.last_user and "rival.com" in t.last_user

    j2, t2 = _judge(text, show_domains=False)
    j2.judge_pair("best ASX shares", "AA", "BB", domain_a="fool.com.au", domain_b="rival.com")
    assert "fool.com.au" not in t2.last_user
    assert "withheld" in t2.last_user
