"""Variant generation with a canned editor transport (no live calls)."""

from __future__ import annotations

from citeworthy.editor.variants import EDITOR_SYSTEM, generate_variants
from citeworthy.models import Passage
from citeworthy.ranker.judge import TransportResponse


class EditorTransport:
    def __init__(self, text="Rewritten passage leading with the answer."):
        self.text = text
        self.last_user = None
        self.calls = 0

    def complete(self, *, system, user, model, temperature, max_tokens):
        self.calls += 1
        self.last_user = user
        assert temperature is None  # editor omits sampling params
        return TransportResponse(self.text, input_tokens=500, output_tokens=200)


def _passage():
    return Passage(id="p1", url="https://www.fool.com.au/asx/", domain="fool.com.au",
                   title="ASX 200", chunk_index=0, text="The ASX 200 tracks 200 companies.")


def test_one_variant_per_weakness_labelled():
    t = EditorTransport()
    variants = generate_variants(_passage(), ["buried_answer", "too_generic"],
                                 transport=t, model="claude-sonnet-4-6", n_variants=4)
    assert [v.label for v in variants] == ["buried_answer", "too_generic"]
    assert t.calls == 2
    assert all(v.text == "Rewritten passage leading with the answer." for v in variants)
    assert all(v.input_tokens == 500 for v in variants)


def test_dedups_and_caps_at_n_variants():
    t = EditorTransport()
    variants = generate_variants(_passage(), ["thin", "thin", "hedgy", "off_topic"],
                                 transport=t, model="m", n_variants=2)
    assert [v.label for v in variants] == ["thin", "hedgy"]  # deduped, capped at 2


def test_empty_weaknesses_uses_default():
    t = EditorTransport()
    variants = generate_variants(_passage(), [], transport=t, model="m", n_variants=4)
    assert len(variants) == 1
    assert variants[0].label == "buried_answer"


def test_prompt_carries_constraints_and_original():
    t = EditorTransport()
    generate_variants(_passage(), ["buried_answer"], transport=t, model="m", n_variants=1)
    assert "The ASX 200 tracks 200 companies." in t.last_user
    assert "Weakness to fix: buried_answer" in t.last_user
    assert "200-400 words" in EDITOR_SYSTEM  # constraints live in the system prompt


def test_strips_stray_fences():
    t = EditorTransport(text="```\nClean text.\n```")
    variants = generate_variants(_passage(), ["thin"], transport=t, model="m", n_variants=1)
    assert variants[0].text == "Clean text."
