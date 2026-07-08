"""Rewrite variant generation (§6.5).

Generate labelled rewrite variants of our passage, each targeting one diagnosed
weakness, preserving all facts and the Motley Fool AU voice. Uses the editor
model (claude-sonnet-4-6 by default) through an injectable transport so tests use
canned responses (§10). Verify the model name at build time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Passage
from ..ranker.judge import JudgeTransport

# What each diagnosed weakness (§6.3 reason_code) instructs the editor to fix.
STRATEGY_INSTRUCTIONS: dict[str, str] = {
    "buried_answer": "Move the direct answer into the very first sentence; lead with the conclusion.",
    "too_generic": "Foreground the concrete figures and named entities already in the text; be specific using only facts already present.",
    "no_evidence": "Surface the concrete evidence already in the passage (figures, dates, names) earlier and more prominently. Add nothing new.",
    "off_topic": "Reframe the opening so it answers the query directly; cut tangential material.",
    "hedgy": "Remove hedging and qualifiers; state the answer plainly (without adding any new claims).",
    "poor_structure": "Tighten the structure: short self-contained sentences, a clear topic sentence, logical order.",
    "stale_signals": "Foreground any explicit dates or current figures already in the text. Do not invent new ones.",
    "promotional": "Remove promotional tone; lead with informational substance.",
    "thin": "Draw out the substantive on-topic detail already implied; do not fabricate new detail.",
    "other": "Improve clarity and extractability while preserving every fact.",
}

DEFAULT_WEAKNESS = "buried_answer"

EDITOR_SYSTEM = """\
You are a senior editor for The Motley Fool Australia. You rewrite a single web
passage so an AI answer engine is more likely to cite it, WITHOUT changing its
meaning or facts.

Hard constraints:
- Preserve every factual claim exactly. Do NOT add new facts, numbers, dates, or
  data, and do NOT invent statistics. You may only rephrase what is already there.
- Preserve the original meaning.
- 200-400 words.
- State the direct, plain answer in the first two sentences.
- Keep The Motley Fool Australia's plain-English, jargon-light voice.
- Compliance: no guarantees or promises of returns, no unqualified performance
  claims, no advice-like imperatives ("you should buy"), and keep any
  past-performance disclaimer that is present in the original.

Output ONLY the rewritten passage text — no preamble, no headings, no quotes."""

EDITOR_USER_TEMPLATE = """\
Weakness to fix: {label} — {instruction}

Original passage:
\"\"\"
{text}
\"\"\""""


@dataclass
class Variant:
    label: str  # the weakness/strategy this variant targets
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


def _clean(text: str) -> str:
    """Strip stray surrounding quotes/fences an editor model might add."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    return t.strip().strip('"').strip()


def generate_variants(
    passage: Passage,
    weaknesses: list[str],
    *,
    transport: JudgeTransport,
    model: str,
    n_variants: int,
    max_tokens: int = 1024,
) -> list[Variant]:
    """Produce up to `n_variants` labelled rewrites, one per weakness (§6.5)."""
    # De-dupe while preserving order; fall back to a sensible default weakness.
    seen: set[str] = set()
    labels: list[str] = []
    for w in weaknesses:
        if w not in seen:
            seen.add(w)
            labels.append(w)
    if not labels:
        labels = [DEFAULT_WEAKNESS]
    labels = labels[:n_variants]

    variants: list[Variant] = []
    for label in labels:
        instruction = STRATEGY_INSTRUCTIONS.get(label, STRATEGY_INSTRUCTIONS["other"])
        user = EDITOR_USER_TEMPLATE.format(label=label, instruction=instruction, text=passage.text)
        resp = transport.complete(
            system=EDITOR_SYSTEM, user=user, model=model,
            temperature=None, max_tokens=max_tokens,
        )
        variants.append(
            Variant(
                label=label,
                text=_clean(resp.text),
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )
        )
    return variants
