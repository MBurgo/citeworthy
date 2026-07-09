"""Extraction, chunking, and pre-filter tests, including chunker property tests."""

from __future__ import annotations

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from citeworthy.config import ChunkingConfig
from citeworthy.extract import (
    ExtractionFailed,
    RobotsDisallowed,
    chunk_passages,
    chunk_text,
    domain_of,
    extract_main_text,
    fetch_html,
    prefilter_chunks,
)


# --- extraction ------------------------------------------------------------


def test_extract_main_text_ok(article_html):
    md = extract_main_text(article_html, min_words=200, url="https://x.com/a")
    assert "ASX 200" in md
    assert md.count("#") >= 3  # headings preserved as markdown


def test_extract_short_page_fails(short_html):
    with pytest.raises(ExtractionFailed):
        extract_main_text(short_html, min_words=200, url="https://x.com/short")


def test_domain_of_strips_www():
    assert domain_of("https://www.fool.com.au/asx-200-explained/") == "fool.com.au"
    assert domain_of("https://marketindex.com.au/asx200") == "marketindex.com.au"


# --- chunk_passages --------------------------------------------------------


def test_chunk_passages_structure(article_html):
    md = extract_main_text(article_html, min_words=200, url="https://x.com/a")
    cfg = ChunkingConfig()
    passages = chunk_passages(
        md, url="https://x.com/a", domain="x.com", title="T", is_ours=False, cfg=cfg
    )
    assert passages, "expected at least one passage"
    # chunk_index is dense and 0-based; ids are unique.
    assert [p.chunk_index for p in passages] == list(range(len(passages)))
    assert len({p.id for p in passages}) == len(passages)
    # Heading context is prepended somewhere (e.g. "How the ASX 200 is calculated — ").
    assert any(" — " in p.text for p in passages)
    # No chunk exceeds the max word bound (heading prefix aside, bodies are bounded).
    for p in passages:
        assert len(p.text.split()) <= cfg.max_words + 12  # +heading words slack


# --- pre-filter ------------------------------------------------------------


def test_prefilter_keeps_top_k(article_html):
    md = extract_main_text(article_html, min_words=200, url="https://x.com/a")
    cfg = ChunkingConfig()
    passages = chunk_passages(
        md, url="https://x.com/a", domain="x.com", title="T", is_ours=False, cfg=cfg
    )
    kept = prefilter_chunks(passages, "ASX 200 calculation", keep=2)
    assert len(kept) == min(2, len(passages))
    # returned in reading order
    assert [p.chunk_index for p in kept] == sorted(p.chunk_index for p in kept)


# --- fetch: caching + robots ----------------------------------------------


def test_fetch_uses_cache(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, text="<html><body>hi</body></html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    url = "https://example.com/page"
    first = fetch_html(url, client=client, cache_dir=tmp_path, respect_robots=False)
    second = fetch_html(url, client=client, cache_dir=tmp_path, respect_robots=False)
    assert first == second
    assert calls["n"] == 1  # second call served from cache


def test_fetch_respects_robots_disallow(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /\n")
        return httpx.Response(200, text="<html><body>secret</body></html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RobotsDisallowed):
        fetch_html(
            "https://example.com/page", client=client, cache_dir=tmp_path
        )


# --- chunker property tests (§10) -----------------------------------------


@settings(max_examples=200)
@given(
    counts=st.lists(st.integers(min_value=1, max_value=250), min_size=1, max_size=10),
    min_words=st.integers(min_value=10, max_value=40),
    extra=st.integers(min_value=10, max_value=120),
)
def test_chunk_text_properties(counts, min_words, extra):
    max_words = min_words + extra

    # Build paragraphs from globally unique tokens so we can verify order + no drop.
    tokens: list[str] = []
    paragraphs: list[str] = []
    idx = 0
    for c in counts:
        para = " ".join(f"t{idx + j}" for j in range(c))
        paragraphs.append(para)
        tokens.extend(f"t{idx + j}" for j in range(c))
        idx += c

    windows = chunk_text(paragraphs, min_words, max_words)

    # 1. No text dropped, duplicated, or reordered.
    assert " ".join(windows).split() == tokens

    # 2. No window exceeds the upper bound.
    for w in windows:
        assert len(w.split()) <= max_words

    # 3. A window below min is only allowed when it couldn't merge forward
    #    (its combined size with the next window would exceed max), or it's last.
    for i in range(len(windows) - 1):
        wc = len(windows[i].split())
        if wc < min_words:
            assert wc + len(windows[i + 1].split()) > max_words


def test_chunk_text_single_short_paragraph():
    windows = chunk_text(["one two three four five"], min_words=200, max_words=400)
    assert windows == ["one two three four five"]  # short tail preserved, not dropped
