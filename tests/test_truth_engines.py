"""Perplexity + Gemini parsing and clients (canned responses, no live calls)."""

from __future__ import annotations

import httpx
import pytest

from citeworthy.truth import gemini, perplexity
from citeworthy.truth.base import TruthParseError


# --- Perplexity ------------------------------------------------------------


def test_perplexity_extracts_search_results(perplexity_json):
    urls = perplexity.extract_sources(perplexity_json)
    assert "https://www.fool.com.au/asx-200-explained/" in urls
    assert len(urls) == 3
    assert perplexity.extract_answer(perplexity_json).startswith("The S&P/ASX 200")


def test_perplexity_falls_back_to_citations():
    payload = {
        "choices": [{"message": {"content": "answer"}}],
        "citations": ["https://a.com", "https://b.com"],
    }
    assert perplexity.extract_sources(payload) == ["https://a.com", "https://b.com"]


def test_perplexity_empty_but_valid():
    payload = {"choices": [{"message": {"content": "no sources"}}]}
    assert perplexity.extract_sources(payload) == []


def test_perplexity_malformed_raises():
    with pytest.raises(TruthParseError):
        perplexity.extract_sources({"unexpected": "shape"})


def test_perplexity_client_sample(perplexity_json):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.perplexity.ai" in str(request.url)
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json=perplexity_json)

    client = perplexity.PerplexityClient("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    sample = client.sample_once("ASX 200")
    assert client.engine == "perplexity"
    assert "https://www.fool.com.au/asx-200-explained/" in sample.cited_urls


# --- Gemini ----------------------------------------------------------------


def test_gemini_prefers_title_domain(gemini_json):
    sources = gemini.extract_sources(gemini_json)
    # web.title holds the publisher domain; the uri is a Vertex redirect.
    assert sources == ["marketindex.com.au", "fool.com.au", "asx.com.au"]
    assert gemini.extract_answer(gemini_json).startswith("The S&P/ASX 200")


def test_gemini_falls_back_to_uri_when_title_not_domain():
    payload = {
        "candidates": [{
            "content": {"parts": [{"text": "x"}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://vertexaisearch.example/redirect/x", "title": "Some Article Title"}},
                ]
            },
        }]
    }
    assert gemini.extract_sources(payload) == ["https://vertexaisearch.example/redirect/x"]


def test_gemini_empty_candidates_is_valid():
    assert gemini.extract_sources({"candidates": []}) == []


def test_gemini_no_candidates_key_raises():
    with pytest.raises(TruthParseError):
        gemini.extract_sources({"promptFeedback": {"blockReason": "SAFETY"}})


def test_gemini_client_sample(gemini_json):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "generativelanguage.googleapis.com" in str(request.url)
        assert request.headers["x-goog-api-key"] == "k"
        body = request.read().decode()
        assert "google_search" in body
        return httpx.Response(200, json=gemini_json)

    client = gemini.GeminiClient("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    sample = client.sample_once("ASX 200")
    assert client.engine == "gemini_grounded"
    assert "fool.com.au" in sample.cited_urls
