"""SerpApiProvider against a canned response via MockTransport."""

from __future__ import annotations

import httpx

from citeworthy.serp.serpapi import SerpApiProvider


def _client(serpapi_json) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "serpapi.com" in str(request.url)
        assert request.url.params.get("q") == "ASX 200"
        assert request.url.params.get("gl") == "au"
        return httpx.Response(200, json=serpapi_json)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_returns_organic_and_paa(serpapi_json):
    provider = SerpApiProvider("fake-key", client=_client(serpapi_json))
    result = provider.search("ASX 200", gl="au", hl="en", location="Australia", top_n=10)

    assert result.organic_urls[0] == "https://www.marketindex.com.au/asx200"
    assert "https://www.fool.com.au/asx-200-explained/" in result.organic_urls
    assert len(result.people_also_ask) == 3
    assert result.people_also_ask[1] == "How many companies are in the ASX 200?"


def test_search_respects_top_n(serpapi_json):
    provider = SerpApiProvider("fake-key", client=_client(serpapi_json))
    result = provider.search("ASX 200", gl="au", hl="en", location="Australia", top_n=2)
    assert len(result.organic_urls) == 2
