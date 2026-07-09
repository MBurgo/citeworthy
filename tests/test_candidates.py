"""build_candidate_set end-to-end with a routing MockTransport (no live calls)."""

from __future__ import annotations

import httpx

from citeworthy.config import Config
from citeworthy.extract import build_candidate_set
from citeworthy.models import TargetQuery
from citeworthy.serp.serpapi import SerpApiProvider


def _routing_client(serpapi_json, article_html, *, robots_allow=True) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if "serpapi.com" in str(request.url):
            return httpx.Response(200, json=serpapi_json)
        if request.url.path == "/robots.txt":
            body = "User-agent: *\n" + ("Allow: /\n" if robots_allow else "Disallow: /\n")
            return httpx.Response(200, text=body)
        return httpx.Response(200, text=article_html)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _query() -> TargetQuery:
    return TargetQuery(
        id="asx-200",
        text="ASX 200",
        tier="money",
        our_url="https://www.fool.com.au/asx-200-explained/",
    )


def test_build_candidate_set(serpapi_json, article_html, tmp_path):
    client = _routing_client(serpapi_json, article_html)
    provider = SerpApiProvider("fake-key", client=client)
    cfg = Config()

    cs = build_candidate_set(
        _query(), provider, cfg, client=client, cache_dir=tmp_path
    )

    assert cs.query_id == "asx-200"
    # People Also Ask captured but not used for ranking (§6.1 step 1).
    assert len(cs.people_also_ask) == 3
    # Our page contributes passages, flagged is_ours.
    ours = [p for p in cs.passages if p.is_ours]
    assert ours, "expected passages from our page"
    assert all(p.domain.endswith("fool.com.au") for p in ours)
    # Competitor passages present from at least one other domain.
    comp_domains = {p.domain for p in cs.passages if not p.is_ours}
    assert comp_domains
    assert cs.stats["total_passages"] == len(cs.passages)


def test_build_candidate_set_skips_robots_disallowed(serpapi_json, article_html, tmp_path):
    client = _routing_client(serpapi_json, article_html, robots_allow=False)
    provider = SerpApiProvider("fake-key", client=client)
    cfg = Config()

    cs = build_candidate_set(
        _query(), provider, cfg, client=client, cache_dir=tmp_path
    )

    assert cs.passages == []
    assert any("robots.txt disallowed" in w for w in cs.warnings)
