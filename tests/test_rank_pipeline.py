"""End-to-end `rank` pipeline on fixtures — no live API calls (§10)."""

from __future__ import annotations

import re

import httpx

from citeworthy import db, report
from citeworthy.config import load_config
from citeworthy.models import Run, TargetQuery
from citeworthy.pipeline import rank_query
from citeworthy.ranker.judge import Judge, TransportResponse
from citeworthy.serp.base import SerpResult

SOURCE_RE = re.compile(r"source: ([^)]+)\)")

# Our domain is weakest, so our passages lose and the diagnosis is non-empty.
DOMAIN_STRENGTH = {
    "marketindex.com.au": 30,
    "commsec.com.au": 25,
    "example-broker.com.au": 20,
    "fool.com.au": 5,
}


class FakeProvider:
    def __init__(self, urls, paa):
        self._urls = urls
        self._paa = paa

    def search(self, query, *, gl, hl, location, top_n):
        return SerpResult(
            query=query,
            organic_urls=list(self._urls[:top_n]),
            people_also_ask=list(self._paa),
        )


class DomainStrengthTransport:
    """Deterministic judge: prefers the passage whose source domain is stronger."""

    def complete(self, *, system, user, model, temperature, max_tokens):
        da, dbm = SOURCE_RE.findall(user)[:2]
        sa = DOMAIN_STRENGTH.get(da.strip(), 10)
        sb = DOMAIN_STRENGTH.get(dbm.strip(), 10)
        winner = "A" if sa >= sb else "B"
        text = (
            f'{{"winner":"{winner}","confidence":"clear",'
            f'"reason_code":"too_generic","reason":"lacks specific figures"}}'
        )
        return TransportResponse(text, input_tokens=1400, output_tokens=120)


def _routing_client(serpapi_json, article_html) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, text=article_html)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _query():
    return TargetQuery(
        id="asx-200", text="ASX 200", tier="money",
        our_url="https://www.fool.com.au/asx-200-explained/",
    )


def _run_pipeline(tmp_path, serpapi_json, article_html, conn=None):
    urls = [r["link"] for r in serpapi_json["organic_results"]]
    paa = [q["question"] for q in serpapi_json["related_questions"]]
    provider = FakeProvider(urls, paa)
    judge = Judge(DomainStrengthTransport(), model="claude-haiku-4-5-20251001", temperature=0.3)
    client = _routing_client(serpapi_json, article_html)
    cfg = load_config("config.yaml")
    return rank_query(
        _query(), cfg,
        provider=provider, judge=judge, client=client,
        run_id="run-test", conn=conn, cache_dir=tmp_path / "cache",
        write_report_file=False,
    )


def test_rank_query_end_to_end(serpapi_json, article_html, tmp_path):
    out = _run_pipeline(tmp_path, serpapi_json, article_html)

    # Ranks are dense 1..n and unique.
    n = len(out.rank_results)
    assert n > 0
    assert sorted(r.rank for r in out.rank_results) == list(range(1, n + 1))
    # selection_prob is a distribution.
    assert abs(sum(r.selection_prob for r in out.rank_results) - 1.0) < 1e-6

    # Our passages are present and, being weakest, don't lead.
    passages = {p.id: p for p in out.candidate_set.passages}
    ours = [r for r in out.rank_results if passages[r.passage_id].is_ours]
    assert ours, "expected our passages in the set"
    leader = min(out.rank_results, key=lambda r: r.rank)
    assert not passages[leader.passage_id].is_ours

    assert out.total_cost_usd > 0

    md = out.report_md
    assert "# Citeworthy — ASX 200" in md
    assert "## Ranked passages" in md
    assert "## Loss diagnosis" in md
    assert "fool.com.au" in md
    assert "too_generic" in md  # our diagnosed weakness


def test_rank_persists_and_report_reads_back(serpapi_json, article_html, tmp_path, monkeypatch):
    db_path = tmp_path / "cw.db"
    db.init_db(db_path)
    conn = db.connect(db_path)
    try:
        from datetime import datetime

        db.create_run(conn, Run(run_id="run-test", kind="rank",
                                 started_at=datetime(2026, 7, 8), config_hash="h"))
        out = _run_pipeline(tmp_path, serpapi_json, article_html, conn=conn)

        # rank_results and matchups are persisted under the run.
        assert db.latest_rank_run_id(conn, "asx-200") == "run-test"
        stored = db.get_rank_results(conn, "run-test", "asx-200")
        assert len(stored) == len(out.rank_results)
        assert db.get_matchups(conn, "run-test", "asx-200")

        # Standalone `report` re-renders from the db.
        monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
        db.upsert_query(conn, _query())  # report needs the query row
        md, path = report.render_query_report(conn, "asx-200")
        assert "# Citeworthy — ASX 200" in md
        assert path.exists()
    finally:
        conn.close()
