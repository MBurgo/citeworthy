"""Citation-share statistics: domains, Wilson intervals, rolling trend (§6.6)."""

from __future__ import annotations

from datetime import datetime, timezone

from citeworthy.models import CitationObservation
from citeworthy.truth.share import (
    compute_shares,
    domain_share,
    intervals_overlap,
    to_domain,
    trend_vs_baseline,
    wilson_interval,
)


def _obs(urls, i=0, engine="perplexity", query="q"):
    return CitationObservation(
        run_id="r", query_id=query, engine=engine, sample_index=i,
        cited_urls=urls, answer_text_hash="h", observed_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
    )


def test_to_domain_normalises():
    assert to_domain("https://www.fool.com.au/asx-200/?utm_source=x") == "fool.com.au"
    assert to_domain("fool.com.au") == "fool.com.au"
    assert to_domain("http://marketindex.com.au:443/asx200") == "marketindex.com.au"


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.4  # zero successes still has an upper bound


def test_citation_share_counts_each_sample_once():
    # Domain cited twice in one sample still counts as one hit for that sample.
    samples = [
        _obs(["https://fool.com.au/a", "https://fool.com.au/b", "https://rival.com/x"]),
        _obs(["https://rival.com/y"], i=1),
        _obs(["https://fool.com.au/c"], i=2),
    ]
    fool = domain_share(samples, "fool.com.au")
    assert fool.successes == 2 and fool.n_samples == 3
    assert abs(fool.share - 2 / 3) < 1e-9

    shares = {s.domain: s for s in compute_shares(samples)}
    assert shares["fool.com.au"].successes == 2
    assert shares["rival.com"].successes == 2


def test_intervals_overlap():
    a = domain_share([_obs(["fool.com.au"]) for _ in range(10)], "fool.com.au")  # 100%
    b = domain_share([_obs(["rival.com"]) for _ in range(10)], "fool.com.au")    # 0%
    assert not intervals_overlap(a, b)  # 100% vs 0% clearly separated


def test_trend_insufficient_for_single_run():
    run = [_obs(["fool.com.au"], i) for i in range(10)]
    verdict = trend_vs_baseline([run], "fool.com.au")
    assert verdict.insufficient is True
    assert verdict.current is not None


def test_trend_flags_non_overlapping_change():
    # Baseline: fool cited 0/10. Later runs: fool cited 10/10. Clear rise.
    baseline = [_obs(["rival.com"], i) for i in range(10)]
    up1 = [_obs(["fool.com.au"], i) for i in range(10)]
    up2 = [_obs(["fool.com.au"], i) for i in range(10)]
    up3 = [_obs(["fool.com.au"], i) for i in range(10)]
    verdict = trend_vs_baseline([baseline, up1, up2, up3], "fool.com.au", baseline_runs=1)
    assert verdict.insufficient is False
    assert verdict.changed is True
    assert verdict.current.share > verdict.baseline.share


def test_trend_needs_runs_beyond_baseline_window():
    # Default baseline window is 3 runs; with only 3 the baseline and current
    # windows coincide, so no trend can be claimed yet.
    runs = [[_obs(["fool.com.au"], i) for i in range(10)] for _ in range(3)]
    assert trend_vs_baseline(runs, "fool.com.au").insufficient is True
    # A 4th run separates baseline (first 3) from the current 3-run window.
    runs.append([_obs(["rival.com"], i) for i in range(10)])
    assert trend_vs_baseline(runs, "fool.com.au").insufficient is False


def test_trend_no_change_when_overlapping():
    a = [_obs(["fool.com.au"] if i < 5 else ["rival.com"], i) for i in range(10)]
    b = [_obs(["fool.com.au"] if i < 5 else ["rival.com"], i) for i in range(10)]
    verdict = trend_vs_baseline([a, b], "fool.com.au", baseline_runs=1)
    assert verdict.changed is False  # both ~50%, intervals overlap
