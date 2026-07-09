"""Citation share computation + statistics (§6.6).

Citation share (per query, engine, run) = the fraction of samples in which a
domain appears in the citation list at least once. With only ~10 samples per run,
a single week's movement is noise: we report shares with Wilson score intervals,
support a 3-run rolling window, and only flag a change from baseline when the
intervals don't overlap. Callers group observations by (query, engine, run)
before calling in.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlparse

from ..models import CitationObservation

# A single run of ~10 samples can't establish a trend (§6.6).
MIN_RUNS_FOR_TREND = 2
ROLLING_WINDOW = 3
Z_95 = 1.96


def to_domain(url_or_domain: str) -> str:
    """Normalise a URL or bare domain to a lower-cased registrable host,
    dropping scheme, path, query (tracking params), port, and a leading www."""
    s = url_or_domain.strip().lower()
    host = urlparse(s).netloc if "://" in s else s.split("/", 1)[0]
    host = host.split(":", 1)[0]  # strip port
    if host.startswith("www."):
        host = host[4:]
    return host


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (§6.6)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    return (max(0.0, low), min(1.0, high))


@dataclass
class ShareStat:
    domain: str
    successes: int  # samples citing this domain at least once
    n_samples: int
    share: float
    ci_low: float
    ci_high: float


def _domain_hits(observations: list[CitationObservation]) -> Counter[str]:
    hits: Counter[str] = Counter()
    for obs in observations:
        # Count each domain at most once per sample (§6.6 "at least once").
        for d in {to_domain(u) for u in obs.cited_urls if u}:
            hits[d] += 1
    return hits


def compute_shares(observations: list[CitationObservation]) -> list[ShareStat]:
    """Per-domain citation share (+ Wilson CI) across the given samples,
    sorted by share descending. Pass observations for one (query, engine)
    window — a single run, or several pooled for a rolling window."""
    n = len(observations)
    hits = _domain_hits(observations)
    stats: list[ShareStat] = []
    for d, c in hits.items():
        lo, hi = wilson_interval(c, n)
        stats.append(ShareStat(d, c, n, c / n if n else 0.0, lo, hi))
    stats.sort(key=lambda s: (-s.share, s.domain))
    return stats


def domain_share(observations: list[CitationObservation], domain: str) -> ShareStat:
    """Share stat for a specific domain (0 successes if never cited)."""
    target = to_domain(domain)
    n = len(observations)
    successes = _domain_hits(observations).get(target, 0)
    lo, hi = wilson_interval(successes, n)
    return ShareStat(target, successes, n, successes / n if n else 0.0, lo, hi)


def intervals_overlap(a: ShareStat, b: ShareStat) -> bool:
    return not (a.ci_high < b.ci_low or b.ci_high < a.ci_low)


@dataclass
class TrendVerdict:
    n_runs: int
    insufficient: bool  # too few runs to claim a trend (§6.6)
    current: ShareStat | None = None
    baseline: ShareStat | None = None
    changed: bool = False  # intervals don't overlap -> a real change


def rolling_window(
    runs_in_order: list[list[CitationObservation]], window: int = ROLLING_WINDOW
) -> list[CitationObservation]:
    """Pool the most recent `window` runs' observations."""
    pooled: list[CitationObservation] = []
    for run_obs in runs_in_order[-window:]:
        pooled.extend(run_obs)
    return pooled


def trend_vs_baseline(
    runs_in_order: list[list[CitationObservation]],
    domain: str,
    *,
    baseline_runs: int = ROLLING_WINDOW,
    window: int = ROLLING_WINDOW,
) -> TrendVerdict:
    """Compare a domain's recent rolling share to its baseline share, flagging a
    change only when the Wilson intervals don't overlap. Refuses to claim a trend
    from too few runs (§6.6 "insufficient data")."""
    n_runs = len(runs_in_order)
    # A trend needs the baseline (first runs) and current (last `window` runs) to
    # be separable — i.e. more runs than the baseline window — else they overlap
    # and any "change" is an artefact (§6.6).
    if n_runs < MIN_RUNS_FOR_TREND or n_runs <= baseline_runs:
        current = domain_share(rolling_window(runs_in_order, window), domain) if runs_in_order else None
        return TrendVerdict(n_runs=n_runs, insufficient=True, current=current)

    baseline_obs = [o for run in runs_in_order[:baseline_runs] for o in run]
    current_obs = rolling_window(runs_in_order, window)
    baseline = domain_share(baseline_obs, domain)
    current = domain_share(current_obs, domain)
    changed = not intervals_overlap(current, baseline)
    return TrendVerdict(
        n_runs=n_runs, insufficient=False,
        current=current, baseline=baseline, changed=changed,
    )
