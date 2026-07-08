"""Fetch + trafilatura extraction + chunking + competitive-set builder.

Implements §6.1 of CITEWORTHY_SPEC.md:
  1. SERP query -> top organic URLs (+ PAA, stored not ranked).
  2. Add our_url if absent.
  3. Fetch each URL (httpx, 15s timeout, 2 retries, realistic UA, robots.txt
     respected, 7-day HTML cache).
  4. Extract main content with trafilatura; < min_words -> extraction_failed.
  5. Chunk into 200-400 word windows with heading context prepended, capped per
     URL by a cheap keyword-overlap + position pre-filter.
  6. For our page, keep ALL chunks.
  7. Aim for 12-20 passages; if over, keep each competitor's single best.

Networked functions take an injectable httpx.Client so tests use MockTransport
and never make live calls (§10).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura

from .config import ChunkingConfig, Config
from .models import Passage
from .serp.base import SerpProvider

log = logging.getLogger("citeworthy.extract")

# A realistic desktop UA so competitor pages don't block us; robots.txt is still
# respected against this UA (§6.1 step 3).
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

DEFAULT_CACHE_DIR = Path("cache")

# Tiny stopword set for the embed-free relevance pre-filter (§6.1 step 5).
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are",
    "with", "at", "by", "from", "as", "that", "this", "it", "be", "how", "what",
    "which", "best", "buy", "your", "you",
}


class ExtractionFailed(Exception):
    """Raised when trafilatura yields fewer than min_words (§6.1 step 4)."""


class RobotsDisallowed(Exception):
    """Raised when robots.txt disallows fetching a URL (§6.1 step 3)."""


class FetchError(Exception):
    """Raised when a URL cannot be fetched after retries."""


# --- Fetching --------------------------------------------------------------


def _cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path]:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{h}.html", cache_dir / f"{h}.meta.json"


def _cache_is_fresh(meta_path: Path, max_age_days: int, now: datetime) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
    except (ValueError, KeyError, OSError):
        return False
    age = now - fetched_at
    return age.days < max_age_days


class _RobotsGate:
    """Caches one RobotFileParser per host, fetched via the shared client."""

    def __init__(self, client: httpx.Client, ua: str) -> None:
        self._client = client
        self._ua = ua
        self._cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._cache.get(host)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(host, "/robots.txt")
            try:
                resp = self._client.get(robots_url)
                if resp.status_code >= 400:
                    rp.parse([])  # no robots -> allow all
                else:
                    rp.parse(resp.text.splitlines())
            except httpx.HTTPError:
                rp.parse([])  # unreachable robots -> allow all
            self._cache[host] = rp
        return rp.can_fetch(self._ua, url)


def fetch_html(
    url: str,
    *,
    client: httpx.Client,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    robots: _RobotsGate | None = None,
    respect_robots: bool = True,
    max_age_days: int = 7,
    retries: int = 2,
    now: datetime | None = None,
) -> str:
    """Fetch raw HTML with 7-day caching, retries, and robots.txt respect.

    Raises RobotsDisallowed / FetchError; callers log and skip (§6.1).
    """
    now = now or datetime.now(timezone.utc)

    if respect_robots:
        gate = robots or _RobotsGate(client, DEFAULT_UA)
        if not gate.allowed(url):
            raise RobotsDisallowed(url)

    html_path, meta_path = _cache_paths(cache_dir, url)
    if _cache_is_fresh(meta_path, max_age_days, now):
        try:
            return html_path.read_text(encoding="utf-8")
        except OSError:
            pass  # fall through and re-fetch

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.get(url)
            if resp.status_code >= 500:
                raise FetchError(f"{url} -> HTTP {resp.status_code}")
            resp.raise_for_status()
            html = resp.text
            cache_dir.mkdir(parents=True, exist_ok=True)
            html_path.write_text(html, encoding="utf-8")
            meta_path.write_text(
                json.dumps({"url": url, "fetched_at": now.isoformat()}),
                encoding="utf-8",
            )
            return html
        except (httpx.HTTPError, FetchError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(2**attempt * 0.1, 2.0))  # short backoff
    raise FetchError(f"failed to fetch {url}: {last_exc}")


# --- Extraction ------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(text.split())


def extract_main_text(html: str, *, min_words: int, url: str | None = None) -> str:
    """Extract main content as markdown (headings preserved). Raises
    ExtractionFailed when the result is shorter than min_words (§6.1 step 4)."""
    md = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if not md or _word_count(md) < min_words:
        got = _word_count(md) if md else 0
        raise ExtractionFailed(f"{url or 'page'}: {got} words < {min_words}")
    return md


def extract_title(html: str, fallback: str) -> str:
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and getattr(meta, "title", None):
            return str(meta.title)
    except Exception:  # noqa: BLE001 - metadata is best-effort
        pass
    return fallback


# --- Chunking --------------------------------------------------------------


def _parse_sections(md: str) -> list[tuple[str, list[str]]]:
    """Split markdown into (heading, [paragraphs]) sections. Content before the
    first heading gets an empty heading."""
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_paras: list[str] = []
    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current_paras:
                sections.append((current_heading, current_paras))
                current_paras = []
            current_heading = line.lstrip("#").strip()
        else:
            current_paras.append(line)
    if current_paras:
        sections.append((current_heading, current_paras))
    return sections


def _split_oversized(unit: str, max_words: int) -> list[str]:
    """Split a paragraph longer than max_words into <= max_words pieces,
    preferring sentence boundaries, hard-splitting on words as a last resort.
    Preserves every word token in order."""
    if _word_count(unit) <= max_words:
        return [unit]

    sentences = re.split(r"(?<=[.!?])\s+", unit)
    pieces: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for s in sentences:
        n = _word_count(s)
        if n > max_words:
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_n = [], 0
            words = s.split()
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i : i + max_words]))
        elif cur_n + n <= max_words:
            cur.append(s)
            cur_n += n
        else:
            pieces.append(" ".join(cur))
            cur, cur_n = [s], n
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_text(paragraphs: list[str], min_words: int, max_words: int) -> list[str]:
    """Merge paragraphs into 200-400 word windows.

    Guarantees: no window exceeds max_words; no word token is dropped or
    duplicated (order preserved); every window except possibly the last has at
    least min_words. The trailing remainder may be shorter when the section is
    short (§6.1 step 5 chunking).
    """
    # Normalise: no unit exceeds max_words.
    units: list[str] = []
    for para in paragraphs:
        units.extend(_split_oversized(para, max_words))

    windows: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for unit in units:
        n = _word_count(unit)
        if cur and cur_n + n > max_words:
            windows.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(unit)
        cur_n += n
        if cur_n >= min_words:
            windows.append(" ".join(cur))
            cur, cur_n = [], 0
    if cur:
        # Absorb a short tail into the previous window if it still fits.
        if windows and _word_count(windows[-1]) + cur_n <= max_words:
            windows[-1] = windows[-1] + " " + " ".join(cur)
        else:
            windows.append(" ".join(cur))
    return windows


def _passage_id(url: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{url}#{chunk_index}".encode("utf-8")).hexdigest()[:16]


def domain_of(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def chunk_passages(
    md_text: str,
    *,
    url: str,
    domain: str,
    title: str,
    is_ours: bool,
    cfg: ChunkingConfig,
) -> list[Passage]:
    """Turn extracted markdown into Passages with heading context prepended."""
    passages: list[Passage] = []
    index = 0
    for heading, paras in _parse_sections(md_text):
        for body in chunk_text(paras, cfg.min_words, cfg.max_words):
            prefix = f"{heading} — " if heading else ""
            passages.append(
                Passage(
                    id=_passage_id(url, index),
                    url=url,  # type: ignore[arg-type]
                    domain=domain,
                    title=title,
                    chunk_index=index,
                    text=prefix + body,
                    is_ours=is_ours,
                )
            )
            index += 1
    return passages


# --- Relevance pre-filter --------------------------------------------------


def _keywords(query_text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", query_text.lower())
    return {t for t in toks if len(t) > 2 and t not in _STOPWORDS}


def _score(passage: Passage, keywords: set[str]) -> float:
    if not keywords:
        overlap = 0.0
    else:
        chunk_toks = set(re.findall(r"[a-z0-9]+", passage.text.lower()))
        overlap = len(keywords & chunk_toks) / len(keywords)
    position_bonus = 1.0 / (1.0 + passage.chunk_index)
    return overlap + 0.15 * position_bonus


def rank_chunks(passages: list[Passage], query_text: str) -> list[Passage]:
    """Return passages sorted by relevance score (desc), then chunk_index."""
    keywords = _keywords(query_text)
    return sorted(passages, key=lambda p: (-_score(p, keywords), p.chunk_index))


def prefilter_chunks(passages: list[Passage], query_text: str, keep: int) -> list[Passage]:
    """Keep the top-`keep` chunks by relevance, returned in reading order."""
    top = rank_chunks(passages, query_text)[:keep]
    return sorted(top, key=lambda p: p.chunk_index)


# --- Competitive-set builder -----------------------------------------------


@dataclass
class CandidateSet:
    query_id: str
    passages: list[Passage] = field(default_factory=list)
    people_also_ask: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


TARGET_MIN = 12
TARGET_MAX = 20


def build_candidate_set(
    query,  # TargetQuery
    provider: SerpProvider,
    config: Config,
    *,
    client: httpx.Client,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    respect_robots: bool = True,
) -> CandidateSet:
    """Assemble the competitive passage set for a query (§6.1)."""
    loc = config.locale
    serp = provider.search(
        query.text, gl=loc.gl, hl=loc.hl, location=loc.location, top_n=config.serp.top_n
    )

    urls: list[str] = list(serp.organic_urls[: config.serp.top_n])
    our_url = str(query.our_url) if query.our_url else None
    if our_url and our_url not in urls:
        urls.append(our_url)

    result = CandidateSet(query_id=query.id, people_also_ask=list(serp.people_also_ask))
    gate = _RobotsGate(client, DEFAULT_UA) if respect_robots else None
    chunk_cfg = config.chunking

    ours: list[Passage] = []
    comp_ranked: list[list[Passage]] = []  # per-URL, sorted by relevance

    for url in urls:
        domain = domain_of(url)
        is_ours = (our_url is not None and url == our_url) or domain.endswith(
            config.our_domain
        )
        try:
            html = fetch_html(
                url, client=client, cache_dir=cache_dir, robots=gate,
                respect_robots=respect_robots,
            )
        except RobotsDisallowed:
            result.warnings.append(f"robots.txt disallowed: {url}")
            continue
        except FetchError as exc:
            result.warnings.append(f"fetch failed: {exc}")
            continue

        try:
            md = extract_main_text(html, min_words=chunk_cfg.min_words, url=url)
        except ExtractionFailed as exc:
            result.warnings.append(f"extraction_failed: {exc}")
            continue

        title = extract_title(html, fallback=domain)
        chunks = chunk_passages(
            md, url=url, domain=domain, title=title, is_ours=is_ours, cfg=chunk_cfg
        )
        if not chunks:
            result.warnings.append(f"no chunks produced: {url}")
            continue

        if is_ours:
            ours.extend(chunks)  # keep ALL our chunks (§6.1 step 6)
        else:
            comp_ranked.append(rank_chunks(chunks, query.text))

    # Per-competitor cap (§6.1 step 5), then assemble.
    max_per = chunk_cfg.max_chunks_per_url
    comp_selected: list[Passage] = []
    for ranked in comp_ranked:
        comp_selected.extend(sorted(ranked[:max_per], key=lambda p: p.chunk_index))

    # Budget: if over, keep each competitor's single best (§6.1 step 7).
    if len(ours) + len(comp_selected) > TARGET_MAX:
        comp_selected = [ranked[0] for ranked in comp_ranked if ranked]

    result.passages = ours + comp_selected
    result.stats = {
        "urls_considered": len(urls),
        "our_chunks": len(ours),
        "competitor_urls": len(comp_ranked),
        "competitor_chunks": len(comp_selected),
        "total_passages": len(result.passages),
        "warnings": len(result.warnings),
    }
    if len(result.passages) < TARGET_MIN:
        result.warnings.append(
            f"only {len(result.passages)} passages (target >= {TARGET_MIN}); "
            "candidate set may be thin."
        )
    return result
