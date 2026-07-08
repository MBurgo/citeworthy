"""Shared test helpers. No live API calls anywhere (CLAUDE.md / §10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def article_html() -> str:
    return load_fixture("asx200_article.html")


@pytest.fixture
def short_html() -> str:
    return load_fixture("short_page.html")


@pytest.fixture
def serpapi_json() -> dict:
    return json.loads(load_fixture("serpapi_asx200.json"))
