"""Configuration models and loader.

config.yaml (validated here) holds all non-secret settings; secrets live in .env.
See §8 of CITEWORTHY_SPEC.md.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("config.yaml")


class LocaleConfig(BaseModel):
    gl: str = "au"
    hl: str = "en"
    location: str = "Australia"


class SerpConfig(BaseModel):
    provider: Literal["serpapi"] = "serpapi"
    top_n: int = 10


class ChunkingConfig(BaseModel):
    min_words: int = 200
    max_words: int = 400
    max_chunks_per_url: int = 6


class JudgeConfig(BaseModel):
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.3
    samples_per_order: int = 2
    show_domains: bool = True
    max_concurrency: int = 8


class EditorConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    variants: int = 4
    max_iterations: int = 3


class TournamentConfig(BaseModel):
    round_robin_max: int = 14
    bootstrap_resamples: int = 200


class TruthConfig(BaseModel):
    samples_per_engine: int = 10
    engines: list[Literal["perplexity", "gemini_grounded"]] = Field(
        default_factory=lambda: ["perplexity", "gemini_grounded"]
    )


class BudgetConfig(BaseModel):
    max_usd_per_rank_run: float = 2.00
    max_usd_per_track_run: float = 3.00


class ModelPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


class Config(BaseModel):
    locale: LocaleConfig = Field(default_factory=LocaleConfig)
    our_domain: str = "fool.com.au"
    serp: SerpConfig = Field(default_factory=SerpConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    editor: EditorConfig = Field(default_factory=EditorConfig)
    tournament: TournamentConfig = Field(default_factory=TournamentConfig)
    truth: TruthConfig = Field(default_factory=TruthConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)

    def config_hash(self) -> str:
        """Stable hash of the resolved config, recorded on every run (§5)."""
        payload = self.model_dump_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """USD cost estimate for a call, from the editable price table (§8, §11).

        Unknown models cost 0.0 (and should be added to config.yaml:pricing).
        """
        price = self.pricing.get(model)
        if price is None:
            return 0.0
        return (
            input_tokens / 1_000_000 * price.input_per_mtok
            + output_tokens / 1_000_000 * price.output_per_mtok
        )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate config.yaml. Missing file falls back to model defaults."""
    p = Path(path)
    if not p.exists():
        return Config()
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)


def api_key(name: str) -> str | None:
    """Fetch an API key from the environment (.env loaded at CLI startup).

    Returns None when absent so callers can degrade gracefully (§9).
    """
    value = os.environ.get(name)
    return value or None
