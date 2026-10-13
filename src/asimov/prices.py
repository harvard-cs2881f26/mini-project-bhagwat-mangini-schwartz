"""Prices (configs/prices.yaml) and the cost of one call: the one home for both (docs/PLAN.md §9.2, §9.5).

Cost = logged tokens x list price per million tokens. Gemini bills thinking tokens as output but saves them as
`usage.thinking_tokens`, apart from `output_tokens`, so an entry with `thinking_as_output` adds them (WORKLOG
2026-10-12, "Cost method"). Free-tier services cost $0; a self-hosted model has no $ price (latency and
parameters are reported instead), so its cost is None.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

PRICES_PATH = Path(__file__).resolve().parents[2] / "configs" / "prices.yaml"
PER_MTOK = 1e6


class Price(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cached_input_per_mtok: float | None = None  # default: the input price
    thinking_as_output: bool = False
    free_tier: bool = False
    self_hosted: str | None = None  # where it runs, e.g. "Modal L40S"
    params_b: float | None = None
    active_params_b: float | None = None

    @model_validator(mode="after")
    def one_way_to_bill(self) -> Price:
        token_prices = [self.input_per_mtok, self.output_per_mtok, self.cached_input_per_mtok]
        if self.free_tier or self.self_hosted:
            if self.free_tier and self.self_hosted:
                raise ValueError("an entry is free_tier or self_hosted, not both")
            if any(p is not None for p in token_prices):
                raise ValueError("a free_tier or self_hosted entry has no token prices")
        elif self.input_per_mtok is None or self.output_per_mtok is None:
            raise ValueError("a token-priced entry needs input_per_mtok and output_per_mtok")
        return self


class PriceTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checked: date  # when the prices were read off the providers' pages
    sources: dict[str, str]
    models: dict[str, Price]


def load_prices(path: Path = PRICES_PATH) -> PriceTable:
    return PriceTable.model_validate(yaml.safe_load(path.read_text()))


def call_cost(usage: dict, price: Price) -> float | None:
    """$ for logged tokens (one call, or several summed); None for a self-hosted model."""
    if price.self_hosted:
        return None
    if price.free_tier:
        return 0.0
    assert price.input_per_mtok is not None and price.output_per_mtok is not None  # checked on load
    cached = usage.get("cached_input_tokens") or 0
    fresh = (usage.get("input_tokens") or 0) - cached
    output = (usage.get("output_tokens") or 0) + ((usage.get("thinking_tokens") or 0) if price.thinking_as_output else 0)
    cached_rate = price.input_per_mtok if price.cached_input_per_mtok is None else price.cached_input_per_mtok
    return (fresh * price.input_per_mtok + cached * cached_rate + output * price.output_per_mtok) / PER_MTOK
