"""The paper's reported numbers (configs/paper/claims.yaml), checked on load.

The one home for the paper's numbers (docs/PLAN.md §9.2): figure read-offs and numbers quoted in
the text for Figs. 4-6, plus its three headline claims. Written once by the paper-reader subagent
from the paper alone, reviewed by Noa, then frozen. Plots, run pages and the paper check read it.

Paper model names are joined to our configs' `name` by `model_key`, so "GPT 5 Mini" (paper) and
"GPT-5 Mini" (ours), or "Claude Opus 4.1" and our stand-in "Claude Opus 4.1 (4.5)", meet.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

CLAIMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "paper" / "claims.yaml"

# Scale of each value as the paper prints it; collect.py converts "percent" to a fraction.
SCALE_RANGE = {"fraction": (0.0, 1.0), "percent": (0.0, 100.0), "seconds": (0.0, 60.0)}
STAND_IN = re.compile(r"\s*\(([^)]*)\)\s*$")  # "Claude Opus 4.1 (4.5)": our model stands in for the paper's


def model_key(name: str) -> str:
    """Join key for a model name: paper name only, lowercase, letters and digits ("gpt5mini")."""
    return re.sub(r"[^a-z0-9]", "", STAND_IN.sub("", name).lower())


def is_stand_in(name: str) -> bool:
    """True for a config name like "Claude Opus 4.1 (4.5)", where our model isn't the paper's."""
    return bool(STAND_IN.search(name))


class Paper(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arxiv: str
    version: str
    title: str
    read_on: str  # YYYY-MM-DD


class Claim(BaseModel):
    """One number the paper reports, for one model (or an average over models when model is None)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    figure: str | None  # "4a", "5b", "6a"...; None for a number that appears only in the text
    section: str  # where it appears, e.g. "4.1" or "Appendix D"
    benchmark: Literal["text", "video", "image"]
    model: str | None  # the paper's model name as printed; None for an average over models
    metric: str  # our summary.json key, e.g. "latent_risk_accuracy"
    value: float
    scale: Literal["fraction", "percent", "seconds"]
    # what `value` is. value: one model's metric. mean / max / min: over `models`.
    # gap: `versus` minus `model` (how far `model` is below `versus`), same metric; with `versus`
    # None it isn't checkable by code (e.g. text minus video, see note). bound: a limit the paper
    # states ("above 90%"), over `models` if listed. baseline: derived from the paper's counts, no model.
    kind: Literal["value", "mean", "max", "min", "gap", "bound", "baseline"]
    models: list[str] = []  # paper names a mean / max / min / bound covers
    versus: str | None = None  # gap only: the model `model` is compared against
    source: Literal["figure", "text"]  # read off a figure, or stated in the text
    quote: str  # the paper's words (text), or where on the figure it was read (figure)
    approx: bool  # True when the value can't be read exactly (most figure read-offs)
    note: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Claim:
        lo, hi = SCALE_RANGE[self.scale]
        if not lo <= self.value <= hi:
            raise ValueError(f"{self.id}: value {self.value} outside {self.scale} range [{lo}, {hi}]")
        if self.kind in ("value", "gap") and self.model is None:
            raise ValueError(f"{self.id}: a {self.kind} claim needs a model")
        if self.kind in ("mean", "max", "min") and (self.model is not None or not self.models):
            raise ValueError(f"{self.id}: a {self.kind} claim lists its `models` and has no single `model`")
        if self.versus is not None and self.kind != "gap":
            raise ValueError(f"{self.id}: only a gap claim has `versus`")
        return self


class Headline(BaseModel):
    """One of the paper's headline claims, with the models it names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    # which computed check collect.py shows under it (docs/PLAN.md §3, claims a-c):
    # close_to_paper = ours vs each paper value; size_gap = large minus small, paired;
    # actions_harder = effect/activated-risk accuracy minus latent-risk/severity accuracy, per model
    check: Literal["close_to_paper", "size_gap", "actions_harder"]
    section: str
    quote: str
    models: list[str] = []  # close_to_paper / actions_harder: the models the claim names
    pairs: list[tuple[str, str]] = []  # size_gap: (large, small) pairs, as the paper pairs them
    note: str | None = None


class Claims(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper: Paper
    claims: list[Claim]
    headline: list[Headline]

    @model_validator(mode="after")
    def _unique_ids(self) -> Claims:
        ids = [c.id for c in self.claims] + [h.id for h in self.headline]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate claim ids: {dupes}")
        return self


def load_claims(path: Path = CLAIMS_PATH) -> Claims:
    return Claims.model_validate(yaml.safe_load(path.read_text()))


def as_fraction(claim: Claim) -> float:
    """The claim's value on our summaries' scale (fractions; seconds stay seconds)."""
    return claim.value / 100 if claim.scale == "percent" else claim.value
