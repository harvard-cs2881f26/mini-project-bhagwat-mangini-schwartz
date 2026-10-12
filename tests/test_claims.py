"""configs/paper/claims.yaml: valid, joinable to our configs and summaries."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from asimov.claims import Claims, is_stand_in, load_claims, model_key
from asimov.collect.metrics import DENOMINATORS

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"text": ROOT / "configs" / "repro_fig4.yaml", "video": ROOT / "configs" / "repro_fig5.yaml"}
BASELINE_PREFIX = "majority_baseline_"


@pytest.fixture(scope="module")
def claims() -> Claims:
    return load_claims()


def test_metrics_are_known_summary_keys(claims):
    for c in claims.claims:
        if c.kind == "baseline":
            assert c.metric.startswith(BASELINE_PREFIX), c.id
        else:
            assert c.metric in DENOMINATORS[c.benchmark], f"{c.id}: unknown metric {c.metric!r} for {c.benchmark}"


def test_paper_model_names_match_a_config_model(claims):
    for benchmark, path in CONFIGS.items():
        ours = {model_key(m["name"]) for m in yaml.safe_load(path.read_text())["models"]}
        for c in (c for c in claims.claims if c.benchmark == benchmark):
            for name in [c.model, c.versus, *c.models]:
                assert name is None or model_key(name) in ours, f"{c.id}: paper model {name!r} matches no model in {path.name}"
    named = [n for h in claims.headline for n in [*h.models, *(x for pair in h.pairs for x in pair)]]
    ours = {model_key(m["name"]) for m in yaml.safe_load(CONFIGS["text"].read_text())["models"]}
    assert all(model_key(n) in ours for n in named)


def test_three_headline_checks(claims):
    assert sorted(h.check for h in claims.headline) == ["actions_harder", "close_to_paper", "size_gap"]


def test_model_key_and_stand_in():
    assert model_key("GPT 5 Mini") == model_key("GPT-5 Mini") == model_key("GPT5-Mini")
    assert model_key("Claude Opus 4.1 (4.5)") == model_key("Claude Opus 4.1")
    assert is_stand_in("Claude Opus 4.1 (4.5)") and not is_stand_in("GPT-5")


def minimal(**claim) -> dict:
    base = {
        "id": "x",
        "figure": "4a",
        "section": "3",
        "benchmark": "text",
        "model": "GPT 5",
        "metric": "latent_risk_accuracy",
        "value": 0.9,
        "scale": "fraction",
        "kind": "value",
        "source": "figure",
        "quote": "q",
        "approx": True,
    }
    return {
        "paper": {"arxiv": "a", "version": "v2", "title": "t", "read_on": "2026-10-12"},
        "claims": [{**base, **claim}],
        "headline": [],
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"value": 1.5},  # fraction above 1
        {"value": 120.0, "scale": "percent"},
        {"kind": "mean"},  # a mean lists models, not one model
        {"kind": "value", "model": None},
        {"versus": "GPT 5 Nano"},  # only gaps have versus
        {"typo_field": 1},  # extra keys are rejected
    ],
)
def test_schema_rejects(bad):
    with pytest.raises(ValidationError):
        Claims.model_validate(minimal(**bad))


def test_duplicate_ids_rejected():
    data = minimal()
    data["claims"].append(dict(data["claims"][0]))
    with pytest.raises(ValidationError):
        Claims.model_validate(data)
