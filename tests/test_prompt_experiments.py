"""Part 2 prompt experiments (EXPLORATORY): an LLM given Jev's picks against the same LLM alone, pure logic."""

import pytest
from pydantic import ValidationError

from asimov.cascade import MEAN, ItemResult, System
from asimov.prompt_experiments import Comparison, ExperimentConfig, compare, system_items

T, F = True, False


def sys_(name: str, rows: dict[str, tuple[tuple[bool, ...], float, float]]) -> System:
    return System(name, {i: ItemResult(c, cost, lat) for i, (c, cost, lat) in rows.items()}, {})


FIRST = sys_("Jev", {"a": ((T, F, F, F), 0.001, 0.2), "b": ((F, T, T, T), 0.001, 0.3)})
LLM = sys_("GPT", {"a": ((F, T, T, T), 0.01, 2.0), "b": ((T, T, F, F), 0.01, 3.0)})
VARIANT = sys_("GPT+Jev", {"a": ((F, F, T, T), 0.02, 2.5), "b": ((F, T, T, F), 0.02, 3.5)})


def test_q2_q4_takes_q1_from_the_first_model_and_adds_both_calls():
    items = system_items("q2_q4", FIRST, VARIANT)
    assert items["a"].correct == (T, F, T, T) and items["b"].correct == (F, T, T, F)
    assert items["a"].cost == pytest.approx(0.021) and items["a"].latency == pytest.approx(2.7)


def test_hints_take_every_answer_from_the_variant_but_pay_for_the_first_call_too():
    items = system_items("hints", FIRST, VARIANT)
    assert items["a"].correct == VARIANT.items["a"].correct
    assert items["b"].cost == pytest.approx(0.021) and items["b"].latency == pytest.approx(3.8)


def test_compare_counts_discordant_pairs_and_reports_cost_and_latency():
    r = compare("q2_q4", FIRST, LLM, VARIANT)
    assert r.n == 2
    q1 = next(m for m in r.rows if m.metric == "latent_risk")
    assert (q1.baseline.value, q1.system.value) == (0.5, 0.5)
    assert (q1.baseline_only, q1.system_only) == (1, 1) and q1.mcnemar_p == 1.0
    mean = next(m for m in r.rows if m.metric == MEAN)
    assert mean.mcnemar_p is None and mean.system.value == pytest.approx((3 / 4 + 2 / 4) / 2)
    assert r.baseline_cost_per_1000 == pytest.approx(10.0) and r.system_cost_per_1000 == pytest.approx(21.0)
    assert r.baseline_p50 == pytest.approx(2.5) and r.system_p50 == pytest.approx(3.25)


def test_compare_needs_the_same_items_everywhere():
    short = sys_("GPT+Jev", {"a": ((F, F, T, T), 0.02, 2.5)})
    with pytest.raises(ValueError, match="same items"):
        compare("hints", FIRST, LLM, short)


def test_config_rejects_unknown_keys_and_kinds():
    ok = {"kind": "hints", "baseline": {"run": "r", "model": "m"}, "variant": {"run": "v", "model": "m"}}
    ExperimentConfig.model_validate({"label": "EXPLORATORY", "first": {"run": "j", "model": "jev"}, "comparisons": [ok]})
    with pytest.raises(ValidationError):
        Comparison.model_validate({**ok, "extra": 1})
    with pytest.raises(ValidationError):
        Comparison.model_validate({**ok, "kind": "other"})
