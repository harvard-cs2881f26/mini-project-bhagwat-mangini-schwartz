"""configs/prices.yaml and asimov.prices: the table loads, covers every model we price, and bills tokens the providers' way."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from asimov.config import RunConfig
from asimov.prices import PRICES_PATH, Price, PriceTable, call_cost, load_prices

ROOT = Path(__file__).resolve().parents[1]
PRICED_CONFIGS = ("repro_fig4", "part2_jev", "part2_codiv", "part2_kev")


def test_real_table_loads_with_a_date():
    table = load_prices(PRICES_PATH)
    assert table.checked.isoformat() == "2026-10-12"


@pytest.mark.parametrize("config", PRICED_CONFIGS)
def test_every_model_in_the_analysed_configs_has_an_entry(config):
    table = load_prices(PRICES_PATH)
    cfg = RunConfig.model_validate(yaml.safe_load((ROOT / "configs" / f"{config}.yaml").read_text()))
    models = [m.model for m in cfg.models]
    assert [m for m in models if m not in table.models] == []


def test_gemini_thinking_tokens_are_billed_as_output():
    price = Price(input_per_mtok=2.0, output_per_mtok=12.0, thinking_as_output=True)
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "thinking_tokens": 1_000_000}
    assert call_cost(usage, price) == pytest.approx(2.0 + 2 * 12.0)


def test_thinking_tokens_are_not_added_when_output_already_counts_them():
    """OpenAI's output_tokens already include reasoning tokens."""
    price = Price(input_per_mtok=1.25, output_per_mtok=10.0)
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "thinking_tokens": 1_000_000}
    assert call_cost(usage, price) == pytest.approx(1.25 + 10.0)


def test_cached_input_has_its_own_rate():
    price = Price(input_per_mtok=1.0, output_per_mtok=0.0, cached_input_per_mtok=0.1)
    usage = {"input_tokens": 1_000_000, "output_tokens": 0, "cached_input_tokens": 400_000}
    assert call_cost(usage, price) == pytest.approx(0.6 + 0.04)


def test_free_tier_costs_nothing_and_self_hosted_has_no_dollar_price():
    usage = {"input_tokens": 500, "output_tokens": 0}
    assert call_cost(usage, Price(free_tier=True)) == 0.0
    assert call_cost(usage, Price(self_hosted="L40S", params_b=4)) is None


def test_the_real_table_reproduces_the_logged_gemini_pro_cost():
    """WORKLOG 'Cost method': tokens x list price, Gemini thinking counted as output."""
    price = load_prices(PRICES_PATH).models["gemini-3.1-pro-preview"]
    usage = {"input_tokens": 1435, "output_tokens": 260, "thinking_tokens": 394}
    assert call_cost(usage, price) == pytest.approx((1435 * 2 + (260 + 394) * 12) / 1e6)


@pytest.mark.parametrize(
    "entry",
    [
        {"input_per_mtok": 1.0, "output_per_mtok": 1.0, "typo": 1},  # unknown key
        {"input_per_mtok": 1.0},  # token-priced but no output price
        {"self_hosted": "L40S", "input_per_mtok": 1.0},  # self-hosted with a token price
        {"free_tier": True, "output_per_mtok": 1.0},  # free with a token price
    ],
)
def test_bad_entries_fail_on_load(entry):
    with pytest.raises(ValidationError):
        PriceTable.model_validate({"checked": "2026-10-12", "sources": {}, "models": {"m": entry}})


def test_load_prices_reads_yaml(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text(yaml.safe_dump({"checked": "2026-10-12", "sources": {"x": "y"}, "models": {"m": {"free_tier": True}}}))
    assert load_prices(path).models["m"].free_tier
