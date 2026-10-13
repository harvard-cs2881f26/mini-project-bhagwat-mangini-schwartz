"""Category aggregation must expose missing responses and denominator choices."""
import csv
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("plot_constraints", Path(__file__).parents[1] / "scripts/plot_constraints.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def write_mapping(tmp_path, rows):
    path = tmp_path / "categories.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "review_status"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_proposals_require_explicit_opt_in(tmp_path):
    category = next(iter(plot.PAPER_CATEGORY_COUNTS))
    path = write_mapping(tmp_path, [{"id": "a", "category": category, "review_status": "proposed"}])
    with pytest.raises(ValueError, match="Unreviewed"):
        plot.load_categories(path, {"a"})
    mapping, status = plot.load_categories(path, {"a"}, allow_proposed=True)
    assert status == "proposed_unreviewed"
    assert mapping["a"]["category"] == category


@pytest.mark.parametrize("case", ["duplicate", "missing", "unknown"])
def test_invalid_mapping_rejected(tmp_path, case):
    row = {"id": "a", "category": next(iter(plot.PAPER_CATEGORY_COUNTS)), "review_status": "reviewed"}
    rows = [row, row] if case == "duplicate" else [row]
    if case == "unknown":
        row["category"] = "invented"
    path = write_mapping(tmp_path, rows)
    with pytest.raises(ValueError):
        plot.load_categories(path, {"a", "b"} if case == "missing" else {"a"})


def test_category_denominators_and_missing_responses():
    gripper, material = list(plot.PAPER_CATEGORY_COUNTS)[:2]
    mapping = {i: {"category": c} for i, c in [("a", gripper), ("b", gripper),
                                               ("c", gripper), ("d", material), ("e", material)]}
    summary = {"model": "test", "n_violations": 2, "per_item": {
        "a": {"violation": True}, "b": {"violation": False},
        "c": {"violation": None}, "d": {"violation": True}}}
    rows = plot.category_rates(summary, mapping, "proposed_unreviewed")
    g, m = rows[:2]
    assert g["within_category_rate"] == .5 and g["share_of_violations"] == .5
    assert g["n_parse_failures"] == 1 and g["n_missing"] == 0
    assert m["within_category_rate"] == 1 and m["share_of_violations"] == .5
    assert m["n_dataset_category"] == 2 and m["n_missing"] == 1
    assert sum(r["share_of_violations"] for r in rows) == 1
    assert rows[-1]["within_category_rate"] is None
