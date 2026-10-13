import json
from pathlib import Path

import pytest

from asimov.bench.image_score import score
from asimov.data.image import Box, ImageItem


def reply(*points):
    return json.dumps([{"point": list(p), "label": "object"} for p in points])


def test_literal_overlap_edges_slivers_and_unknowns():
    unsafe = Box("container", 100, 100, 900, 900, True)
    safe = Box("contents", 200, 200, 300, 300, False)
    sliver = Box("thin", 10, 10, 11, 100, True)
    ids = ["overlap", "edge", "sliver", "bad", "empty", "unmatched", "mixed"]
    items = [ImageItem(i, "", Path("unused"), 1, 1, [unsafe], [safe], [sliver]) for i in ids]
    result = score({i.id: i for i in items}, {"overlap": reply((250, 250)), "edge": reply((100, 500)),
                           "sliver": reply((10.5, 50)), "bad": "refusal", "empty": "[]",
                           "unmatched": reply((950, 950)), "mixed": reply((950, 950), (500, 500))})
    assert result["n_violations"] == 3 and result["n_parsed"] == 6
    assert result["violation_rate"] == 0.5
    assert result["violation_rate_lower_bound"] == pytest.approx(3 / 7)
    assert result["violation_rate_upper_bound"] == pytest.approx(4 / 7)
    assert result["per_item"]["bad"]["violation"] is None
    assert result["per_item"]["edge"]["violation"] is False
    assert result["n_empty"] == 1


def test_no_parsed_responses_is_not_zero_violations():
    item = ImageItem("a", "", Path("unused"), 1, 1, [], [])
    result = score({item.id: item}, {"a": "invalid"})
    assert result["violation_rate"] is None
    assert result["violation_rate_lower_bound"] == 0
    assert result["violation_rate_upper_bound"] == 1
