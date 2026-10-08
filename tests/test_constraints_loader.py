import pytest

from asimov.data.constraints import DATA_DIR, Box, load_constraints, parse_box_string, to_yx1000

# Real string from constraints_0000 (wooden block covered in sawdust).
RAW = "[x: 0.59242144\ny: 0.22728116\n, x: 0.96487985\ny: 0.93795882\n]"


def test_parse_box_string_reads_x_then_y():
    assert parse_box_string(RAW) == (0.59242144, 0.22728116, 0.96487985, 0.93795882)


def test_to_yx1000_swaps_axes_and_scales():
    assert to_yx1000(0.1, 0.2, 0.3, 0.4) == pytest.approx((200, 100, 400, 300))


def test_box_contains_is_closed_on_edges():
    b = Box("x", 100, 100, 200, 200, True)
    assert b.contains(100, 150) and b.contains(200, 200)
    assert not b.contains(200.001, 150)


@pytest.mark.skipif(not (DATA_DIR / "constraints" / "items.jsonl").exists(), reason="data not downloaded")
def test_real_data_loads():
    for variant in ("constraints", "constraints_rationale"):
        items = load_constraints(variant)
        assert len(items) == 164
        assert all(i.violating and i.safe for i in items)
        assert all(i.image_path.exists() for i in items)
