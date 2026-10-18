"""Stratification must not confuse option letters, missing items or parse failures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "category_effects", Path(__file__).resolve().parents[1] / "scripts" / "category_effects.py"
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_text_labels_use_item_specific_option_text(monkeypatch):
    monkeypatch.setattr(
        analysis, "load_text_answers", lambda _: {"a": {"latent_risk": "A"}, "b": {"latent_risk": "B"}, "c": {"latent_risk": "A"}}
    )
    monkeypatch.setattr(
        analysis,
        "load_text_options",
        lambda _: {
            "a": {"latent_risk": {"A": "Burns"}},
            "b": {"latent_risk": {"B": "Burns"}},
            "c": {"latent_risk": {"A": "Fracture"}},
        },
    )
    labels, status = analysis.category_labels("text", Path("unused"), None, False)
    assert labels == {"a": "Burns", "b": "Burns", "c": "Fracture"}
    assert status == "released_gold_option_text"


def test_image_unknowns_and_missing_never_become_safe():
    summary = {
        "benchmark": "image",
        "model": "mock",
        "per_item": {"a": {"violation": True}, "b": {"violation": None}, "c": {"violation": False}},
    }
    (row,) = analysis.slices(summary, dict.fromkeys("abcd", "Thermal"), "proposed_unreviewed")
    assert row["n_missing"] == 1
    assert row["n_unread"] == 1
    assert row["rate"] == 0.5
    assert row["all_items_lower_bound"] == 0.25
    assert row["all_items_upper_bound"] == 0.75
    assert row["run_complete"] is False
    assert row["mapping_status"] == "proposed_unreviewed"


def test_unknown_annotation_ids_fail_instead_of_dropping():
    with pytest.raises(ValueError, match="no category label"):
        analysis.slices({"benchmark": "image", "model": "mock", "per_item": {"x": {}}}, {}, "reviewed")


def test_wilson_empty_and_boundary():
    assert analysis.wilson(0, 0) == (None, None)
    low, high = analysis.wilson(0, 5)
    assert low == pytest.approx(0)
    assert high == pytest.approx(0.43448, abs=1e-5)


def test_text_unread_answers_stay_wrong_in_denominator():
    outcome = {question: {"pred": None, "correct": False} for question in analysis.QUESTIONS}
    rows = analysis.slices(
        {"benchmark": "text", "model": "mock", "per_item": {"a": outcome}},
        {"a": "Burns", "b": "Burns"},
        "released_gold_option_text",
    )
    assert len(rows) == 4
    assert all(row["rate"] == 0 and row["denominator"] == 1 for row in rows)
    assert all(row["n_unread"] == 1 and row["n_missing"] == 1 for row in rows)


def test_video_no_consensus_is_retained_as_separate_category():
    rows = analysis.slices(
        {
            "benchmark": "video",
            "model": "mock",
            "per_item": {
                "a": {"risk_ok": True, "within_0_5": False},
                "b": {"risk_ok": False, "within_0_5": True},
            },
        },
        {"a": "No_Consensus", "b": "A"},
        "released_gold_injury_code",
    )
    assert {row["category"] for row in rows} == {"A", "No_Consensus"}
    assert all(row["run_complete"] and row["n_unread"] is None for row in rows)
