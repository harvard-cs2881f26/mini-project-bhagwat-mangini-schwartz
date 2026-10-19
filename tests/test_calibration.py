"""Calibration (Brier, ECE) of decision-model probabilities in asimov.collect, from saved per_item and probs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from asimov.bench.text_score import QUESTIONS
from asimov.collect.calibration import calibration, calibration_columns, calibrations
from asimov.collect.load import ModelResult, load_model
from asimov.collect.pages import calibration_section
from asimov.collect.tables import index_row
from asimov.stats import brier, ece

Q1, Q2 = QUESTIONS[0], QUESTIONS[1]


def outcome(pred: str | None, correct: bool) -> dict:
    return {"pred": pred, "gold": "A", "correct": correct}


def model(per_item: dict, probs: dict, name: str = "Jev") -> ModelResult:
    summary = {"name": name, "model": "jev", "n": len(per_item), "benchmark": "text"}
    return ModelResult(summary, per_item, (), (), {}, probs=probs)


def test_confidence_is_the_probability_on_the_answer_given():
    per_item = {
        "t1": {Q1: outcome("A", True)},
        "t2": {Q1: outcome("B", False)},
        "t3": {Q1: outcome("A", False)},
    }
    probs = {
        "t1": {Q1: {"A": 0.9, "B": 0.1}},
        "t2": {Q1: {"A": 0.5, "B": 0.5}},  # a tie went to the service's own answer, B
        "t3": {Q1: {"A": 0.3, "B": 0.7}},  # answered A though B had the larger probability
    }
    c = calibration(model(per_item, probs), Q1)
    assert (c.n, c.n_total) == (3, 3)
    assert c.confidence == pytest.approx((0.9 + 0.5 + 0.3) / 3)
    assert c.accuracy == pytest.approx(1 / 3)
    assert c.brier == pytest.approx(brier([0.9, 0.5, 0.3], [1, 0, 0]))
    assert c.ece == pytest.approx(ece([0.9, 0.5, 0.3], [1, 0, 0]))
    assert c.confidence is not None and c.accuracy is not None
    assert c.gap == pytest.approx(c.confidence - c.accuracy)


def test_answers_without_a_probability_are_left_out_not_scored_as_zero():
    per_item = {
        "t1": {Q1: outcome("A", True)},
        "t2": {Q1: outcome(None, False)},  # refused: no answer, no probabilities
        "t3": {Q1: outcome("A", True)},  # answered, but the service sent no probabilities for Q1
    }
    probs = {"t1": {Q1: {"A": 0.8, "B": 0.2}}, "t2": {Q1: None}, "t3": {}}
    c = calibration(model(per_item, probs), Q1)
    assert (c.n, c.n_total) == (1, 3)
    assert c.confidence == pytest.approx(0.8)
    assert c.brier == pytest.approx(0.04)


def test_a_question_with_no_probabilities_at_all_has_empty_metrics():
    c = calibration(model({"t1": {Q1: outcome("A", True)}}, {"t1": {Q2: {"A": 1.0}}}), Q1)
    assert (c.n, c.brier, c.ece, c.gap) == (0, None, None, None)


def test_llm_rows_have_no_calibration():
    m = model({"t1": {q: outcome("A", True) for q in QUESTIONS}}, {})
    assert calibrations(m) == []
    assert calibration_columns(m) == {}


def test_columns_cover_every_question():
    per_item = {"t1": {q: outcome("A", True) for q in QUESTIONS}}
    probs = {"t1": {q: {"A": 0.6, "B": 0.4} for q in QUESTIONS}}
    cols = calibration_columns(model(per_item, probs))
    assert set(cols) == {f"{q}_{k}" for q in QUESTIONS for k in ("brier", "ece", "n_calibrated")}
    assert cols[f"{Q1}_brier"] == pytest.approx(0.16)
    assert cols[f"{Q1}_n_calibrated"] == 1


def write_decision_model(mdir: Path, probs_by_item: dict) -> None:
    """A text summary.json + responses.jsonl for one decision model, with probs on the saved rows."""
    mdir.mkdir(parents=True)
    key = "k"
    per_item = {i: {q: outcome("A", True) for q in QUESTIONS} for i in probs_by_item}
    summary = {"name": "Jev", "model": "jev", "provider": "systemone", "benchmark": "text", "run": "r", "key": key, "n": 2}
    (mdir / "summary.json").write_text(json.dumps({**summary, "per_item": per_item}))
    rows = [{"id": i, "key": key, "latency_s": 0.2, "text": "{}", "probs": p} for i, p in probs_by_item.items()]
    (mdir / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_index_row_and_run_page_show_calibration(tmp_path: Path):
    from asimov.collect.load import RunData
    from asimov.collect.run_ids import RunLabel

    run_dir = tmp_path / "txt9_r"
    (run_dir).mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"run": "r", "benchmark": "text", "prompt": "p", "thinking": "min"}))
    probs = {"t1": {q: {"A": 0.9, "B": 0.1} for q in QUESTIONS}, "t2": {q: {"A": 0.7, "B": 0.3} for q in QUESTIONS}}
    write_decision_model(run_dir / "jev", probs)
    m = load_model(run_dir / "jev")
    assert len(m.probs) == 2
    run = RunData("r", "txt9_r", "text", yaml.safe_load((run_dir / "config.yaml").read_text()), (m,))
    labels = {"r": RunLabel("txt9", "d")}
    row = index_row(run, m, {}, labels)
    assert row[f"{Q1}_n_calibrated"] == 2
    assert row[f"{Q1}_brier"] == pytest.approx(((0.1) ** 2 + (0.3) ** 2) / 2)
    section = calibration_section(run)
    assert section is not None and "## Calibration of the option probabilities" in section
    assert "| Jev | Q1 latent risk type | 2 / 2 |" in section
    assert "| 0.800 | 1.000 |" in section and "| -0.200 |" in section  # mean confidence, accuracy, signed gap


def test_run_page_without_probabilities_has_no_calibration_section():
    from asimov.collect.load import RunData

    m = model({"t1": {q: outcome("A", True) for q in QUESTIONS}}, {}, name="GPT-5")
    run = RunData("r", "txt9_r", "text", None, (m,))
    assert calibration_section(run) is None
