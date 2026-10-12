"""asimov.plot_data (what each figure plots) and scripts/plot.py (files written). No pixel checks."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import yaml

from asimov.claims import Claims
from asimov.collect.load import load_results
from asimov.plot_data import (
    Box,
    Point,
    baseline,
    colors,
    ours,
    paper,
    place_labels,
    same_spot_ranks,
    severity_metric,
    text_vs_video,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot  # noqa: E402  # pyright: ignore[reportMissingImports]
from results_tree import make_results  # noqa: E402  # pyright: ignore[reportMissingImports]

PAPER = {"arxiv": "2509.21651", "version": "v2", "title": "T", "read_on": "2026-10-12"}


def claim(id_: str, figure: str | None, benchmark: str, model: str | None, metric: str, value: float, **extra) -> dict:
    return {
        "id": id_,
        "figure": figure,
        "section": "3",
        "benchmark": benchmark,
        "model": model,
        "metric": metric,
        "value": value,
        "scale": "fraction",
        "kind": "value",
        "source": "figure",
        "quote": "q",
        "approx": True,
        **extra,
    }


CLAIMS = {
    "paper": PAPER,
    "claims": [
        claim("a1", "4a", "text", "GPT 5", "latent_risk_accuracy", 0.92),
        claim("a2", "4a", "text", "GPT 5", "latent_severity_accuracy", 0.82),
        claim("a3", "4a", "text", "GPT 5 Nano", "latent_risk_accuracy", 0.87),  # no severity value: no paper point
        claim("t5", "5a", "text", "GPT 5", "latent_risk_accuracy", 0.92),
        claim("v5", "5a", "video", "GPT 5", "risk_accuracy", 0.44),
        claim("b1", "5b", "video", "GPT 5", "timing_mae_s", 1.76, scale="seconds"),
        claim("b2", "5b", "video", "GPT 5", "within_0_5_rate", 8.6, scale="percent"),
        claim("base", None, "text", None, "majority_baseline_latent_severity", 0.445, kind="baseline"),
    ],
    "headline": [],
}


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return make_results(tmp_path)


@pytest.fixture
def runs(results):
    return {r.name: r for r in load_results(results)}


@pytest.fixture
def claims() -> Claims:
    return Claims.model_validate(CLAIMS)


def test_severity_switch():
    assert severity_metric("latent_severity", "4way") == "latent_severity_accuracy"
    assert severity_metric("activated_risk", "binary") == "activated_risk_severe_binary_accuracy"
    assert severity_metric("latent_severity", "recall") == "latent_severity_severe_recall"


def test_our_points_have_wilson_cis(runs):
    points = ours(runs["text_run"], "latent_severity_accuracy", "latent_risk_accuracy")
    assert [p.name for p in points] == ["GPT-5", "GPT-5 Nano"]
    gpt5 = points[0]
    assert (gpt5.x, gpt5.y) == (1.0, 1.0)
    assert gpt5.y_ci is not None and gpt5.y_ci[0] < 1.0 == gpt5.y_ci[1]


def test_paper_points_need_both_values(runs, claims):
    points = paper(runs["text_run"], claims, "4a", "latent_severity_accuracy", "latent_risk_accuracy")
    assert [(p.name, p.x, p.y) for p in points] == [("GPT-5", 0.82, 0.92)]


def test_fig5b_scales_percent(runs, claims):
    points = paper(runs["video_run"], claims, "5b", "timing_mae_s", "within_0_5_rate", y_scale=100)
    assert points[0].x == pytest.approx(1.76) and points[0].y == pytest.approx(8.6)
    mine = ours(runs["video_run"], "timing_mae_s", "within_0_5_rate", y_scale=100)
    assert mine[0].y == pytest.approx(100 / 3)


def test_baseline(claims):
    assert baseline(claims, "latent_severity_accuracy") == pytest.approx(0.445)
    assert baseline(claims, "latent_risk_accuracy") is None


def test_text_vs_video(runs, claims):
    bars = text_vs_video(runs["text_run"], runs["video_run"], claims, only_paper_models=True)
    assert [(b.name, b.text, b.paper_text, b.paper_video) for b in bars] == [("GPT-5", 1.0, 0.92, 0.44)]
    assert bars[0].video == pytest.approx(2 / 3)


def test_colors_follow_the_model_not_the_figure():
    c = colors(["GPT-5 Nano", "GPT-5"])
    assert c["gpt5nano"] != c["gpt5"]
    assert colors(["GPT-5 Nano", "GPT-5", "Claude Opus 4.1 (4.5)"])["gpt5"] == c["gpt5"]


def test_plot_writes_paper_and_comparison_figures(results, tmp_path, capsys):
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(yaml.safe_dump(CLAIMS))
    headline = {"figures": {"fig4": {"run": "text_run"}, "fig5": {"run": "video_run"}, "fig6": {"run": None}}}
    (results / "headline.yaml").write_text(yaml.safe_dump(headline))
    out = tmp_path / "figures"
    args = [
        "--results",
        str(results),
        "--claims",
        str(claims_path),
        "--out",
        str(out),
        "--config",
        str(ROOT / "configs/repro_fig4.yaml"),
    ]
    assert plot.main(args) == 0
    for fig in ("fig4a", "fig4b", "fig5a", "fig5b"):
        for version in ("paper", "comparison"):
            assert (out / f"{fig}_{version}.png").exists() and (out / f"{fig}_{version}.pdf").exists()
    assert "fig6" in capsys.readouterr().out  # stub says it waits for the image scorer
    assert plot.main([*args, "--severity", "binary"]) == 0
    assert (out / "fig4a_comparison_severity-binary.png").exists()


def test_same_spot_ranks_ring_models_that_share_a_score():
    points = [Point("A", 0.4, 1.0), Point("B", 0.6, 1.0), Point("C", 0.4, 1.0), Point("D", 0.4, 1.0)]
    assert same_spot_ranks(points) == [(0, 3), (0, 1), (1, 3), (2, 3)]


BOUNDS = Box(0, 0, 100, 100)


def test_labels_keep_home_when_free():
    options = [[Box(10, 10, 30, 20)], [Box(50, 50, 70, 60)]]
    assert place_labels(options, [], step=10, bounds=BOUNDS) == [(0, 0.0), (0, 0.0)]


def test_overlapping_label_moves_down_then_up_deterministically():
    same = [[Box(10, 50, 30, 60)]] * 3
    assert place_labels(same, [], step=10, bounds=BOUNDS) == [(0, 0.0), (0, -10), (0, 10)]


def test_label_switches_side_before_moving():
    right, left = Box(10, 50, 30, 60), Box(-15, 50, 5, 60)
    bounds = Box(-20, 0, 100, 100)
    assert place_labels([[right], [right, left]], [], step=10, bounds=bounds) == [(0, 0.0), (1, 0.0)]


def test_labels_avoid_markers_and_stay_inside_the_plot():
    marker = Box(10, 45, 30, 65)
    top = [[Box(10, 92, 30, 102)]]  # starts above the plot: must come down
    assert place_labels(top, [], step=10, bounds=BOUNDS) == [(0, -10)]
    assert place_labels([[Box(10, 50, 30, 60)]], [marker], step=10, bounds=BOUNDS) == [(0, -20)]


def test_no_model_legend_without_points():
    fig, ax = plt.subplots()
    plot.model_legend(ax, [], {})
    assert ax.get_legend() is None
    plot.model_legend(ax, [Point("GPT-5", 0.5, 0.5)], colors(["GPT-5"]))
    legend = ax.get_legend()
    assert legend is not None and [t.get_text() for t in legend.get_texts()] == ["GPT-5"]
    plt.close(fig)
