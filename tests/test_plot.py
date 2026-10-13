"""asimov.plot_data (what each figure plots) and scripts/plot.py (files written). No pixel checks."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import yaml

from asimov.claims import Claims
from asimov.collect.load import load_results
from asimov.plot_data import (
    FALLBACK_COLOR,
    Bar,
    Box,
    Point,
    Scene,
    baseline,
    color,
    in_paper_order,
    leader,
    needs_leader,
    ours,
    paper,
    penalty,
    place_labels,
    same_spot_ranks,
    segments_cross,
    severity_metric,
    text_vs_video,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FOR_TEST = ("latent_risk", "latent_severity", "effect", "activated_risk")
sys.path.insert(0, str(ROOT / "scripts"))
import plot  # noqa: E402  # pyright: ignore[reportMissingImports]
from results_tree import make_results, text_run  # noqa: E402  # pyright: ignore[reportMissingImports]

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


def test_paper_colors_follow_the_model():
    assert color("GPT-5 Nano") == color("GPT 5 Nano") == "#937860"
    assert color("Claude Opus 4.1 (4.5)") == "#c44e52"  # our stand-in gets the paper model's color
    assert color("Some Other Model") == FALLBACK_COLOR


def test_points_in_paper_legend_order():
    names = ["GPT-5", "Claude Opus 4.1 (4.5)", "Gemini 2.5 Flash-Lite (3.1)", "Unknown"]
    ordered = in_paper_order([Point(n, 0.5, 0.5) for n in names])
    assert [p.name for p in ordered] == ["Gemini 2.5 Flash-Lite (3.1)", "Claude Opus 4.1 (4.5)", "GPT-5", "Unknown"]


def test_plot_writes_each_runs_figures_under_its_run_id(results, tmp_path, capsys):
    text_run(results, "text_run_b", [[False, True, True, True]] * 4, scored_at="2026-10-13T10:00:00-0400")
    later = "2026-10-14T10:00:00-0400"
    text_run(results, "text_run_c", [[False, True, True, True]] * 4, scored_at=later, thinking="default")  # twin of 0 and 1
    (results / "run_ids.yaml").write_text(yaml.safe_dump({"text_run": "txt0", "video_run": "vid0"}))  # b, c: new
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(yaml.safe_dump(CLAIMS))
    headline = {"figures": {"fig4": {"run": "text_run"}, "fig5": {"run": "video_run"}, "fig6": {"run": None}}}
    (results / "headline.yaml").write_text(yaml.safe_dump(headline))
    out = tmp_path / "figures"
    args = ["--results", str(results), "--claims", str(claims_path), "--out", str(out)]
    assert plot.main(args) == 0
    expected = {
        "txt0": ("fig4a", "fig4b", "fig5a"),
        "txt1": ("fig4a", "fig4b", "fig5a"),
        "txt2": ("fig4a", "fig4b", "fig5a"),
        "vid0": ("fig5a", "fig5b"),
    }
    twins = {"txt0": ("txt2",), "txt1": ("txt2",), "txt2": ("txt0", "txt1"), "vid0": ()}
    for run_id, figs in expected.items():
        drawn = {p.name for p in (out / run_id).iterdir()}
        own = ("accuracy_heatmap", "severity_scoring", "accuracy_vs_cost") if run_id.startswith("txt") else ()
        own += tuple(f"thinking_vs_{t}" for t in twins[run_id])
        extra = {f"{f}.{ext}" for f in own for ext in ("png", "pdf")}
        assert drawn == {f"{f}_{v}.{ext}" for f in figs for v in ("paper", "comparison") for ext in ("png", "pdf")} | extra
    assert not list(out.glob("*.png")) and not list(out.glob("*.pdf"))  # nothing at the top level
    assert "fig6" in capsys.readouterr().out  # stub says it waits for the image scorer
    assert plot.main([*args, "--severity", "binary"]) == 0
    assert (out / "txt1" / "fig4a_comparison_severity-binary.png").exists()


def test_plot_skips_fig5a_without_a_headline_run_of_the_other_benchmark(results, tmp_path):
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(yaml.safe_dump(CLAIMS))
    (results / "headline.yaml").write_text(yaml.safe_dump({"figures": {"fig4": {"run": "text_run"}, "fig5": {"run": None}}}))
    out = tmp_path / "figures"
    assert plot.main(["--results", str(results), "--claims", str(claims_path), "--out", str(out)]) == 0
    assert (out / "txt0" / "fig4a_paper.png").exists() and not (out / "txt0" / "fig5a_paper.png").exists()
    assert (out / "vid0" / "fig5a_paper.png").exists()  # the text headline run is its partner


def test_same_spot_ranks_ring_models_that_share_a_score():
    points = [Point("A", 0.4, 1.0), Point("B", 0.6, 1.0), Point("C", 0.4, 1.0), Point("D", 0.4, 1.0)]
    assert same_spot_ranks(points) == [(0, 3), (0, 1), (1, 3), (2, 3)]


BOUNDS = Box(0, 0, 100, 100)


def test_labels_keep_home_when_free():
    options = [[Box(10, 10, 30, 20)], [Box(50, 50, 70, 60)]]
    assert place_labels(options, [], step=10, bounds=BOUNDS) == [(0, 0.0, 0.0), (0, 0.0, 0.0)]


def test_overlapping_label_moves_down_then_up_deterministically():
    same = [[Box(10, 50, 30, 60)]] * 3
    assert place_labels(same, [], step=10, bounds=BOUNDS) == [(0, 0.0, 0.0), (0, 0.0, -10), (0, 0.0, 10)]


def test_label_switches_side_before_moving():
    right, left = Box(10, 50, 30, 60), Box(-15, 50, 5, 60)
    bounds = Box(-20, 0, 100, 100)
    assert place_labels([[right], [right, left]], [], step=10, bounds=bounds) == [(0, 0.0, 0.0), (1, 0.0, 0.0)]


def test_labels_avoid_markers_and_stay_inside_the_plot():
    marker = Box(10, 45, 30, 65)
    top = [[Box(10, 92, 30, 102)]]  # starts above the plot: must come down
    assert place_labels(top, [], step=10, bounds=BOUNDS) == [(0, 0.0, -10)]
    assert place_labels([[Box(10, 50, 30, 60)]], [marker], step=10, bounds=BOUNDS) == [(0, 0.0, -20)]


def test_label_moves_by_small_steps_not_a_full_label_height():
    clipped = Box(0, 58, 40, 70)  # covers the label's top 2 units
    assert place_labels([[Box(10, 50, 30, 60)]], [clipped], step=2.5, bounds=BOUNDS) == [(0, 0.0, -2.5)]


def test_label_takes_the_nearest_free_spot_sideways_too():
    wall = Box(0, 40, 12, 100)  # overlaps the label's left 2 units; going down would take 20
    assert place_labels([[Box(10, 50, 30, 60)]], [wall], step=2.5, bounds=BOUNDS) == [(0, 2.5, 0.0)]


def test_label_avoids_sitting_between_its_marker_and_another():
    own, other = Box(40, 40, 50, 50), Box(62, 40, 72, 50)
    right, left = Box(52, 40, 60, 50), Box(30, 40, 38, 50)  # right is as near the other marker as its own
    options = [[right, left]]
    assert place_labels(options, [own, other], step=2.5, bounds=BOUNDS) == [(0, 0.0, 0.0)]  # markers unknown
    assert place_labels(options, [own, other], step=2.5, bounds=BOUNDS, markers=[own]) == [(0, 0.0, 0.0)]  # other: no label
    both = [[right, left], [Box(74, 40, 90, 50)]]  # the other marker's own label, to its right
    assert place_labels(both, [own, other], step=2.5, bounds=BOUNDS, markers=[own, other])[0] == (1, 0.0, 0.0)


def test_crossings_count_labels_and_lines_a_leader_line_runs_through():
    marker, label = Box(0, 0, 10, 10), Box(60, 0, 80, 10)
    line = leader(label, marker)
    assert line == ((5, 5), (70, 5))
    in_the_way = Scene([Box(30, 0, 40, 10)], [], [], BOUNDS)
    assert penalty(label, line, in_the_way, marker, []) == 1
    crossing_line = Scene([], [((35, -20), (35, 30))], [], BOUNDS)
    assert penalty(label, line, crossing_line, marker, []) == 1
    through_the_label = Scene([], [((70, -20), (70, 30))], [], BOUNDS)  # ends of own line on it don't count twice
    assert penalty(label, line, through_the_label, marker, []) == 1
    assert penalty(label, line, Scene([], [], [], BOUNDS), marker, []) == 0


def test_leader_only_when_the_label_is_away_from_its_marker():
    marker = Box(0, 0, 10, 10)
    assert not needs_leader(Box(12, 0, 32, 10), marker)  # 2 units off, label height 10
    assert needs_leader(Box(16, 0, 36, 10), marker)
    assert segments_cross(((0, 0), (10, 10)), ((0, 10), (10, 0)))
    assert not segments_cross(((0, 0), (10, 0)), ((0, 5), (10, 5)))
    assert Box(4, 4, 6, 6).crossed_by((0, 0), (10, 10)) and not Box(4, 4, 6, 6).crossed_by((0, 10), (2, 8))


def test_no_model_legend_without_points():
    fig, ax = plt.subplots()
    plot.model_legend(ax, [], plot.COMPARISON)
    assert ax.get_legend() is None
    plot.model_legend(ax, [Point("GPT-5", 0.5, 0.5)], plot.COMPARISON)
    legend = ax.get_legend()
    assert legend is not None and [t.get_text() for t in legend.get_texts()] == ["GPT-5"]
    plt.close(fig)


def test_legend_moves_outside_when_it_would_cover_a_marker():
    fig, ax = plt.subplots()
    corners = [
        Point(n, x, y) for n, x, y in [("GPT-5", 0, 0), ("GPT-5 Mini", 0, 1), ("GPT-5 Nano", 1, 0), ("Claude Sonnet 4", 1, 1)]
    ]
    corners += [Point(f"M{i}", i / 10, j / 10) for i in range(11) for j in range(11)]  # no free spot inside
    plot.scatter(ax, corners, plot.PAPER_FIG4)
    plot.model_legend(ax, corners, plot.PAPER_FIG4)
    legend = ax.get_legend()
    assert legend is not None and legend.get_bbox_to_anchor().x0 > ax.get_window_extent().x1  # outside, right
    assert not plot.legend_covers(ax, legend, [])
    plt.close(fig)


def test_fig5a_leaves_a_missing_value_out_instead_of_drawing_zero(tmp_path):
    bars = [Bar("GPT-5", 0.9, None, (0.85, 0.95), None, None, None)]
    paths = plot.fig5a(bars, "fig5a_test", tmp_path, comparison=True)
    assert all(p.exists() for p in paths)


def test_reading_notes_match_the_figure():
    fig4 = plot.scatter_notes("latent_severity_accuracy", "latent_risk_accuracy")
    assert "Higher = picks the right injury type." in fig4 and any("thin line" in n for n in fig4)
    fig5b = plot.scatter_notes("timing_mae_s", "within_0_5_rate", reversed_x=True, connected=False)
    assert "Further right = smaller timing error (axis reversed)." in fig5b
    assert not any("thin line" in n for n in fig5b)


def test_tick_names_wrap_without_running_into_each_other():
    assert plot.tick_name("Gemini 2.5 Flash-Lite (3.1)") == "Gemini 2.5\nFlash-Lite\n(3.1)"
    assert plot.tick_name("GPT-5") == "GPT-5"


def test_severity_scoring_chart_for_text_runs(results, tmp_path):
    from asimov.claims import Claims

    run = next(r for r in load_results(results) if r.benchmark == "text")
    paths = plot.severity_scoring(run, Claims.model_validate(CLAIMS), tmp_path)
    assert {p.name for p in paths} == {"severity_scoring.png", "severity_scoring.pdf"}
    assert [label for _, label, _ in plot.SCORING] == ["Multiple choice (exact level)", "Severe vs not", "Severe recall"]


def test_accuracy_heatmap_for_text_runs(results, tmp_path):
    from asimov.claims import Claims

    run = next(r for r in load_results(results) if r.benchmark == "text")
    paths = plot.accuracy_heatmap(run, Claims.model_validate(CLAIMS), tmp_path)
    assert {p.name for p in paths} == {"accuracy_heatmap.png", "accuracy_heatmap.pdf"}
    assert list(plot.HEATMAP_COLUMNS) == list(QUESTIONS_FOR_TEST)


def test_accuracy_vs_cost_leaves_out_models_without_tokens(results, tmp_path):
    from asimov.plot_data import cost_points

    run = next(r for r in load_results(results) if r.benchmark == "text")
    points = cost_points(run)
    assert [(p.name, p.tokens) for p in points] == [("GPT-5 Nano", 25.0), ("GPT-5", 25.0)]
    assert plot.cost_notes(points)[-1].startswith("Each panel")  # nothing left out: no extra note
    missing = [*points, plot.CostPoint("Claude Sonnet 4", None, (0.5, 0.5, 0.5, 0.5))]
    assert plot.cost_notes(missing)[-1] == "Not shown (no output token counts): Claude Sonnet 4."
    paths = plot.accuracy_vs_cost(run, tmp_path)
    assert {p.name for p in paths} == {"accuracy_vs_cost.png", "accuracy_vs_cost.pdf"}


def test_token_ticks_read_as_plain_numbers():
    assert [plot.token_tick(v) for v in (200, 500, 1000, 2000, 1500, 10000)] == ["200", "500", "1k", "2k", "1.5k", "10k"]
    assert plot.token_ticks(220, 534) == [300, 400, 500]
    assert plot.token_ticks(220, 2300) == [300, 400, 500, 700, 1000, 2000]
    assert plot.token_ticks(310, 390) == [310, 390]  # fewer than two steps fit: the ends, rounded inward


def test_thinking_chart_for_twins(results, tmp_path):
    text_run(results, "text_run_default", [[True, False, True, True]] * 4, thinking="default")
    runs = {r.name: r for r in load_results(results)}
    ids = {"text_run": "txt0", "text_run_default": "txt1"}
    paths = plot.thinking_chart(runs["text_run_default"], runs["text_run"], ids, tmp_path)
    assert {p.name for p in paths} == {"thinking_vs_txt0.png", "thinking_vs_txt0.pdf"}
    assert plot.thinking_legend_labels(runs["text_run_default"], runs["text_run"], ids) == ["min (txt0)", "default (txt1)"]


def test_small_multiple_panels_get_at_least_two_percent_ticks():
    assert plot.panel_percent_step((0.4, 0.95)) == 0.1
    assert plot.panel_percent_step((0.84, 0.942)) == 0.05  # 85%, 90%
    assert plot.panel_percent_step((0.854, 0.942)) == 0.02  # only 90% at 5-point steps
    assert plot.panel_percent_step((0.861, 0.899)) == 0.01  # only 88% at 2-point steps
