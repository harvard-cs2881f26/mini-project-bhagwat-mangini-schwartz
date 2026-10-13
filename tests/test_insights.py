"""asimov.collect.insights: the one-sentence "Key insight" under each chart, on synthetic chart data."""

from asimov.collect.insights import (
    cost_insight,
    heatmap_insight,
    paper_range_insight,
    severity_insight,
    speed_insight,
    text_vs_video_insight,
    thinking_insight,
)
from asimov.collect.metrics import Estimate
from asimov.collect.tables import SpeedRow
from asimov.plot_data import Bar, CostPoint, GridRow, RangeCount, ScoringRow, TwinRow

NONE4 = (None, None, None, None)


def bar(name: str, text: float | None, video: float | None) -> Bar:
    return Bar(name, text, video, None, None, None, None)


def est(v: float) -> Estimate:
    return Estimate(v, None, None, 10)


def scoring(name: str, mc: float | None, recall: float | None) -> ScoringRow:
    return ScoringRow(name, (None if mc is None else est(mc), None, None if recall is None else est(recall)), None)


# ---- comparison versions of fig4a, fig4b, fig5b ---------------------------------------------------


def test_paper_range_counts_each_axis():
    counts = [RangeCount("latent_risk_accuracy", 6, 8), RangeCount("latent_severity_accuracy", 1, 8)]
    assert paper_range_insight(counts) == (
        "On injury type, the paper's value is inside our 95% range for 6 of 8 models; on severity, for 1 of 8 models."
    )


def test_paper_range_with_one_axis_or_none_reported():
    counts = [RangeCount("within_0_5_rate", 1, 1), RangeCount("timing_mae_s", 0, 0)]
    assert paper_range_insight(counts) == (
        "On stepping in within 0.5 s, the paper's value is inside our 95% range for 1 of 1 model; "
        "the paper reports no timing error values for these models."
    )
    flipped = [RangeCount("effect_accuracy", 0, 0), RangeCount("activated_risk_accuracy", 2, 3)]
    assert paper_range_insight(flipped) == (
        "The paper reports no action effect values for these models; "
        "on severity after action, the paper's value is inside our 95% range for 2 of 3 models."
    )
    none = [RangeCount("latent_risk_accuracy", 0, 0), RangeCount("latent_severity_accuracy", 0, 0)]
    assert paper_range_insight(none) == "The paper reports no values for these models on this chart."


# ---- fig5a ------------------------------------------------------------------------------------------


def test_text_vs_video_every_model_lower():
    bars = [bar("A", 0.9, 0.8), bar("B", 0.9, 0.46)]
    assert text_vs_video_insight(bars) == "Every model scores lower on video than on text, by 10 to 44 points."


def test_text_vs_video_mixed_single_and_missing():
    mixed = [bar("A", 0.9, 0.8), bar("B", 0.5, 0.6), bar("C", 0.7, 0.7), bar("D", 0.7, None)]
    assert text_vs_video_insight(mixed) == (
        "1 of 3 models scores lower on video than on text, by 10 points; 1 scores higher (by 10 points); 1 scores the same."
    )
    two_lower = [bar("A", 0.9, 0.8), bar("B", 0.9, 0.6), bar("C", 0.5, 0.6)]
    assert text_vs_video_insight(two_lower) == (
        "2 of 3 models score lower on video than on text, by up to 30 points; 1 scores higher (by 10 points)."
    )
    assert text_vs_video_insight([bar("A", 0.9, 0.8)]) == "A scores lower on video than on text, by 10 points."
    assert text_vs_video_insight([bar("A", 0.7, 0.7)]) == "A scores the same on video as on text."
    assert text_vs_video_insight([bar("A", 0.9, None)]) == "No model has both a text and a video score."


# ---- accuracy heatmap --------------------------------------------------------------------------------


def test_heatmap_lowest_question_and_largest_gap():
    rows = [
        GridRow("A", (0.9, 0.7, 0.8, 0.6), (0.92, 0.96, None, None)),
        GridRow("B", (0.9, 0.5, 0.8, 0.5), NONE4),  # ties: counts for Q2 and Q4
        GridRow("C", (0.9, 0.8, 0.7, 0.75), (0.9, None, None, None)),
    ]
    assert heatmap_insight(rows) == (
        "Q4 (severity after action) is the lowest-scoring question for 2 of 3 models; "
        "the largest gap to the paper is A on severity (70% vs 96%)."
    )


def test_heatmap_ties_and_no_paper_values():
    rows = [
        GridRow("A", (0.5, 0.6, 0.9, 0.9), (0.6, None, None, None)),
        GridRow("B", (0.9, 0.6, 0.5, 0.9), (None, 0.7, None, None)),
    ]
    assert heatmap_insight(rows) == (
        "Q1 (injury type) and Q3 (action effect) are each the lowest-scoring question for 1 of 2 models; "
        "the largest gaps to the paper (10 points each) are A on injury type (50% vs 60%) and B on severity (60% vs 70%)."
    )
    assert heatmap_insight([GridRow("A", (0.5, 0.6, 0.7, 0.8), NONE4)]) == (
        "Q1 (injury type) is the lowest-scoring question for 1 of 1 model; the paper reports no values for these models."
    )
    assert heatmap_insight([GridRow("A", NONE4, NONE4)]) == "No model has accuracy values."


# ---- severity scoring --------------------------------------------------------------------------------


def test_severity_recall_vs_multiple_choice():
    rows = [scoring("A", 0.7, 0.99), scoring("B", 0.6, 0.7), scoring("C", 0.8, 0.77)]
    assert severity_insight(rows) == (
        "Severe recall is higher than multiple-choice accuracy for 2 of 3 models, by up to 29 points; "
        "it is lower for 1 (by 3 points)."
    )
    assert severity_insight([scoring("A", 0.7, 0.99), scoring("B", 0.6, 0.7)]) == (
        "Severe recall is higher than multiple-choice accuracy for every model, by 10 to 29 points."
    )
    assert severity_insight([scoring("A", 0.7, 0.7)]) == "Severe recall equals multiple-choice accuracy for A."
    assert severity_insight([scoring("A", 0.7, None)]) == "No model has both a multiple-choice and a severe-recall score."


# ---- accuracy vs cost --------------------------------------------------------------------------------


def test_cost_fewest_tokens_and_gap_to_the_best():
    points = [
        CostPoint("A", 280.4, (0.9, 0.7, 0.6, 0.6)),
        CostPoint("B", 1200.0, (0.95, 0.75, 0.65, 0.62)),
        CostPoint("C", None, (1.0, 1.0, 1.0, 1.0)),  # no tokens: not on the chart, not compared
    ]
    assert cost_insight(points) == (
        "A uses the fewest output tokens per item (280, vs 1,200 for B, the most) "
        "and is within 5 points of the most accurate model on every question."
    )
    best = [CostPoint("A", 280.0, (0.95, 0.8, 0.7, 0.7)), CostPoint("B", 1200.0, (0.9, 0.75, 0.65, 0.62))]
    assert cost_insight(best) == (
        "A uses the fewest output tokens per item (280, vs 1,200 for B, the most) "
        "and is the most accurate model on every question."
    )


def test_cost_ties_one_model_and_no_tokens():
    tied = [
        CostPoint("A", 300.0, (0.9, 0.7, 0.6, 0.6)),
        CostPoint("B", 300.2, (0.8, 0.7, 0.6, 0.6)),
        CostPoint("C", 900.0, (0.9, 0.7, 0.6, 0.6)),
    ]
    assert cost_insight(tied) == (
        "A and B use the fewest output tokens per item (300, vs 900 for C, the most) "
        "and are within 10 points of the most accurate model on every question."
    )
    assert cost_insight([CostPoint("A", 280.0, (0.9, 0.7, 0.6, 0.6))]) == (
        "A is the only model with token counts (280 output tokens per item)."
    )
    assert cost_insight([CostPoint("A", None, (0.9, 0.7, 0.6, 0.6))]) == "No model reports output tokens."


# ---- thinking twins ----------------------------------------------------------------------------------


def test_thinking_biggest_change_with_token_ratio():
    rows = [
        TwinRow("A", (0.55, 0.5, 0.5, 0.4), (0.86, 0.5, 0.5, 0.4), 365.0, 2271.0),
        TwinRow("B", (0.9, 0.7, 0.6, 0.6), (0.9, 0.74, 0.6, 0.55), 300.0, 300.0),
        TwinRow("C", (0.9, 0.7, 0.6, 0.6), (0.92, 0.7, 0.6, 0.6), None, None),
    ]
    assert thinking_insight(rows, "min", "default") == (
        "From min to default thinking, A changes the most (+31 points on injury type, at 6.2x the output tokens); "
        "every other model changes by at most 5 points on any question."
    )


def test_thinking_ties_one_model_and_no_change():
    tied = [
        TwinRow("A", (0.5, 0.5, 0.5, 0.5), (0.6, 0.5, 0.5, 0.5), 100.0, 150.0),
        TwinRow("B", (0.5, 0.6, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5), None, 10.0),
        TwinRow("C", (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.52, 0.5), 10.0, 10.0),
    ]
    assert thinking_insight(tied, "min", "default") == (
        "From min to default thinking, A (+10 points on injury type, at 1.5x the output tokens) and "
        "B (-10 points on severity) change the most; the other model changes by at most 2 points on any question."
    )
    one = [TwinRow("A", (0.55, 0.5, 0.5, 0.4), (0.86, 0.5, 0.5, 0.4), 365.0, 2271.0)]
    assert thinking_insight(one, "min", "default") == (
        "From min to default thinking, A's largest change is +31 points on injury type, at 6.2x the output tokens."
    )
    same = [TwinRow("A", (0.5,) * 4, (0.501,) * 4, 1.0, 1.0)]
    assert thinking_insight(same, "low", "high") == (
        "From low to high thinking, no model's accuracy changes by a whole point on any question."
    )
    assert thinking_insight([TwinRow("A", NONE4, NONE4, None, None)], "min", "default") == ("No model has accuracy in both runs.")


# ---- cost and speed table ----------------------------------------------------------------------------


def speed(name: str, out: float | None, p50: float | None, cost: float | None = None) -> SpeedRow:
    return SpeedRow(name, 1360.0, out, p50, None if p50 is None else p50 * 1.5, None if cost is None else cost / 4, cost)


def test_speed_fastest_slowest_tokens_and_cost():
    rows = [speed("A", 284.0, 1.47), speed("B", 534.0, 12.66), speed("C", 220.0, 3.29)]
    assert speed_insight(rows) == (
        "A is the fastest per reply (median 1.5 s) and B the slowest (12.7 s); "
        "B writes the most output tokens per item (534) and C the fewest (220)."
    )
    priced = [speed("A", 284.0, 1.47, 0.5), speed("B", 534.0, 12.66, 1.25), speed("C", 220.0, 3.29, 0.25)]
    assert speed_insight(priced).endswith("; the whole run cost $2.00.")
    partly = [speed("A", 284.0, 1.47, 0.5), speed("B", 534.0, 12.66, 1.25), speed("C", 220.0, 3.29)]
    assert speed_insight(partly).endswith("; the 2 priced models cost $1.75 in all.")


def test_speed_ties_one_model_and_no_data():
    tied = [speed("A", 300.0, 1.52), speed("B", 300.0, 1.48), speed("C", 200.0, 9.0)]  # both show as 1.5 s
    assert speed_insight(tied) == (
        "A and B are the fastest per reply (median 1.5 s) and C the slowest (9.0 s); "
        "A and B write the most output tokens per item (300) and C the fewest (200)."
    )
    assert speed_insight([speed("A", 284.0, 1.47)]) == "A takes a median 1.5 s per reply and writes 284 output tokens per item."
    empty = [SpeedRow("A", None, None, None, None, None, None)]
    assert speed_insight(empty) == "No latency, token or price data for this run."
