"""asimov.stats against published or hand-computed values."""

import pytest

from asimov.stats import bootstrap_mean_ci, brier, ece, holm, mcnemar_exact, paired_bootstrap_diff, wilson_ci


def test_wilson_matches_newcombe_1998():
    # Newcombe (1998), Stat Med 17:857, Table I: 81/263 -> 0.2553 to 0.3662 (Wilson score interval)
    lo, hi = wilson_ci(81, 263)
    assert lo == pytest.approx(0.2553, abs=5e-5)
    assert hi == pytest.approx(0.3662, abs=5e-5)


def test_wilson_edges_stay_inside_0_1():
    # Newcombe Table I: 0/10 -> 0 to 0.2775; by symmetry 10/10 -> 0.7225 to 1
    assert wilson_ci(0, 10) == pytest.approx((0.0, 0.2775), abs=5e-5)
    assert wilson_ci(10, 10) == pytest.approx((0.7225, 1.0), abs=5e-5)


def test_wilson_rejects_empty():
    with pytest.raises(ValueError):
        wilson_ci(0, 0)


def test_mcnemar_exact_hand_values():
    # discordant 1 vs 9: two-sided binomial p = 2 * (C(10,0) + C(10,1)) / 2**10
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_exact(9, 1) == pytest.approx(22 / 1024)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_holm_textbook():
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> monotone 0.02, 0.03, 0.06, 0.06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # capped at 1
    assert holm([]) == []


def test_paired_bootstrap_identical_and_opposite():
    a = [True, False, True, True, False]
    assert paired_bootstrap_diff(a, a) == (0.0, 0.0, 0.0)
    assert paired_bootstrap_diff([1] * 6, [0] * 6) == (1.0, 1.0, 1.0)


def test_paired_bootstrap_is_seeded_and_brackets_the_diff():
    a = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1] * 5
    b = [1, 0, 0, 0, 1, 0, 1, 0, 0, 1] * 5
    first = paired_bootstrap_diff(a, b)
    assert first == paired_bootstrap_diff(a, b)
    diff, lo, hi = first
    assert diff == pytest.approx(0.3)
    assert lo < diff < hi
    assert paired_bootstrap_diff(a, b, seed=1) != first


def test_paired_bootstrap_needs_pairs():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1])


def test_brier_hand_value():
    assert brier([1.0, 0.0, 0.5], [1, 1, 0]) == pytest.approx((0 + 1 + 0.25) / 3)


def test_ece_hand_value():
    # bin [0.1, 0.2): conf 0.15, acc 0 -> 0.15 x 2/4; bin [0.8, 0.9): conf 0.85, acc 0.5 -> 0.35 x 2/4
    assert ece([0.15, 0.15, 0.85, 0.85], [0, 0, 1, 0]) == pytest.approx(0.25)
    assert ece([1.0, 1.0], [1, 1]) == pytest.approx(0.0)  # p = 1 lands in the last bin


def test_bootstrap_mean_ci():
    assert bootstrap_mean_ci([2.0, 2.0, 2.0]) == (2.0, 2.0, 2.0)
    mean, lo, hi = bootstrap_mean_ci([0.0, 1.0, 2.0, 3.0, 4.0] * 4)
    assert mean == pytest.approx(2.0) and lo < 2.0 < hi
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])
