"""Metric formulas: the one home for them (docs/PLAN.md §9.2). Pure functions, no I/O.

Choices and the alternatives considered: WORKLOG.md, "2026-10-12: Session 2, stats + plots (Noa)",
"Statistics methods: DECIDED".
- Accuracy CIs: Wilson score interval, 95%. It stays inside [0, 1] and behaves at 0/n and n/n.
- Paired differences (two systems on the same items): percentile bootstrap over items, 10,000
  resamples, seed 0, so reruns give the same CI.
- Paired significance: exact two-sided McNemar (binomial on the discordant pairs); Holm across
  a family of comparisons.
- Calibration (Part 2): Brier score and expected calibration error over 10 equal-width bins.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb, sqrt

import numpy as np
from numpy.typing import ArrayLike

Z95 = 1.959963984540054  # standard normal quantile for a two-sided 95% interval
N_BOOT = 10_000
SEED = 0
ECE_BINS = 10


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n <= 0:
        raise ValueError("wilson_ci needs n > 0")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_ci(x: ArrayLike, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Mean of x with a 95% percentile bootstrap CI over its items: (mean, lo, hi)."""
    v = np.asarray(x, dtype=float)
    if not v.size:
        raise ValueError("bootstrap_mean_ci needs a non-empty list")
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """mean(a) - mean(b) over paired items, with a 95% percentile bootstrap CI: (diff, lo, hi)."""
    if len(a) != len(b) or not len(a):
        raise ValueError(f"paired_bootstrap_diff needs two equal, non-empty lists (got {len(a)} and {len(b)})")
    return bootstrap_mean_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n_boot, seed)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value; b and c are the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error between predicted probabilities and 0/1 outcomes."""
    p, y = np.asarray(probs, dtype=float), np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece(probs: Sequence[float], outcomes: Sequence[int], bins: int = ECE_BINS) -> float:
    """Expected calibration error: bin-size-weighted |accuracy - confidence| over equal-width bins."""
    p, y = np.asarray(probs, dtype=float), np.asarray(outcomes, dtype=float)
    which = np.minimum((p * bins).astype(int), bins - 1)  # p = 1.0 goes in the last bin
    total = 0.0
    for b in np.unique(which):
        mask = which == b
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)
