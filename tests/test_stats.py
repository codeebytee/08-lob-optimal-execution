"""Cost statistics and the paired comparison machinery."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.stats import (bootstrap_ci, cost_stats, histogram,
                             mean_variance_objective, paired_test)


def test_cost_stats_on_a_known_sample():
    x = np.arange(0.0, 101.0)
    s = cost_stats(x)
    assert s.n == 101
    assert s.mean == pytest.approx(50.0)
    assert s.median == pytest.approx(50.0)
    assert s.p05 == pytest.approx(5.0)
    assert s.p95 == pytest.approx(95.0)
    assert s.worst == 100.0 and s.best == 0.0


def test_cvar_is_the_mean_of_the_tail_not_the_quantile():
    x = np.concatenate([np.zeros(95), np.array([10.0, 20, 30, 40, 500])])
    s = cost_stats(x)
    assert s.p95 < s.cvar95
    assert s.cvar95 == pytest.approx(np.mean(x[x >= np.percentile(x, 95)]))


def test_cost_stats_ignores_non_finite():
    s = cost_stats([1.0, 2.0, np.nan, np.inf, 3.0])
    assert s.n == 3 and s.mean == pytest.approx(2.0)


def test_cost_stats_on_empty_input():
    s = cost_stats([])
    assert s.n == 0 and np.isnan(s.mean)


def test_stderr_shrinks_with_sample_size():
    rng = np.random.default_rng(0)
    small = cost_stats(rng.normal(size=100))
    big = cost_stats(rng.normal(size=10_000))
    assert big.stderr < small.stderr / 5.0


def test_paired_test_detects_a_small_shift_that_unpaired_would_miss():
    """The whole reason for common random numbers: a 0.2 shift buried in noise
    of standard deviation 10 is invisible unpaired and obvious paired."""
    rng = np.random.default_rng(1)
    common = rng.normal(scale=10.0, size=400)
    a = common + rng.normal(scale=0.05, size=400)
    b = common + 0.2 + rng.normal(scale=0.05, size=400)
    t = paired_test(a, b)
    assert t.mean_diff == pytest.approx(-0.2, abs=0.02)
    assert t.t_stat < -10.0
    unpaired_se = np.sqrt(np.var(a, ddof=1) / 400 + np.var(b, ddof=1) / 400)
    assert abs(np.mean(a) - np.mean(b)) / unpaired_se < 1.0


def test_paired_test_win_rate_and_sign():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    t = paired_test(a, b)
    assert t.mean_diff == pytest.approx(-1.0)
    assert t.win_rate == 1.0


def test_paired_test_handles_a_tie():
    t = paired_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert t.mean_diff == 0.0
    assert np.isnan(t.t_stat)
    assert t.win_rate == 0.0


def test_paired_test_truncates_to_the_shorter_sample():
    t = paired_test([1.0, 2.0, 3.0, 4.0], [0.0, 1.0])
    assert t.n == 2


def test_histogram_shares_bins_when_a_range_is_given():
    a = histogram([1.0, 2.0, 3.0], bins=10, lo=0.0, hi=10.0)
    b = histogram([7.0, 8.0], bins=10, lo=0.0, hi=10.0)
    assert a["edges"] == b["edges"]
    assert sum(a["counts"]) == 3 and sum(b["counts"]) == 2


def test_histogram_of_empty_input():
    h = histogram([])
    assert h["edges"] == [] and h["counts"] == []


def test_histogram_handles_a_degenerate_range():
    h = histogram([5.0, 5.0, 5.0], bins=4)
    assert sum(h["counts"]) == 3


def test_mean_variance_objective_penalises_dispersion():
    tight = [5.0, 5.0, 5.0, 5.0, 5.0]
    wide = [0.0, 10.0, 0.0, 10.0, 5.0]
    assert np.mean(tight) == pytest.approx(np.mean(wide))
    assert mean_variance_objective(wide, 0.1) > mean_variance_objective(tight, 0.1)
    # At zero risk aversion the two are indifferent.
    assert mean_variance_objective(wide, 0.0) == pytest.approx(
        mean_variance_objective(tight, 0.0))


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    x = rng.normal(loc=3.0, scale=1.0, size=500)
    ci = bootstrap_ci(x, n_boot=500, seed=3)
    assert ci["lo"] < ci["point"] < ci["hi"]
    assert ci["lo"] < 3.0 < ci["hi"]


def test_bootstrap_needs_data():
    ci = bootstrap_ci([1.0])
    assert np.isnan(ci["point"])
