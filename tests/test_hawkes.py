"""Hawkes arrivals: theory identities, simulation, and recovery by MLE."""

from __future__ import annotations

import numpy as np
import pytest

from src.flow.hawkes import BivariateHawkes, ExpHawkes, fit_mle


def test_branching_ratio_and_stationarity():
    h = ExpHawkes(mu0=0.5, alpha=0.6, beta=1.2)
    assert h.branching_ratio == pytest.approx(0.5)
    assert h.is_stationary
    assert not ExpHawkes(0.5, 1.3, 1.2).is_stationary
    assert ExpHawkes(0.5, 1.3, 1.2).stationary_intensity() == np.inf


def test_stationary_intensity_identity():
    h = ExpHawkes(mu0=0.4, alpha=0.5, beta=1.0)
    assert h.stationary_intensity() == pytest.approx(0.4 / 0.5)


def test_simulated_rate_matches_theory():
    """The empirical event rate must hit mu0/(1-n) within sampling error."""
    h = ExpHawkes(mu0=0.4, alpha=0.5, beta=1.0)
    rng = np.random.default_rng(4)
    T = 20000.0
    t = h.simulate(T, rng)
    rate = t.size / T
    assert rate == pytest.approx(h.stationary_intensity(), rel=0.08)


def test_poisson_limit():
    """alpha = 0 collapses to a Poisson process of rate mu0."""
    h = ExpHawkes(mu0=0.7, alpha=0.0, beta=1.0)
    rng = np.random.default_rng(5)
    t = h.simulate(5000.0, rng)
    assert t.size / 5000.0 == pytest.approx(0.7, rel=0.06)
    gaps = np.diff(t)
    # Exponential inter-arrivals: mean and standard deviation both 1/rate.
    assert gaps.std() / gaps.mean() == pytest.approx(1.0, rel=0.12)


def test_clustering_shows_up_as_overdispersion():
    """A Hawkes process has a larger variance-to-mean count ratio than Poisson,
    which is the property the execution stress test relies on."""
    rng = np.random.default_rng(6)
    T, window = 6000.0, 10.0
    hawk = ExpHawkes(0.3, 0.6, 1.0).simulate(T, rng)
    pois = ExpHawkes(ExpHawkes(0.3, 0.6, 1.0).stationary_intensity(), 0.0,
                     1.0).simulate(T, rng)
    edges = np.arange(0, T + window, window)
    ch, _ = np.histogram(hawk, bins=edges)
    cp, _ = np.histogram(pois, bins=edges)
    assert ch.var() / ch.mean() > 1.6 * (cp.var() / cp.mean())


def test_loglik_matches_direct_computation():
    """Check the recursive likelihood against the naive O(n^2) version."""
    h = ExpHawkes(0.5, 0.4, 1.1)
    rng = np.random.default_rng(7)
    t = h.simulate(300.0, rng)
    assert t.size > 30

    direct = 0.0
    for i, ti in enumerate(t):
        lam = h.mu0 + h.alpha * np.exp(-h.beta * (ti - t[:i])).sum()
        direct += np.log(lam)
    direct -= h.mu0 * 300.0 + (h.alpha / h.beta) * np.sum(
        1.0 - np.exp(-h.beta * (300.0 - t)))
    assert h.loglik(t, 300.0) == pytest.approx(direct, rel=1e-9)


def test_loglik_is_maximised_at_the_truth():
    """The true parameters must beat perturbed ones on a long sample."""
    truth = ExpHawkes(0.4, 0.5, 1.0)
    rng = np.random.default_rng(8)
    T = 8000.0
    t = truth.simulate(T, rng)
    best = truth.loglik(t, T)
    for h in (ExpHawkes(0.6, 0.5, 1.0), ExpHawkes(0.4, 0.2, 1.0),
              ExpHawkes(0.4, 0.5, 2.5), ExpHawkes(0.2, 0.8, 1.0)):
        assert h.loglik(t, T) < best


@pytest.mark.slow
def test_mle_recovers_parameters():
    truth = ExpHawkes(0.5, 0.6, 1.2)
    rng = np.random.default_rng(9)
    T = 12000.0
    t = truth.simulate(T, rng)
    fit = fit_mle(t, T)
    assert fit.mu0 == pytest.approx(truth.mu0, rel=0.25)
    assert fit.branching_ratio == pytest.approx(truth.branching_ratio, abs=0.12)
    assert fit.beta == pytest.approx(truth.beta, rel=0.45)


def test_residuals_are_unit_exponential():
    """Time-rescaling goodness of fit: residuals under the true model have
    mean and standard deviation 1."""
    truth = ExpHawkes(0.5, 0.5, 1.0)
    rng = np.random.default_rng(10)
    t = truth.simulate(3000.0, rng)
    r = truth.residuals(t)
    assert r.mean() == pytest.approx(1.0, rel=0.12)
    assert r.std() == pytest.approx(1.0, rel=0.18)


def test_residuals_detect_a_wrong_model():
    """Residuals from a badly wrong model are not unit exponential."""
    truth = ExpHawkes(0.4, 0.7, 1.0)
    rng = np.random.default_rng(12)
    t = truth.simulate(3000.0, rng)
    wrong = ExpHawkes(0.4, 0.0, 1.0)         # ignores the clustering
    r = wrong.residuals(t)
    assert abs(r.std() - 1.0) > 0.25


def test_bivariate_excitation_is_split_between_sides():
    h = BivariateHawkes(mu0=1.0, alpha=1.0, beta=2.0, cross=0.25)
    h.reset(0.0)
    h.excite(0.0, side=+1)
    lb, ls = h.intensities(0.0)
    assert lb == pytest.approx(1.0 + 0.75)
    assert ls == pytest.approx(1.0 + 0.25)


def test_bivariate_excitation_decays():
    h = BivariateHawkes(mu0=1.0, alpha=1.0, beta=2.0, cross=0.0)
    h.reset(0.0)
    h.excite(0.0, side=+1)
    lb, _ = h.intensities(np.log(2.0) / 2.0)     # one half-life of the kernel
    assert lb == pytest.approx(1.5, rel=1e-9)


def test_bivariate_disabled_alpha_is_constant():
    h = BivariateHawkes(mu0=0.8, alpha=0.0, beta=1.0)
    h.reset(0.0)
    h.excite(1.0, side=-1)
    assert h.intensities(5.0) == pytest.approx((0.8, 0.8))
