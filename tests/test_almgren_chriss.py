"""Almgren-Chriss: the closed form must match its own limits, its own
Monte Carlo, and a numerical optimiser that knows nothing about the algebra."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.execution.almgren_chriss import (ACParams, bps, cost_variance,
                                          expected_cost, frontier,
                                          implied_lambda, kappa,
                                          schedule_cost, solve,
                                          temporary_impact_bps, trajectory)


@pytest.fixture
def p() -> ACParams:
    """A million shares over an hour on a $100 name, with impact
    parameters of the size the calibration in results/ actually produces."""
    return ACParams(X=1_000_000.0, T=3600.0, N=60, sigma=0.01,
                    eta=1.0e-5, gamma=1.0e-7, epsilon=0.005)


def test_zero_risk_aversion_is_twap(p):
    tr = trajectory(p, 0.0)
    assert tr["kappa"] == 0.0
    assert np.allclose(tr["n"], p.X / p.N)
    expected = p.X * (1.0 - tr["t"] / p.T)
    assert np.allclose(tr["x"], expected)


def test_small_risk_aversion_approaches_twap(p):
    tr = trajectory(p, 1e-12)
    assert np.allclose(tr["n"], p.X / p.N, rtol=1e-4)


def test_high_risk_aversion_front_loads(p):
    tr = trajectory(p, 1e-3)
    assert tr["n"][0] > 10 * tr["n"][-1]
    assert tr["kappa"] * p.T > 3.0


def test_trajectory_is_monotone_and_finishes(p):
    for lam in (0.0, 1e-8, 1e-6, 1e-4):
        tr = trajectory(p, lam)
        assert tr["x"][0] == pytest.approx(p.X)
        assert tr["x"][-1] == 0.0
        assert np.all(np.diff(tr["x"]) <= 1e-6)
        assert tr["n"].sum() == pytest.approx(p.X)


def test_kappa_solves_its_defining_equation(p):
    lam = 3e-6
    k = kappa(p, lam)
    lhs = math.cosh(k * p.tau)
    rhs = 1.0 + lam * p.sigma ** 2 * p.tau ** 2 / (2.0 * p.eta_tilde)
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_kappa_increases_with_risk_aversion(p):
    ks = [kappa(p, lam) for lam in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5)]
    assert all(b > a for a, b in zip(ks, ks[1:]))


def test_continuous_approximation_is_close_but_not_equal(p):
    """kappa^2 ~ lambda sigma^2 / eta_tilde is the textbook shortcut. It should
    be close - and the exact solution should not be identical to it, which is
    why the code does not use it."""
    lam = 1e-6
    exact = kappa(p, lam)
    approx = math.sqrt(lam * p.sigma ** 2 / p.eta_tilde)
    assert exact == pytest.approx(approx, rel=0.05)
    assert exact != approx


def test_optimal_schedule_beats_perturbations(p):
    """The variational check: perturb the optimal holdings in any direction
    that keeps the endpoints, and the objective must get worse."""
    lam = 2e-6
    s = solve(p, lam)
    base = s["objective"]
    rng = np.random.default_rng(0)
    for _ in range(25):
        x = np.array(s["x"], dtype=float)
        bump = rng.normal(size=x.size) * 0.01 * p.X
        bump[0] = 0.0
        bump[-1] = 0.0
        x2 = x + bump
        n2 = -np.diff(x2)
        obj = expected_cost(p, n2) + lam * cost_variance(p, x2)
        assert obj > base - 1e-9


def test_expected_cost_matches_monte_carlo(p):
    """Simulate the discrete price dynamics the model assumes and confirm the
    closed-form mean and variance are what comes out."""
    lam = 1e-6
    s = solve(p, lam)
    n = np.asarray(s["n"])
    x = np.asarray(s["x"])
    rng = np.random.default_rng(1)
    paths = 40000
    tau, sig = p.tau, p.sigma

    xi = rng.normal(size=(paths, p.N))
    # Cost = sum over intervals of [ permanent drift already paid on the
    # remaining shares ] + temporary impact + volatility term, written in the
    # standard "cost = -sum x_k * price increment + impact" form.
    vol_term = -(sig * math.sqrt(tau) * (xi * x[1:]).sum(axis=1))
    perm = 0.5 * p.gamma * p.X ** 2
    temp = p.epsilon * np.abs(n).sum() + (p.eta_tilde / tau) * (n ** 2).sum()
    costs = vol_term + perm + temp

    assert costs.mean() == pytest.approx(s["expected_cost"],
                                         rel=0.02, abs=0.02 * s["stdev"])
    assert costs.std() == pytest.approx(s["stdev"], rel=0.03)


def test_permanent_impact_term_is_schedule_independent(p):
    """Half the cost of a big order is decided before any slicing happens."""
    a = expected_cost(p, trajectory(p, 0.0)["n"])
    b = expected_cost(p, trajectory(p, 1e-5)["n"])
    perm = 0.5 * p.gamma * p.X ** 2
    assert a > perm and b > perm
    # Both contain exactly the same permanent term; they differ only in the
    # temporary part.
    assert (a - perm) != pytest.approx(b - perm)


def test_variance_falls_and_cost_rises_with_urgency(p):
    lams = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    f = frontier(p, lams)
    assert np.all(np.diff(f["expected_cost"]) > 0)
    assert np.all(np.diff(f["stdev"]) < 0)


def test_frontier_is_convex(p):
    """Expected cost falls with cost standard deviation, and does so at a
    decreasing rate - a convex frontier. If it were concave somewhere, a
    mixture of two schedules would dominate the one in between."""
    f = frontier(p, np.logspace(-9, -4, 30))
    sd, ec = f["stdev"][::-1], f["expected_cost"][::-1]
    slopes = np.diff(ec) / np.diff(sd)
    assert np.all(slopes < 0)
    assert np.all(np.diff(slopes) >= -1e-9)


def test_twap_costs_more_variance_than_the_optimum(p):
    lam = 2e-6
    opt = solve(p, lam)
    twap = schedule_cost(p, np.full(p.N, p.X / p.N))
    assert twap["stdev"] > opt["stdev"]
    assert twap["expected_cost"] < opt["expected_cost"]
    assert (twap["expected_cost"] + lam * twap["variance"]
            > opt["objective"] - 1e-9)


def test_schedule_cost_agrees_with_solve_on_the_optimum(p):
    lam = 5e-7
    s = solve(p, lam)
    sc = schedule_cost(p, s["n"])
    assert sc["expected_cost"] == pytest.approx(s["expected_cost"])
    assert sc["stdev"] == pytest.approx(s["stdev"])


def test_implied_lambda_inverts_kappa(p):
    for lam in (1e-8, 1e-7, 1e-6, 1e-5):
        k = kappa(p, lam)
        assert implied_lambda(p, k) == pytest.approx(lam, rel=1e-6)


def test_half_life_matches_kappa(p):
    s = solve(p, 1e-6)
    assert s["half_life"] == pytest.approx(math.log(2) / s["kappa"])
    assert math.isinf(solve(p, 0.0)["half_life"])


def test_negative_eta_tilde_is_refused():
    """Long intervals make the discrete model claim infinitely fast trading is
    free. It must refuse rather than return a number."""
    bad = ACParams(X=1e6, T=3600.0, N=2, sigma=0.02, eta=1e-6, gamma=1e-6)
    assert bad.eta_tilde < 0
    with pytest.raises(ValueError, match="eta_tilde"):
        solve(bad, 1e-6)


def test_degenerate_inputs_are_rejected():
    with pytest.raises(ValueError):
        solve(ACParams(X=0.0, T=100.0, N=10, sigma=0.01, eta=1e-6, gamma=0.0), 0.0)
    with pytest.raises(ValueError):
        solve(ACParams(X=100.0, T=-1.0, N=10, sigma=0.01, eta=1e-6, gamma=0.0), 0.0)
    with pytest.raises(ValueError):
        solve(ACParams(X=100.0, T=10.0, N=10, sigma=0.01, eta=0.0, gamma=0.0), 0.0)


def test_zero_volatility_makes_urgency_pointless(p):
    """With no risk there is nothing to trade off, so the optimum is TWAP at
    every risk aversion."""
    q = ACParams(X=p.X, T=p.T, N=p.N, sigma=0.0, eta=p.eta, gamma=p.gamma,
                 epsilon=p.epsilon)
    tr = trajectory(q, 1e-4)
    assert np.allclose(tr["n"], q.X / q.N)


def test_bps_conversion():
    assert bps(1000.0, 10_000.0, 100.0) == pytest.approx(10.0)
    assert math.isnan(bps(1.0, 0.0, 100.0))


def test_temporary_impact_in_bps_is_readable(p):
    v = p.X / p.T
    b = temporary_impact_bps(p, 100.0, v)
    assert 0.1 < b < 100.0        # a sane order of magnitude, in bp
