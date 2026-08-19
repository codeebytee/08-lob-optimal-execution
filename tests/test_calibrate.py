"""Calibration: the scaling identity, the Kyle guess, and impact recovery.

The impact-recovery test is the important one and it is marked slow, because it
runs the whole machine: calibrate a venue, execute parent orders against paired
counterfactual sessions, and check that the coefficient the estimator recovers
is the coefficient the simulator was built with. That is the closest this
project gets to a proof that the measurement is not measuring itself.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.flow.calibrate import (calibrate_venue, kyle_lambda_guess,
                                measure_impact, scale_flow)
from src.lob.simulator import MarketSimulator


def test_scale_flow_multiplies_every_rate(flow_cfg):
    s = scale_flow(flow_cfg, 3.0)
    assert s.limit_k == pytest.approx(3 * flow_cfg.limit_k)
    assert s.market_rate == pytest.approx(3 * flow_cfg.market_rate)
    assert s.cancel_theta == pytest.approx(3 * flow_cfg.cancel_theta)
    assert s.stale_cancel_theta == pytest.approx(3 * flow_cfg.stale_cancel_theta)
    # The Hawkes branching ratio is alpha/beta and must be untouched: scaling
    # time cannot change how clustered the flow is.
    assert s.hawkes_alpha / s.hawkes_beta == pytest.approx(
        flow_cfg.hawkes_alpha / flow_cfg.hawkes_beta)
    # And the shape parameters are exponents, not rates.
    assert s.limit_alpha == flow_cfg.limit_alpha
    assert s.cancel_alpha == flow_cfg.cancel_alpha


def test_scaling_leaves_the_book_shape_alone(stats, book_cfg, flow_cfg):
    """The claim that justifies one-dimensional calibration: scaling every rate
    is a time change, so the stationary depth and spread do not move."""
    def measure(c):
        depth, spread = [], []
        for seed in range(4):
            sim = MarketSimulator(stats, book_cfg, scale_flow(flow_cfg, c),
                                  np.random.default_rng(seed), record_every=5.0,
                                  kyle_lambda=0.0, sigma_exo_per_sec=0.0)
            sim.run_until(600.0)
            depth += [s.bid_lots[0] + s.ask_lots[0] for s in sim.snapshots]
            spread += [(s.best_ask - s.best_bid) / stats.tick_size
                       for s in sim.snapshots]
        return np.mean(depth), np.median(spread)

    d1, s1 = measure(1.0)
    d3, s3 = measure(3.0)
    assert d3 == pytest.approx(d1, rel=0.25)
    assert s3 == pytest.approx(s1, abs=1.0)


def test_kyle_guess_has_the_right_units_and_magnitude(stats):
    lam = kyle_lambda_guess(stats, lot_size=100, c=1.0)
    # Buying a full day's volume should move the price by about one daily
    # standard deviation - that is the identity the guess is built from.
    move = lam * (stats.adv_shares / 100)
    assert move == pytest.approx(stats.price * stats.sigma_daily, rel=1e-9)
    assert lam > 0


def test_kyle_guess_scales_with_the_constant(stats):
    a = kyle_lambda_guess(stats, 100, c=1.0)
    b = kyle_lambda_guess(stats, 100, c=2.5)
    assert b == pytest.approx(2.5 * a)


@pytest.mark.slow
def test_calibration_hits_adv_and_volatility(stats, book_cfg, flow_cfg):
    cal = calibrate_venue(stats, book_cfg, flow_cfg, T=600.0)
    assert cal.sim_adv_shares == pytest.approx(stats.adv_shares, rel=0.25)
    assert cal.sim_sigma_per_sec == pytest.approx(cal.target_sigma_per_sec,
                                                  rel=0.30)
    assert 1.0 <= cal.median_spread_ticks <= 6.0
    assert 0.5 < cal.mean_touch_lots < 60.0
    assert cal.sigma_exo <= cal.target_sigma_per_sec + 1e-12


@pytest.mark.slow
def test_measured_permanent_impact_recovers_the_injected_coefficient(
        stats, book_cfg, flow_cfg):
    """The end-to-end check. gamma is estimated from paired counterfactual
    sessions; the venue was built with a known Kyle coefficient. They must be
    the same number, within the loss that comes from measuring the mid rather
    than the latent price."""
    cal = calibrate_venue(stats, book_cfg, flow_cfg, T=600.0)
    ic = measure_impact(stats, book_cfg, cal, participations=(0.05, 0.10, 0.20),
                        paths=25, horizon=900.0, n_slices=15)
    injected_per_share = cal.kyle_lambda / book_cfg.lot_size
    assert ic.gamma == pytest.approx(injected_per_share, rel=0.5)
    assert ic.gamma > 0
    assert ic.eta > 0
    assert ic.epsilon > 0
    # Cost per share must rise with participation, or the venue is not charging
    # for speed at all.
    costs = [r["cost_per_share"] for r in ic.rows]
    assert costs[-1] > costs[0]
