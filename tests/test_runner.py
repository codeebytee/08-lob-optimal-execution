"""Executing a parent order end to end: completion, cost sign, and the
look-ahead guard."""

from __future__ import annotations

import numpy as np
import pytest

from src.execution.almgren_chriss import ACParams
from src.execution.runner import (apply_control_variate, baseline_path,
                                  control_variate, run_parent)
from src.execution.schedules import POV, TWAP, VWAP, AlmgrenChriss
from src.lob.book import BUY, SELL

HORIZON = 600.0
SLICES = 10


def go(stats, book_cfg, flow_cfg, algo, X=20_000, seed=1, **kw):
    return run_parent(stats, book_cfg, flow_cfg, algo, X, seed, HORIZON,
                      SLICES, **kw)


def test_parent_order_completes(stats, book_cfg, flow_cfg):
    res = go(stats, book_cfg, flow_cfg, TWAP())
    assert res.filled_shares == res.target_shares
    assert res.completed
    assert np.isfinite(res.avg_price)


def test_buying_costs_money_on_average(stats, book_cfg, flow_cfg):
    """Across paths, a buy programme must pay something: the spread at least.
    A model where buying is free is a model with a sign error in it."""
    out = []
    for seed in range(12):
        res = go(stats, book_cfg, flow_cfg, TWAP(), seed=seed,
                 kyle_lambda=0.002)
        base = baseline_path(stats, book_cfg, flow_cfg, seed, HORIZON, SLICES,
                             kyle_lambda=0.002)
        out.append(res.shortfall_bps - control_variate(base, BUY))
    assert np.mean(out) > 0


def test_selling_has_the_mirror_sign(stats, book_cfg, flow_cfg):
    buy, sell = [], []
    for seed in range(10):
        b = go(stats, book_cfg, flow_cfg, TWAP(), seed=seed, side=BUY,
               kyle_lambda=0.002)
        s = go(stats, book_cfg, flow_cfg, TWAP(), seed=seed, side=SELL,
               kyle_lambda=0.002)
        base = baseline_path(stats, book_cfg, flow_cfg, seed, HORIZON, SLICES,
                             kyle_lambda=0.002)
        buy.append(b.shortfall_bps - control_variate(base, BUY))
        sell.append(s.shortfall_bps - control_variate(base, SELL))
    # Both sides pay; neither is systematically free.
    assert np.mean(buy) > 0 and np.mean(sell) > 0


def test_a_bigger_order_costs_more(stats, book_cfg, flow_cfg):
    small, big = [], []
    for seed in range(10):
        base = baseline_path(stats, book_cfg, flow_cfg, seed, HORIZON, SLICES,
                             kyle_lambda=0.002)
        cv = control_variate(base, BUY)
        small.append(go(stats, book_cfg, flow_cfg, TWAP(), X=5_000, seed=seed,
                        kyle_lambda=0.002).shortfall_bps - cv)
        big.append(go(stats, book_cfg, flow_cfg, TWAP(), X=200_000, seed=seed,
                      kyle_lambda=0.002).shortfall_bps - cv)
    assert np.mean(big) > np.mean(small)


def test_common_random_numbers_reproduce(stats, book_cfg, flow_cfg):
    a = go(stats, book_cfg, flow_cfg, TWAP(), seed=5)
    b = go(stats, book_cfg, flow_cfg, TWAP(), seed=5)
    assert a.shortfall_bps == b.shortfall_bps
    assert a.avg_price == b.avg_price


def test_two_algorithms_share_the_same_market(stats, book_cfg, flow_cfg):
    """Same seed, different algorithm: the arrival price and the counterfactual
    path must be identical, or the comparison is not paired."""
    a = go(stats, book_cfg, flow_cfg, TWAP(), seed=5)
    b = go(stats, book_cfg, flow_cfg, VWAP(flow_cfg), seed=5)
    assert a.arrival_mid == b.arrival_mid


def test_slices_cover_the_order(stats, book_cfg, flow_cfg):
    res = go(stats, book_cfg, flow_cfg, TWAP())
    assert len(res.slices) == SLICES
    assert sum(s.filled_shares for s in res.slices) == res.filled_shares
    assert res.slices[-1].remaining == 0


def test_first_slice_sees_no_volume(stats, book_cfg, flow_cfg):
    """The look-ahead guard, observed from outside: POV cannot trade in slice 0
    because no volume has printed yet."""
    algo = POV(rate=0.2)
    res = go(stats, book_cfg, flow_cfg, algo)
    assert res.slices[0].filled_shares == 0
    assert res.filled_shares == res.target_shares      # and still completes


def test_market_vwap_excludes_our_own_prints(stats, book_cfg, flow_cfg):
    """A huge parent must not drag the benchmark it is measured against."""
    small = go(stats, book_cfg, flow_cfg, TWAP(), X=2_000, seed=3,
               kyle_lambda=0.0)
    big = go(stats, book_cfg, flow_cfg, TWAP(), X=400_000, seed=3,
             kyle_lambda=0.0)
    assert small.market_vwap == pytest.approx(big.market_vwap, rel=0.02)


def test_shortfall_identity(stats, book_cfg, flow_cfg):
    """Reconstruct the reported shortfall from its parts."""
    res = go(stats, book_cfg, flow_cfg, TWAP(), X=50_000, seed=4,
             kyle_lambda=0.001)
    exec_part = (res.avg_price - res.arrival_mid) * res.filled_shares
    unfilled = res.target_shares - res.filled_shares
    opp = (res.final_mid - res.arrival_mid) * unfilled
    assert res.shortfall_usd == pytest.approx(exec_part + opp, rel=1e-9)
    assert res.shortfall_bps == pytest.approx(
        1e4 * res.shortfall_usd / (res.target_shares * res.arrival_mid))


def test_unfilled_shares_are_charged_opportunity_cost(stats, book_cfg, flow_cfg):
    """An order too big for the venue must not look cheap. Whatever it fails to
    buy is marked at the final price."""
    res = go(stats, book_cfg, flow_cfg, TWAP(), X=40_000_000, seed=2,
             kyle_lambda=0.001)
    assert res.filled_shares < res.target_shares
    assert res.opportunity_usd != 0.0
    assert np.isfinite(res.shortfall_bps)


def test_control_variate_has_mean_zero(stats, book_cfg, flow_cfg):
    cvs = [control_variate(baseline_path(stats, book_cfg, flow_cfg, seed,
                                         HORIZON, SLICES), BUY)
           for seed in range(40)]
    m, se = np.mean(cvs), np.std(cvs, ddof=1) / np.sqrt(len(cvs))
    assert abs(m) < 3.0 * se + 1e-9


def test_control_variate_reduces_variance(stats, book_cfg, flow_cfg):
    raw, adj = [], []
    for seed in range(25):
        res = go(stats, book_cfg, flow_cfg, TWAP(), X=20_000, seed=seed,
                 kyle_lambda=0.001)
        base = baseline_path(stats, book_cfg, flow_cfg, seed, HORIZON, SLICES,
                             kyle_lambda=0.001)
        apply_control_variate(res, base)
        raw.append(res.shortfall_bps)
        adj.append(res.shortfall_adj_bps)
    assert np.std(adj) < 0.6 * np.std(raw)


def test_passive_child_orders_earn_some_spread(stats, book_cfg, flow_cfg):
    """The passive child type should fill part of the order without crossing,
    and should cost less than the aggressive version on average."""
    passive_frac, cost_p, cost_a = [], [], []
    for seed in range(10):
        p = go(stats, book_cfg, flow_cfg, TWAP(), X=10_000, seed=seed,
               child_type="limit_then_market", kyle_lambda=0.001)
        a = go(stats, book_cfg, flow_cfg, TWAP(), X=10_000, seed=seed,
               kyle_lambda=0.001)
        base = baseline_path(stats, book_cfg, flow_cfg, seed, HORIZON, SLICES,
                             kyle_lambda=0.001)
        cv = control_variate(base, BUY)
        passive_frac.append(p.passive_fill_frac)
        cost_p.append(p.shortfall_bps - cv)
        cost_a.append(a.shortfall_bps - cv)
    assert np.mean(passive_frac) > 0.05
    assert np.mean(cost_p) < np.mean(cost_a)


def test_unknown_child_type_is_refused(stats, book_cfg, flow_cfg):
    with pytest.raises(ValueError, match="child_type"):
        go(stats, book_cfg, flow_cfg, TWAP(), child_type="telepathy")


def test_ac_schedule_front_loads_in_the_simulator(stats, book_cfg, flow_cfg):
    p = ACParams(X=50_000.0, T=HORIZON, N=SLICES,
                 sigma=stats.sigma_per_second(23400.0), eta=1e-5, gamma=1e-7,
                 epsilon=0.005)
    res = go(stats, book_cfg, flow_cfg, AlmgrenChriss(p, 5e-5), X=50_000, seed=1)
    first_half = sum(s.filled_shares for s in res.slices[:SLICES // 2])
    assert first_half > 0.6 * res.filled_shares


def test_baseline_is_the_same_market_minus_the_order(stats, book_cfg, flow_cfg):
    base = baseline_path(stats, book_cfg, flow_cfg, 12, HORIZON, SLICES)
    res = go(stats, book_cfg, flow_cfg, TWAP(), X=100, seed=12)
    # A one-lot parent barely perturbs anything, so the two sessions should end
    # in nearly the same place.
    assert res.arrival_mid == pytest.approx(base.arrival)
    assert res.final_mid == pytest.approx(base.final_mid, abs=0.05)
