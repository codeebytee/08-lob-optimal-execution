"""The venue: does the simulated session behave like the market it claims to?"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.lob.book import BUY, SELL
from src.lob.simulator import MarketSimulator
from src.utils.config import u_shape


def make(stats, book_cfg, flow_cfg, seed=1, **kw):
    return MarketSimulator(stats, book_cfg, flow_cfg,
                           np.random.default_rng(seed), **kw)


def test_book_stays_consistent_through_a_session(stats, book_cfg, flow_cfg):
    sim = make(stats, book_cfg, flow_cfg)
    sim.run_until(300.0)
    sim.book.check_invariants()
    assert sim.n_events > 100
    assert sim.book.best_bid < sim.book.best_ask


def test_same_seed_reproduces_the_session(stats, book_cfg, flow_cfg):
    a = make(stats, book_cfg, flow_cfg, seed=7)
    b = make(stats, book_cfg, flow_cfg, seed=7)
    a.run_until(200.0)
    b.run_until(200.0)
    assert a.n_events == b.n_events
    assert a.mid == b.mid
    assert a.market_lots_traded == b.market_lots_traded


def test_different_seeds_diverge(stats, book_cfg, flow_cfg):
    a = make(stats, book_cfg, flow_cfg, seed=7)
    b = make(stats, book_cfg, flow_cfg, seed=8)
    a.run_until(200.0)
    b.run_until(200.0)
    assert a.n_events != b.n_events or a.mid != b.mid


def test_stepping_in_pieces_equals_one_step(stats, book_cfg, flow_cfg):
    """run_until must be a pure function of the end time, or the execution
    loop's slice boundaries would perturb the market they measure."""
    a = make(stats, book_cfg, flow_cfg, seed=3)
    a.run_until(120.0)
    b = make(stats, book_cfg, flow_cfg, seed=3)
    for t in np.arange(10.0, 120.1, 10.0):
        b.run_until(float(t))
    assert a.n_events == b.n_events
    assert a.mid == pytest.approx(b.mid)
    assert a.market_lots_traded == b.market_lots_traded


def test_realised_volatility_matches_the_calibration(stats, book_cfg, flow_cfg):
    """The mid's realised vol should land near the target the name was
    calibrated to. This is the check that the venue is not quietly ten times
    too jumpy."""
    target = stats.sigma_per_second(23400.0)
    est = []
    for seed in range(6):
        sim = make(stats, book_cfg, flow_cfg, seed=seed, kyle_lambda=0.0,
                   record_every=5.0)
        sim.run_until(900.0)
        mids = np.array([0.5 * (s.best_bid + s.best_ask) for s in sim.snapshots])
        est.append(np.std(np.diff(mids)) / math.sqrt(5.0))
    assert np.mean(est) == pytest.approx(target, rel=0.35)


def test_mid_is_close_to_a_martingale(stats, book_cfg, flow_cfg):
    """Order-flow-only books mean-revert. This one should not: the variance of
    the mid must grow roughly linearly in time, so the variance ratio between
    two horizons is near their ratio."""
    short, long = [], []
    for seed in range(12):
        sim = make(stats, book_cfg, flow_cfg, seed=seed, record_every=10.0)
        sim.run_until(600.0)
        mids = np.array([0.5 * (s.best_bid + s.best_ask) for s in sim.snapshots])
        d1 = np.diff(mids)                       # 10-second changes
        d4 = mids[4::4] - mids[:-4:4]            # 40-second changes
        short.append(np.var(d1))
        long.append(np.var(d4))
    vr = np.mean(long) / (4.0 * np.mean(short))
    assert 0.6 < vr < 1.6


def test_volume_scales_with_the_flow_rates(stats, book_cfg, flow_cfg):
    """Doubling every rate doubles the printed volume - the property the ADV
    calibration relies on."""
    base, fast = [], []
    doubled = dataclasses.replace(
        flow_cfg, limit_k=flow_cfg.limit_k * 2, market_rate=flow_cfg.market_rate * 2,
        cancel_theta=flow_cfg.cancel_theta * 2,
        stale_cancel_theta=flow_cfg.stale_cancel_theta * 2,
        hawkes_alpha=flow_cfg.hawkes_alpha * 2, hawkes_beta=flow_cfg.hawkes_beta * 2)
    for seed in range(4):
        a = make(stats, book_cfg, flow_cfg, seed=seed)
        a.run_until(400.0)
        b = make(stats, book_cfg, doubled, seed=seed)
        b.run_until(400.0)
        base.append(a.market_lots_traded)
        fast.append(b.market_lots_traded)
    assert np.mean(fast) / np.mean(base) == pytest.approx(2.0, rel=0.2)


def test_depth_profile_is_stationary(stats, book_cfg, flow_cfg):
    """Depth at the touch should not trend over a session - a book that grows
    without bound or empties out is not a market."""
    sim = make(stats, book_cfg, flow_cfg, seed=5, record_every=10.0)
    sim.run_until(1200.0)
    touch = np.array([s.bid_lots[0] + s.ask_lots[0] for s in sim.snapshots])
    first, last = touch[:len(touch) // 3], touch[-len(touch) // 3:]
    assert 0.4 < last.mean() / max(first.mean(), 1e-9) < 2.5
    assert 0 < touch.mean() < 200


def test_spread_is_a_few_ticks(stats, book_cfg, flow_cfg):
    sim = make(stats, book_cfg, flow_cfg, seed=2, record_every=5.0)
    sim.run_until(900.0)
    spr = np.array([(s.best_ask - s.best_bid) / stats.tick_size
                    for s in sim.snapshots])
    assert 1.0 <= np.median(spr) <= 6.0


def test_kyle_impact_moves_the_price_the_stated_amount(stats, book_cfg, flow_cfg):
    """A pure mechanical check: with no flow and no exogenous vol, buying N
    lots must move the latent price by exactly lambda * N."""
    quiet = dataclasses.replace(flow_cfg, market_rate=0.0, limit_k=1e-9,
                                cancel_theta=0.0, stale_cancel_theta=0.0,
                                hawkes_alpha=0.0)
    sim = make(stats, book_cfg, quiet, seed=1, sigma_exo_per_sec=0.0,
               kyle_lambda=0.001)
    before = sim.latent
    fills = sim.agent_market(BUY, 500)             # 5 lots
    lots = sum(f.shares for f in fills) // book_cfg.lot_size
    assert lots > 0
    assert sim.latent - before == pytest.approx(0.001 * lots)


def test_agent_market_order_returns_only_its_own_fills(stats, book_cfg, flow_cfg):
    sim = make(stats, book_cfg, flow_cfg, seed=4)
    sim.run_until(60.0)
    fills = sim.agent_market(BUY, 300)
    assert sum(f.shares for f in fills) <= 300
    assert all(not f.passive for f in fills)
    assert sim.agent_shares_done == sum(f.shares for f in fills)


def test_agent_order_below_one_lot_does_nothing(stats, book_cfg, flow_cfg):
    sim = make(stats, book_cfg, flow_cfg, seed=4)
    sim.run_until(30.0)
    assert sim.agent_market(BUY, 37) == []
    assert sim.agent_limit(BUY, 0, 37) == 0


def test_agent_limit_order_can_fill_passively(stats, book_cfg, flow_cfg):
    """Rest at the touch for long enough and anonymous flow should hit it."""
    filled = 0
    for seed in range(8):
        sim = make(stats, book_cfg, flow_cfg, seed=seed)
        sim.run_until(60.0)
        sim.agent_limit(BUY, 0, 200)
        before = sim.agent_shares_done
        sim.run_until(300.0)
        filled += sim.agent_shares_done - before
    assert filled > 0


def test_cancelling_agent_orders_removes_them(stats, book_cfg, flow_cfg):
    sim = make(stats, book_cfg, flow_cfg, seed=6)
    sim.run_until(60.0)
    sim.agent_limit(BUY, 2, 500)
    assert sim.agent_open_shares() > 0
    sim.cancel_agent_orders()
    assert sim.agent_open_shares() == 0
    sim.book.check_invariants()


def test_anonymous_volume_excludes_the_agent(stats, book_cfg, flow_cfg):
    sim = make(stats, book_cfg, flow_cfg, seed=6)
    sim.run_until(120.0)
    before = sim.market_lots_traded
    sim.agent_market(BUY, 1000)
    assert sim.market_lots_traded == before


def test_u_shape_multiplier_matches_the_array_version(stats, book_cfg, flow_cfg):
    """The simulator's scalar fast path must agree with utils.config."""
    sim = make(stats, book_cfg, flow_cfg, start_fraction=0.0)
    for t in (0.0, 100.0, 5000.0, 20000.0):
        u = math.floor(t / 30.0) * 30.0 / sim.seconds_per_day
        assert sim._multiplier(t) == pytest.approx(float(u_shape(u, flow_cfg)))


def test_u_shape_is_mean_one_over_the_day(flow_cfg):
    u = np.linspace(0, 1, 20001)
    assert float(np.mean(u_shape(u, flow_cfg))) == pytest.approx(1.0, rel=1e-3)


def test_u_shape_is_heavier_at_the_ends(flow_cfg):
    assert u_shape(0.0, flow_cfg) > u_shape(0.5, flow_cfg)
    assert u_shape(1.0, flow_cfg) > u_shape(0.5, flow_cfg)


def test_exogenous_path_is_shared_across_runs(stats, book_cfg, flow_cfg):
    """Two simulators on the same seed must see the same price path even when
    one of them trades - this is what makes the counterfactual work."""
    a = make(stats, book_cfg, flow_cfg, seed=21)
    b = make(stats, book_cfg, flow_cfg, seed=21)
    a.run_until(60.0)
    a.agent_market(BUY, 5000)
    a.run_until(300.0)
    b.run_until(300.0)
    assert a.exo_seed == b.exo_seed
    assert a._exogenous(300.0) == pytest.approx(b._exogenous(300.0))


def test_zero_volatility_leaves_the_price_moving_only_on_impact(stats, book_cfg,
                                                                flow_cfg):
    sim = make(stats, book_cfg, flow_cfg, seed=3, sigma_exo_per_sec=0.0,
               kyle_lambda=0.0)
    sim.run_until(300.0)
    assert sim.latent == pytest.approx(stats.price)


def test_vol_multiplier_scales_the_price_moves(stats, book_cfg, flow_cfg):
    calm, wild = [], []
    for seed in range(5):
        for mult, bucket in ((0.5, calm), (2.0, wild)):
            sim = make(stats, book_cfg, flow_cfg, seed=seed, record_every=10.0,
                       vol_multiplier=mult, kyle_lambda=0.0)
            sim.run_until(600.0)
            mids = np.array([0.5 * (s.best_bid + s.best_ask)
                             for s in sim.snapshots])
            bucket.append(np.std(np.diff(mids)))
    assert np.mean(wild) / np.mean(calm) == pytest.approx(4.0, rel=0.35)


def test_snapshots_are_recorded_on_schedule(stats, book_cfg, flow_cfg):
    sim = make(stats, book_cfg, flow_cfg, seed=9, record_every=25.0)
    sim.run_until(500.0)
    ts = [s.t for s in sim.snapshots]
    assert ts[0] == 0.0
    assert np.allclose(np.diff(ts), 25.0)
    assert len(sim.snapshots[0].bid_lots) == 8
