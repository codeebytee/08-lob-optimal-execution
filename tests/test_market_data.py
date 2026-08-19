"""Market-data estimators, validated where the answer is known.

The two spread estimators are tested against a synthetic Roll model, in which
the true spread is set by hand. That is the only setting where "does this
estimator work" has a definite answer - and it is what licenses the claim in
``src/data/market.py`` that they are implemented correctly but *uninformative*
on liquid names, rather than simply broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.market import (MarketSnapshot, NameStats, abdi_ranaldo_spread,
                             close_to_close_vol, corwin_schultz_spread,
                             load_cached, load_snapshot, synthetic_snapshot,
                             write_cache)
from src.utils.config import MarketConfig


def roll_bars(n_days: int, spread_frac: float, sigma_daily: float,
              ticks_per_day: int, seed: int = 0):
    """Simulate daily bars from a Roll (1984) model.

    An efficient price random walk, and a transaction price that sits half a
    spread either side of it depending on whether the trade was buyer or seller
    initiated. Daily high, low and close are taken from the transaction prices,
    which is exactly the data the estimators claim to work on.
    """
    rng = np.random.default_rng(seed)
    n = n_days * ticks_per_day
    eff = 100.0 * np.exp(np.cumsum(
        rng.normal(scale=sigma_daily / np.sqrt(ticks_per_day), size=n)))
    signs = rng.choice([-1.0, 1.0], size=n)
    px = eff * (1.0 + signs * spread_frac / 2.0)
    px = px.reshape(n_days, ticks_per_day)
    return px.max(axis=1), px.min(axis=1), px[:, -1]


def test_corwin_schultz_recovers_a_known_spread():
    h, l, c = roll_bars(2000, spread_frac=0.01, sigma_daily=0.02,
                        ticks_per_day=40, seed=1)
    est = corwin_schultz_spread(h, l)
    assert est == pytest.approx(0.01, rel=0.35)


def test_abdi_ranaldo_recovers_a_known_spread():
    h, l, c = roll_bars(2000, spread_frac=0.01, sigma_daily=0.02,
                        ticks_per_day=40, seed=2)
    est = abdi_ranaldo_spread(h, l, c)
    assert est == pytest.approx(0.01, rel=0.35)


def test_both_estimators_lose_the_spread_when_volatility_dominates():
    """The finding behind the project's decision not to use them: hold the
    spread fixed, raise the intraday volatility, and the estimates blow up.
    This is why a 0.13 bp SPY spread comes back as 20 bp."""
    kw = dict(spread_frac=0.0005, ticks_per_day=40)
    quiet = roll_bars(1500, sigma_daily=0.002, seed=3, **kw)
    loud = roll_bars(1500, sigma_daily=0.05, seed=3, **kw)
    cs_quiet = corwin_schultz_spread(quiet[0], quiet[1])
    cs_loud = corwin_schultz_spread(loud[0], loud[1])
    assert cs_loud > 5.0 * max(cs_quiet, 1e-9)
    assert cs_loud > 10.0 * 0.0005


def test_corwin_schultz_is_never_negative():
    h, l, _ = roll_bars(500, 0.002, 0.03, 30, seed=4)
    assert corwin_schultz_spread(h, l) >= 0.0


def test_spread_estimators_handle_short_series():
    assert np.isnan(corwin_schultz_spread([100.0], [99.0]))
    assert np.isnan(abdi_ranaldo_spread([100.0], [99.0], [99.5]))


def test_close_to_close_vol_recovers_sigma():
    rng = np.random.default_rng(5)
    sigma_annual = 0.32
    r = rng.normal(scale=sigma_annual / np.sqrt(252), size=4000)
    close = 100.0 * np.exp(np.cumsum(r))
    assert close_to_close_vol(close) == pytest.approx(sigma_annual, rel=0.05)


def test_vol_ignores_non_positive_and_missing_prices():
    close = np.array([100.0, np.nan, 101.0, 0.0, 102.0] * 30)
    v = close_to_close_vol(close)
    assert np.isfinite(v) and v > 0


def test_vol_needs_enough_data():
    assert np.isnan(close_to_close_vol([100.0, 101.0]))


def test_name_stats_unit_conversions():
    s = NameStats(ticker="X", name="X", price=100.0, sigma_annual=0.252,
                  adv_shares=1e7, tick_size=0.01, spread_ticks=2.0)
    assert s.sigma_daily == pytest.approx(0.252 / np.sqrt(252))
    # Arithmetic vol per second: price * annual vol / sqrt(seconds in a year of
    # trading).
    assert s.sigma_per_second(23400.0) == pytest.approx(
        100.0 * 0.252 / np.sqrt(252 * 23400))
    assert s.half_spread_usd == pytest.approx(0.01)
    assert s.spread_bps == pytest.approx(2.0)
    assert s.adv_usd == pytest.approx(1e9)


def test_synthetic_snapshot_is_labelled_and_complete():
    cfg = MarketConfig()
    snap = synthetic_snapshot(cfg)
    assert snap.source == "synthetic" and not snap.is_real
    assert set(snap.tickers) == set(cfg.tickers)
    for t in snap.tickers:
        s = snap[t]
        assert s.price > 0 and 0 < s.sigma_annual < 2 and s.adv_shares > 0


def test_synthetic_snapshot_is_deterministic():
    cfg = MarketConfig()
    a, b = synthetic_snapshot(cfg), synthetic_snapshot(cfg)
    assert a.to_frame().equals(b.to_frame())


def test_cache_roundtrip(tmp_path, monkeypatch):
    import src.data.market as market

    cfg = MarketConfig(cache_file="snap.csv")
    monkeypatch.setattr(market, "REPO_ROOT", tmp_path)
    snap = synthetic_snapshot(cfg)
    path = write_cache(snap, cfg)
    assert path.exists()
    back = load_cached(cfg)
    assert back is not None
    assert back.tickers == snap.tickers
    assert back[cfg.tickers[0]].price == pytest.approx(
        snap[cfg.tickers[0]].price)
    assert "estimators=" in (tmp_path / "snap.csv.meta.txt").read_text()


def test_loader_falls_back_to_synthetic_without_cache_or_network(tmp_path,
                                                                 monkeypatch):
    import src.data.market as market

    monkeypatch.setattr(market, "REPO_ROOT", tmp_path)
    cfg = MarketConfig(cache_file="missing.csv")
    snap = load_snapshot(cfg, allow_network=False)
    assert snap.source == "synthetic"


def test_snapshot_indexing_and_frame():
    snap = synthetic_snapshot(MarketConfig())
    t = snap.tickers[0]
    assert isinstance(snap[t], NameStats)
    df = snap.to_frame()
    assert {"price", "sigma_annual", "adv_shares"} <= set(df.columns)
    assert len(df) == len(snap.tickers)
