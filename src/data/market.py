"""The one place this project touches real market data.

Execution research does not need a tick feed to be honest, but it does need to
be *anchored*: a simulator whose volatility and daily volume are invented will
produce invented costs. So the calibration layer is thin and explicit - four
numbers per name, all estimable from free daily bars:

======================  ====================================================
``price``               last close; the arrival price the simulation starts at
``sigma_annual``        close-to-close realised vol, annualised
``adv_shares``          median daily share volume (median, not mean: one
                        index-rebalance day otherwise doubles the estimate)
``tick_size``           $0.01 for every US name above $1, kept per-name so a
                        tick-constrained study can change it
======================  ====================================================

**The spread is deliberately not on that list, and that is a result rather than
an omission.** Two published estimators recover an effective spread from daily
high/low/close bars:

*Corwin-Schultz (2012).* A two-day high-low range contains two days of variance
but only one spread, while the sum of two single-day ranges contains two of
each. Solving that pair of equations separates them:

    beta  = (ln H_t/L_t)^2 + (ln H_{t+1}/L_{t+1})^2
    gamma = (ln H_{t,t+1}/L_{t,t+1})^2
    alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2)
              - sqrt(gamma / (3 - 2 sqrt 2))
    S     = 2 (e^alpha - 1) / (1 + e^alpha)

*Abdi-Ranaldo (2017).* The close sits inside the spread while the high-low
midpoint estimates the efficient price, so

    S^2 = 4 E[ (c_t - eta_t)(c_t - eta_{t+1}) ],   eta_t = (h_t + l_t)/2

Both are validated in ``tests/test_market_data.py`` against a synthetic Roll
model where the true spread is known, and both recover it. Both then fail on
real US large caps: on 2026 data Corwin-Schultz returns 0.1 bp for SPY and
17 bp for AAPL, and Abdi-Ranaldo returns 47 bp for SPY and 0 for AAPL - two
estimators disagreeing by two orders of magnitude on names whose true quoted
spread is one tick. The reason is structural: both estimators assume the
intraday range is dominated by the spread, and for a name that trades a 90 bp
daily range on a 0.13 bp spread it is dominated by volatility instead.

So the simulator does not take its spread from daily data. It seeds the book at
one tick and lets the flow model decide what the spread becomes; the estimates
above are carried as *diagnostics* and shown on the page as an example of a
measurement that free data cannot make. Pretending otherwise would put a 17 bp
spread on AAPL and inflate every cost number in the project by a factor of ten.

Loader order is cache -> network -> synthetic, and every path records which one
it took in ``MarketSnapshot.source`` so the page can say so out loud.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.config import REPO_ROOT, MarketConfig


@dataclass(frozen=True)
class NameStats:
    """Calibration for a single name - everything the simulator needs."""

    ticker: str
    name: str
    price: float
    sigma_annual: float
    adv_shares: float
    tick_size: float
    spread_ticks: float = 1.0        # what the simulator seeds the book at
    spread_bps_cs: float = float("nan")   # diagnostic: Corwin-Schultz
    spread_bps_ar: float = float("nan")   # diagnostic: Abdi-Ranaldo

    @property
    def adv_usd(self) -> float:
        return self.price * self.adv_shares

    @property
    def sigma_daily(self) -> float:
        return self.sigma_annual / np.sqrt(252.0)

    @property
    def spread_bps(self) -> float:
        """The spread the model actually uses, in bp of price."""
        return self.spread_ticks * self.tick_size / self.price * 1e4

    def sigma_per_second(self, seconds_per_day: float = 23400.0) -> float:
        """Arithmetic (dollar) volatility per second, which is the sigma the
        Almgren-Chriss closed form wants: its cost is in dollars, not in log
        points. Valid as a local approximation over a horizon short enough that
        the price does not travel far from ``price`` - which is the regime the
        whole model lives in anyway."""
        return self.price * self.sigma_annual / np.sqrt(252.0 * seconds_per_day)

    @property
    def half_spread_usd(self) -> float:
        return 0.5 * self.spread_ticks * self.tick_size


@dataclass(frozen=True)
class MarketSnapshot:
    """A dated set of per-name calibrations, plus where it came from."""

    stats: Dict[str, NameStats]
    as_of: str
    source: str          # "cache" | "yfinance" | "synthetic"
    fetched_at: str
    n_days: int

    @property
    def tickers(self) -> List[str]:
        return list(self.stats)

    def __getitem__(self, ticker: str) -> NameStats:
        return self.stats[ticker]

    @property
    def is_real(self) -> bool:
        return self.source != "synthetic"

    def to_frame(self) -> pd.DataFrame:
        rows = [{"ticker": t, "name": s.name, "price": s.price,
                 "sigma_annual": s.sigma_annual, "adv_shares": s.adv_shares,
                 "tick_size": s.tick_size, "spread_ticks": s.spread_ticks,
                 "spread_bps_cs": s.spread_bps_cs,
                 "spread_bps_ar": s.spread_bps_ar}
                for t, s in self.stats.items()]
        return pd.DataFrame(rows).set_index("ticker")


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

_CS_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz_spread(high, low, adjust_overnight: bool = True) -> float:
    """Corwin-Schultz (2012) effective spread as a fraction of price.

    Returned as the mean of the daily estimates with negatives truncated at
    zero, which is what the original paper does. ``adjust_overnight`` applies
    the paper's correction for gaps: when the whole of day t+1's range sits
    outside day t's, the jump between them is an overnight return rather than a
    spread, so the second day's range is shifted back onto the first.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    ok = np.isfinite(high) & np.isfinite(low) & (low > 0) & (high > 0)
    high, low = high[ok], low[ok]
    if high.size < 3:
        return float("nan")

    h1, l1 = high[:-1], low[:-1]
    h2, l2 = high[1:].copy(), low[1:].copy()
    if adjust_overnight:
        gap = np.where(l2 > h1, l2 - h1, np.where(h2 < l1, h2 - l1, 0.0))
        h2 = h2 - gap
        l2 = l2 - gap

    beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
    gamma = np.log(np.maximum(h1, h2) / np.minimum(l1, l2)) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_K - np.sqrt(gamma / _CS_K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isfinite(s), s, 0.0)
    return float(np.mean(np.maximum(s, 0.0)))


def abdi_ranaldo_spread(high, low, close) -> float:
    """Abdi-Ranaldo (2017) closed-form high-low-close spread, as a fraction of
    price.

    ``S = 2 sqrt(max(0, 4 E[(c_t - eta_t)(c_t - eta_{t+1})]))/2``; the outer
    ``max`` is the paper's own treatment of the negative-variance case, which
    happens whenever the close-to-midrange covariance comes out positive.
    """
    h = np.log(np.asarray(high, dtype=float))
    l = np.log(np.asarray(low, dtype=float))
    c = np.log(np.asarray(close, dtype=float))
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    h, l, c = h[ok], l[ok], c[ok]
    if c.size < 3:
        return float("nan")
    eta = 0.5 * (h + l)
    x = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    s2 = 4.0 * float(np.mean(x))
    return float(2.0 * np.sqrt(max(s2, 0.0)) / 2.0)


def close_to_close_vol(close, trading_days: int = 252) -> float:
    """Annualised close-to-close log-return volatility."""
    c = np.asarray(close, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size < 20:
        return float("nan")
    r = np.diff(np.log(c))
    return float(np.std(r, ddof=1) * np.sqrt(trading_days))


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

def _cache_paths(cfg: MarketConfig):
    p = REPO_ROOT / cfg.cache_file
    return p, p.with_suffix(p.suffix + ".meta.txt")


def _read_meta(meta_path: Path) -> Dict[str, str]:
    if not meta_path.exists():
        return {}
    out: Dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_cached(cfg: MarketConfig) -> Optional[MarketSnapshot]:
    csv_path, meta_path = _cache_paths(cfg)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path).set_index("ticker")
    meta = _read_meta(meta_path)
    stats = {}
    for t in cfg.tickers:
        if t not in df.index:
            continue
        row = df.loc[t]
        stats[t] = NameStats(
            ticker=t,
            name=str(row.get("name", cfg.label(t))),
            price=float(row["price"]),
            sigma_annual=float(row["sigma_annual"]),
            adv_shares=float(row["adv_shares"]),
            tick_size=float(row.get("tick_size", cfg.tick_size)),
            spread_ticks=float(row.get("spread_ticks", cfg.seed_spread_ticks)),
            spread_bps_cs=float(row.get("spread_bps_cs", float("nan"))),
            spread_bps_ar=float(row.get("spread_bps_ar", float("nan"))),
        )
    if not stats:
        return None
    return MarketSnapshot(stats=stats, as_of=meta.get("as_of", "unknown"),
                          source=meta.get("source", "cache"),
                          fetched_at=meta.get("fetched_at", "unknown"),
                          n_days=int(meta.get("n_days", 0) or 0))


def fetch_yfinance(cfg: MarketConfig) -> Optional[MarketSnapshot]:
    """Daily bars -> the calibration numbers. Returns None on any failure: a
    missing network is an expected state here, not an exception to raise."""
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        raw = yf.download(list(cfg.tickers), start=cfg.start, auto_adjust=False,
                          progress=False, group_by="column", threads=True)
    except Exception:
        return None
    if raw is None or len(raw) == 0:
        return None

    def col(field: str, ticker: str):
        try:
            return np.asarray(raw[field][ticker], dtype=float)
        except Exception:
            return None

    stats: Dict[str, NameStats] = {}
    for t in cfg.tickers:
        close, high, low, vol = (col("Close", t), col("High", t),
                                 col("Low", t), col("Volume", t))
        if close is None or high is None or low is None or vol is None:
            continue
        finite = np.isfinite(close)
        if int(finite.sum()) < cfg.min_history_days:
            continue
        last = float(close[finite][-1])
        sigma = close_to_close_vol(close[-cfg.vol_window:], cfg.trading_days)
        v = vol[-cfg.volume_window:]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0 or not np.isfinite(sigma):
            continue
        adv = float(np.median(v))
        w = cfg.spread_window
        cs = corwin_schultz_spread(high[-w:], low[-w:])
        ar = abdi_ranaldo_spread(high[-w:], low[-w:], close[-w:])
        stats[t] = NameStats(ticker=t, name=cfg.label(t), price=last,
                             sigma_annual=sigma, adv_shares=adv,
                             tick_size=cfg.tick_size,
                             spread_ticks=float(cfg.seed_spread_ticks),
                             spread_bps_cs=float(cs * 1e4),
                             spread_bps_ar=float(ar * 1e4))
    if len(stats) < 2:
        return None

    as_of = str(pd.Timestamp(raw.index[-1]).date())
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return MarketSnapshot(stats=stats, as_of=as_of, source="yfinance",
                          fetched_at=now, n_days=int(len(raw.index)))


def synthetic_snapshot(cfg: MarketConfig) -> MarketSnapshot:
    """Last resort: plausible but invented numbers, labelled as such.

    Drawn from a seeded RNG so the page still renders a full, self-consistent
    cross-section offline. It is labelled ``synthetic`` everywhere it surfaces,
    because a chart that silently shows invented liquidity is worse than no
    chart at all.
    """
    rng = np.random.default_rng(cfg.synthetic_seed)
    stats = {}
    for t in cfg.tickers:
        price = float(np.round(np.exp(rng.uniform(np.log(20), np.log(400))), 2))
        sigma = float(np.round(rng.uniform(0.15, 0.45), 4))
        adv = float(np.round(np.exp(rng.uniform(np.log(1e6), np.log(6e7))), 0))
        stats[t] = NameStats(ticker=t, name=cfg.label(t), price=price,
                             sigma_annual=sigma, adv_shares=adv,
                             tick_size=cfg.tick_size,
                             spread_ticks=float(cfg.seed_spread_ticks))
    return MarketSnapshot(stats=stats, as_of="synthetic", source="synthetic",
                          fetched_at="n/a", n_days=0)


def load_snapshot(cfg: Optional[MarketConfig] = None,
                  allow_network: bool = False) -> MarketSnapshot:
    """Cache first, then network if asked, then synthetic. Never raises.

    Cache-first is deliberate: a research run has to be reproducible, and a
    loader that silently re-pulls means yesterday's figure cannot be
    regenerated. ``scripts/refresh_data.py`` is the only caller that passes
    ``allow_network=True``.
    """
    cfg = cfg or MarketConfig()
    snap = load_cached(cfg)
    if snap is not None:
        return snap
    if allow_network:
        snap = fetch_yfinance(cfg)
        if snap is not None:
            return snap
    return synthetic_snapshot(cfg)


def write_cache(snap: MarketSnapshot, cfg: MarketConfig) -> Path:
    csv_path, meta_path = _cache_paths(cfg)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    snap.to_frame().to_csv(csv_path, float_format="%.6f")
    meta_path.write_text(
        f"source={snap.source}\nas_of={snap.as_of}\n"
        f"fetched_at={snap.fetched_at}\nn_days={snap.n_days}\n"
        f"tickers={','.join(snap.tickers)}\n"
        "estimators=close-to-close vol (252d), median daily share volume "
        "(63d); spread columns are DIAGNOSTIC ONLY - see src/data/market.py\n",
        encoding="utf-8")
    return csv_path


__all__ = ["NameStats", "MarketSnapshot", "corwin_schultz_spread",
           "abdi_ranaldo_spread", "close_to_close_vol", "load_cached",
           "fetch_yfinance", "synthetic_snapshot", "load_snapshot",
           "write_cache"]
