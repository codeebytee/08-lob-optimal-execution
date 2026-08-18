"""Typed configuration loaded from ``config.yaml``.

The rule for this repo: a number a model needs is either an argument the caller
passes, or it comes from here. Nothing in ``src/`` invents a constant.

The configuration splits along the two halves of the project:

* ``MarketConfig`` / ``BookConfig`` / ``FlowConfig`` describe the *simulated
  venue* - what the book looks like and how anonymous order flow arrives.
* ``ExecutionConfig`` / ``ImpactConfig`` describe the *parent order problem* -
  how large, how urgent, and what trading it costs.

Keeping them apart matters, because the central experiment of the project is to
calibrate the impact parameters from the venue and then ask whether the
closed-form model built on those parameters predicts what the venue actually
charges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class MarketConfig:
    """Which names to calibrate against, and how to estimate each number."""

    tickers: Tuple[str, ...] = ("SPY", "AAPL", "MSFT", "XOM", "KO",
                                "TSLA", "GME", "IWM")
    labels: Dict[str, str] = field(default_factory=dict)
    start: str = "2021-01-04"
    cache_file: str = "data/market_snapshot.csv"
    trading_days: int = 252
    seconds_per_day: float = 23400.0      # 09:30-16:00 US equities, in seconds
    min_history_days: int = 250
    vol_window: int = 252                 # bars used for close-to-close vol
    volume_window: int = 63               # bars used for median daily volume
    spread_window: int = 252              # bars used for the spread diagnostics
    tick_size: float = 0.01
    # The book is seeded at this spread and the flow model takes it from there.
    # It is not estimated from daily bars, because daily bars cannot see it -
    # see the module docstring of src/data/market.py.
    seed_spread_ticks: int = 1
    synthetic_seed: int = 20260818
    default_ticker: str = "MSFT"

    def label(self, ticker: str) -> str:
        return self.labels.get(ticker, ticker)


@dataclass(frozen=True)
class BookConfig:
    """Shape of the simulated limit order book."""

    n_levels: int = 10                    # price levels tracked each side
    tick_size: float = 0.01
    lot_size: int = 100                   # shares per displayed unit
    initial_depth_lots: int = 25          # lots resting on each level at t=0
    depth_decay: float = 0.0              # >0 makes far levels start thinner
    max_queue_lots: int = 100000          # guard against runaway accumulation


@dataclass(frozen=True)
class FlowConfig:
    """Anonymous order flow: the Cont-Stoikov-Talreja intensities, plus the
    Hawkes self-excitation that turns market orders clustered."""

    # Limit orders arrive at distance k ticks from the opposite best with
    # intensity lambda(k) = limit_k / k**limit_alpha, in lots per second.
    limit_k: float = 4.0
    limit_alpha: float = 0.55
    # Market orders, lots per second per side (the Hawkes baseline when
    # hawkes_enabled).
    market_rate: float = 1.0
    # Cancellations, per second per resting lot at distance k:
    # theta(k) = cancel_theta / k**cancel_alpha.
    cancel_theta: float = 0.90
    cancel_alpha: float = 0.35
    # Hazard for quotes the price has walked away from, per lot per second.
    # These are not "deep book"; they are stale, and in a real book they are
    # pulled within milliseconds.
    stale_cancel_theta: float = 2.5
    # Order sizes in lots, geometric with these means.
    limit_size_mean_lots: float = 2.0
    market_size_mean_lots: float = 2.0
    # How far an anonymous market order will walk from the touch. A cap is
    # needed: without one, a single tail-sized order in a thin book sets the
    # session's volatility on its own.
    sweep_limit_ticks: int = 5
    # Hawkes self-excitation on market orders: intensity
    #     mu_t = mu0 + sum_{t_i < t} alpha * exp(-beta (t - t_i)).
    # Branching ratio n = alpha/beta must be < 1 for stationarity.
    hawkes_enabled: bool = True
    hawkes_alpha: float = 0.60
    hawkes_beta: float = 1.30
    hawkes_cross: float = 0.25            # fraction of excitation that spills
                                          # to the opposite side (buy -> sell)
    # The latent efficient price the book is quoted around. Order flow alone
    # produces a mean-reverting price; a latent random walk is what makes the
    # simulated day have the volatility the calibration says it has.
    latent_price: bool = True
    # Kyle-style permanent impact: each net lot traded moves the latent price
    # by kyle_lambda dollars. Calibrated, not guessed - see
    # src/flow/calibrate.py.
    kyle_lambda: float = 0.0              # 0 => calibrate at runtime
    seed: int = 20260818
    # Intraday U-shape on all intensities: multiplier(u) for u in [0,1] is
    # u_a + u_b * ((1-u)**u_p + u**u_p). Volume is heavy at the open and close;
    # this is what gives the VWAP schedule something real to track.
    u_a: float = 0.62
    u_b: float = 0.95
    u_p: float = 3.0


@dataclass(frozen=True)
class ImpactConfig:
    """Almgren-Chriss impact parameters, in the units the closed form wants.

    ``gamma`` is permanent impact in dollars per share traded per share of
    rate; ``eta`` is temporary impact in dollars per share per (share/second).
    Both are set to 0 in the config and *calibrated from the simulator* by
    ``src/flow/calibrate.py``; the fields exist so a calibration can be pinned
    for reproducibility.
    """

    gamma: float = 0.0                    # 0 => calibrate
    eta: float = 0.0                      # 0 => calibrate
    epsilon_from_spread: bool = True      # fixed cost = half spread + fees
    fee_bps: float = 0.15                 # exchange take fee, per side
    # Almgren-Chriss assumes linear temporary impact. The simulator does not:
    # walking a book with finite depth is convex. This exponent is *estimated*
    # from the sim and reported, not imposed.
    fit_impact_exponent: bool = True
    calib_participations: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.07,
                                               0.10, 0.15, 0.20)
    calib_paths: int = 120


@dataclass(frozen=True)
class ExecutionConfig:
    """The parent order and the algorithms that work it."""

    side: str = "buy"                     # buy | sell
    horizon_seconds: float = 1800.0       # 30 minutes
    n_slices: int = 30                    # one child decision per minute
    parent_pct_adv: float = 0.05          # parent size as a fraction of ADV
    risk_aversion: float = 1.0e-6         # lambda, in 1/dollars
    algos: Tuple[str, ...] = ("TWAP", "VWAP", "POV", "AC", "Adaptive")
    pov_rate: float = 0.15                # POV participation target
    adaptive_tilt: float = 1.0            # Almgren-Lorenz aggressiveness-in-
                                          # the-money coefficient
    child_order_type: str = "market"      # market | limit_then_market
    limit_offset_ticks: int = 1           # for the passive child type
    mc_paths: int = 400
    seed: int = 20260818
    # Efficient frontier: risk aversions swept, log-spaced.
    lambda_lo: float = 1.0e-8
    lambda_hi: float = 1.0e-4
    lambda_n: int = 41


@dataclass(frozen=True)
class SweepConfig:
    """The precomputed grid shipped to the page."""

    sizes_pct_adv: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20)
    lambdas: Tuple[float, ...] = (1.0e-7, 1.0e-6, 5.0e-6, 2.0e-5)
    vol_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0)
    paths: int = 300
    tape_seconds: float = 900.0           # length of the animated book replay
    tape_snapshot_every: float = 2.0      # seconds between recorded frames


@dataclass(frozen=True)
class FrontendConfig:
    round_decimals: int = 6
    data_file: str = "docs/data.js"
    hist_bins: int = 41


@dataclass(frozen=True)
class AppConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    book: BookConfig = field(default_factory=BookConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    impact: ImpactConfig = field(default_factory=ImpactConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)

    @staticmethod
    def load(path: Path | str | None = None) -> "AppConfig":
        """Read ``config.yaml``. A missing file is not an error: the dataclass
        defaults above are the same numbers, so the library still runs from an
        arbitrary working directory (a notebook, a test, the build script)."""
        p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        raw: Dict[str, Any] = {}
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        def sub(cls, key):
            d = raw.get(key, {}) or {}
            known = set(cls.__dataclass_fields__)
            clean = {}
            for k, v in d.items():
                if k not in known:
                    continue
                if isinstance(v, list):
                    v = tuple(v)
                clean[k] = v
            return cls(**clean)

        return AppConfig(
            market=sub(MarketConfig, "market"),
            book=sub(BookConfig, "book"),
            flow=sub(FlowConfig, "flow"),
            impact=sub(ImpactConfig, "impact"),
            execution=sub(ExecutionConfig, "execution"),
            sweep=sub(SweepConfig, "sweep"),
            frontend=sub(FrontendConfig, "frontend"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def u_shape(u, cfg: FlowConfig):
    """Intraday activity multiplier at fraction-of-day ``u`` in [0, 1].

    Normalised so the mean over the day is 1, which is the property the rest of
    the code relies on: switching the U-shape on must change *when* volume
    arrives, never how much arrives in total. Without that normalisation, every
    cost number would move when the shape parameters changed and nothing would
    be comparable.
    """
    import numpy as np

    u = np.asarray(u, dtype=float)
    shape = cfg.u_a + cfg.u_b * ((1.0 - u) ** cfg.u_p + u ** cfg.u_p)
    # Analytic mean of the shape over [0, 1]: a + 2b/(p+1).
    mean = cfg.u_a + 2.0 * cfg.u_b / (cfg.u_p + 1.0)
    return shape / mean


__all__ = ["AppConfig", "MarketConfig", "BookConfig", "FlowConfig",
           "ImpactConfig", "ExecutionConfig", "SweepConfig", "FrontendConfig",
           "REPO_ROOT", "DEFAULT_CONFIG_PATH", "u_shape"]
