"""Calibrate the venue to a real name, and the cost model to the venue.

Three separate jobs, in the order they have to happen.

**1. Flow rates -> ADV.** All three intensities are multiplied by one scale
factor chosen so the simulated session prints the name's median daily volume.
Uniform scaling is the right knob because it is a pure time change for the
queueing system: multiply every arrival, cancellation and trade rate by ``c``
and the book's stationary shape is unchanged while volume scales exactly by
``c``. That makes the calibration a one-dimensional fixed point rather than a
search over three parameters, and it is why the depth profile does not have to
be re-tuned per name. It is not a free lunch - the latent price's volatility is
*not* rescaled, so what actually changes across names is the ratio of trading
speed to price speed, which is the ratio that matters for execution anyway.

**2. Permanent impact -> Kyle's lambda.** The starting value comes from the
oldest dimensional argument in microstructure: if a day's volume moves the
price by about a day's volatility, then ``lambda ~ sigma_daily * P / ADV``. It
is then *verified inside the simulator* by paired counterfactual runs, which is
the part that makes it a measurement rather than an assumption.

**3. The Almgren-Chriss parameters -> the simulator.** ``gamma`` and ``eta``
are estimated from the venue by executing parent orders and measuring what
happened, not by assuming what the venue was built with. The identification
uses paired runs on common random numbers: for each seed the session is run
twice, once with the parent order and once without. The difference in the final
mid is the permanent impact, cleanly separated from the path the price would
have taken anyway:

    gamma = E[ side * (mid_end_with - mid_end_without) ] / X

Without pairing, this estimate is buried under a price path whose standard
deviation over half an hour is ten to a hundred times the impact being
measured, and no plausible number of paths recovers it. With pairing, a few
dozen paths is enough. The temporary parameter then falls out of the shortfall
identity for a TWAP schedule,

    E[IS]/X = epsilon + gamma X / 2 + eta v ,   v = X / T

by subtracting the two terms already known.

Because a TWAP parent has ``v = X/T``, size and rate are collinear, and no
regression of shortfall alone can separate ``gamma`` from ``eta``. Papers that
report both from a single such regression have not separated them; they have
reported one number twice. The counterfactual is what breaks the collinearity.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.market import NameStats
from ..execution.runner import baseline_path, control_variate, run_parent
from ..execution.schedules import TWAP
from ..lob.book import BUY
from ..lob.simulator import MarketSimulator
from ..utils.config import BookConfig, FlowConfig, ImpactConfig


@dataclass(frozen=True)
class VenueCalibration:
    """Everything the rest of the project needs about one calibrated name."""

    ticker: str
    flow: FlowConfig
    rate_scale: float
    kyle_lambda: float          # $ per lot of signed volume
    sigma_exo: float            # $ per sqrt(second), exogenous component
    sim_adv_shares: float
    sim_sigma_per_sec: float
    target_sigma_per_sec: float
    median_spread_ticks: float
    mean_touch_lots: float

    def to_dict(self) -> Dict[str, float]:
        return {"ticker": self.ticker, "rate_scale": self.rate_scale,
                "kyle_lambda": self.kyle_lambda, "sigma_exo": self.sigma_exo,
                "sim_adv_shares": self.sim_adv_shares,
                "sim_sigma_per_sec": self.sim_sigma_per_sec,
                "target_sigma_per_sec": self.target_sigma_per_sec,
                "median_spread_ticks": self.median_spread_ticks,
                "mean_touch_lots": self.mean_touch_lots}


@dataclass(frozen=True)
class ImpactCalibration:
    """Fitted Almgren-Chriss parameters plus the evidence behind them."""

    ticker: str
    gamma: float                # $ per share per share
    eta: float                  # $ per share per (share/second)
    epsilon: float              # $ per share
    exponent: float             # fitted power on the rate; AC assumes 1
    exponent_r2: float
    linear_r2: float
    rows: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, object]:
        return {"ticker": self.ticker, "gamma": self.gamma, "eta": self.eta,
                "epsilon": self.epsilon, "exponent": self.exponent,
                "exponent_r2": self.exponent_r2, "linear_r2": self.linear_r2,
                "rows": self.rows}


def scale_flow(flow: FlowConfig, c: float) -> FlowConfig:
    """Multiply every rate by ``c``. See the module docstring for why this is
    the only scaling that leaves the book's shape alone."""
    return dataclasses.replace(
        flow,
        limit_k=flow.limit_k * c,
        market_rate=flow.market_rate * c,
        cancel_theta=flow.cancel_theta * c,
        stale_cancel_theta=flow.stale_cancel_theta * c,
        hawkes_alpha=flow.hawkes_alpha * c,
        hawkes_beta=flow.hawkes_beta * c,
    )


def _session_stats(stats: NameStats, book: BookConfig, flow: FlowConfig,
                   seed: int, T: float, kyle_lambda: float,
                   sigma_exo: Optional[float],
                   seconds_per_day: float) -> Dict[str, float]:
    sim = MarketSimulator(stats, book, flow, np.random.default_rng(seed),
                          sigma_exo_per_sec=sigma_exo,
                          kyle_lambda=kyle_lambda,
                          seconds_per_day=seconds_per_day,
                          record_every=5.0)
    sim.run_until(T)
    mids = np.array([0.5 * (s.best_bid + s.best_ask) for s in sim.snapshots])
    spreads = np.array([(s.best_ask - s.best_bid) / stats.tick_size
                        for s in sim.snapshots])
    touch = np.array([0.5 * (s.bid_lots[0] + s.ask_lots[0])
                      for s in sim.snapshots])
    dm = np.diff(mids)
    return {"adv": sim.market_lots_traded * book.lot_size * seconds_per_day / T,
            "sigma": float(np.std(dm) / math.sqrt(5.0)) if dm.size else 0.0,
            "spread": float(np.median(spreads)),
            "touch": float(np.mean(touch))}


def kyle_lambda_guess(stats: NameStats, lot_size: int, c: float = 1.0) -> float:
    """``lambda ~ c * sigma_daily_$ / ADV``, expressed per lot.

    The reading: buying a full day's volume moves the price by about ``c``
    daily standard deviations. ``c = 1`` is the textbook version and lands
    within a factor of two of the impact literature's estimates for large caps,
    which is as much precision as a free-data project can honestly claim.
    """
    sigma_daily_usd = stats.price * stats.sigma_daily
    return float(c * sigma_daily_usd / stats.adv_shares * lot_size)


def calibrate_venue(stats: NameStats, book: BookConfig, flow: FlowConfig,
                    seed: int = 11, T: float = 900.0, iterations: int = 2,
                    seconds_per_day: float = 23400.0,
                    kyle_c: float = 1.0) -> VenueCalibration:
    """Scale the flow to the name's ADV and solve for the exogenous vol.

    The volatility step is subtle enough to spell out. The mid moves for two
    reasons: the exogenous latent random walk, and the Kyle term, through which
    random order-flow imbalance also moves the price. Those add in variance, so
    setting the exogenous component equal to the target would leave the
    simulated session more volatile than the real name. The fix is to measure
    the flow-induced variance with the exogenous term switched off, then set

        sigma_exo^2 = max(sigma_target^2 - sigma_flow^2, 0)

    If the flow term alone already exceeds the target, ``sigma_exo`` is zero and
    the calibration says so rather than quietly taking a square root of a
    negative number.
    """
    kyle = kyle_lambda_guess(stats, book.lot_size, kyle_c)
    target_sigma = stats.sigma_per_second(seconds_per_day)

    c = 1.0
    meas = _session_stats(stats, book, flow, seed, T, kyle, target_sigma,
                          seconds_per_day)
    for _ in range(iterations):
        if meas["adv"] <= 0:
            break
        c *= stats.adv_shares / meas["adv"]
        meas = _session_stats(stats, book, scale_flow(flow, c), seed, T, kyle,
                              target_sigma, seconds_per_day)

    scaled = scale_flow(flow, c)
    # Flow-only volatility: same everything, exogenous term switched off.
    flow_only = _session_stats(stats, book, scaled, seed + 1, T, kyle, 0.0,
                               seconds_per_day)
    sigma_flow = flow_only["sigma"]
    sigma_exo = math.sqrt(max(target_sigma ** 2 - sigma_flow ** 2, 0.0))

    final = _session_stats(stats, book, scaled, seed + 2, T, kyle, sigma_exo,
                           seconds_per_day)
    return VenueCalibration(
        ticker=stats.ticker, flow=dataclasses.replace(scaled, kyle_lambda=kyle),
        rate_scale=c, kyle_lambda=kyle, sigma_exo=sigma_exo,
        sim_adv_shares=final["adv"], sim_sigma_per_sec=final["sigma"],
        target_sigma_per_sec=target_sigma,
        median_spread_ticks=final["spread"], mean_touch_lots=final["touch"])


def measure_impact(stats: NameStats, book: BookConfig, cal: VenueCalibration,
                   participations: Sequence[float], paths: int,
                   horizon: float, n_slices: int, base_seed: int = 5000,
                   seconds_per_day: float = 23400.0,
                   side: int = BUY) -> ImpactCalibration:
    """Estimate ``gamma``, ``eta`` and the impact exponent from the venue.

    For each target participation rate, a TWAP parent sized to that fraction of
    expected volume over the horizon is run against ``paths`` seeds, and each
    seed is *also* run with no parent order at all. Everything is measured as a
    difference between the pair.
    """
    lot = book.lot_size
    expected_volume = cal.sim_adv_shares * horizon / seconds_per_day
    rows: List[Dict[str, float]] = []

    for rho in participations:
        X = int(round(max(rho * expected_volume, lot) / lot) * lot)
        perm, per_share, done = [], [], []
        for i in range(paths):
            seed = base_seed + i
            res = run_parent(stats, book, cal.flow, TWAP(), X, seed, horizon,
                             n_slices, side=side, kyle_lambda=cal.kyle_lambda,
                             sigma_exo=cal.sigma_exo,
                             seconds_per_day=seconds_per_day)
            base = baseline_path(stats, book, cal.flow, seed, horizon, n_slices,
                                 kyle_lambda=cal.kyle_lambda,
                                 sigma_exo=cal.sigma_exo,
                                 seconds_per_day=seconds_per_day)
            perm.append(side * (res.final_mid - base.final_mid))
            # Control variate: the same session without the order would have
            # cost this much on a flat schedule, and it has mean zero.
            cv_usd = control_variate(base, side) * 1e-4 * base.arrival
            per_share.append(res.shortfall_usd / X - cv_usd)
            done.append(res.filled_shares / X)
        v = X / horizon
        rows.append({"participation": float(rho), "X": float(X),
                     "rate": float(v),
                     "perm_usd": float(np.mean(perm)),
                     "perm_se": float(np.std(perm, ddof=1) / math.sqrt(paths)),
                     "cost_per_share": float(np.mean(per_share)),
                     "cost_se": float(np.std(per_share, ddof=1) / math.sqrt(paths)),
                     "fill_rate": float(np.mean(done))})

    X_arr = np.array([r["X"] for r in rows])
    perm_arr = np.array([r["perm_usd"] for r in rows])
    rate_arr = np.array([r["rate"] for r in rows])
    cps_arr = np.array([r["cost_per_share"] for r in rows])

    # gamma: permanent impact is linear in size through the origin, so the
    # least-squares slope with no intercept is the estimator.
    gamma = float(np.sum(perm_arr * X_arr) / np.sum(X_arr ** 2))

    # epsilon: the model's fixed cost is the half spread plus fees, and it is
    # the intercept of the per-share cost as the rate goes to zero. Taken from
    # the venue's own measured spread rather than fitted, so that the fit has
    # one free parameter and not two.
    epsilon = 0.5 * cal.median_spread_ticks * stats.tick_size

    resid = cps_arr - epsilon - 0.5 * gamma * X_arr
    eta = float(np.sum(resid * rate_arr) / np.sum(rate_arr ** 2))
    pred = epsilon + 0.5 * gamma * X_arr + eta * rate_arr
    ss_res = float(np.sum((cps_arr - pred) ** 2))
    ss_tot = float(np.sum((cps_arr - cps_arr.mean()) ** 2))
    linear_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # The exponent: is temporary impact really linear in the rate? Fit
    # resid = a * rate**b in logs. AC assumes b = 1; the square-root law says
    # b = 1/2. The answer for this venue is in results/impact_calibration.csv.
    ok = (resid > 0) & (rate_arr > 0)
    if ok.sum() >= 3:
        lx = np.log(rate_arr[ok])
        ly = np.log(resid[ok])
        b, a = np.polyfit(lx, ly, 1)
        fit = a + b * lx
        exp_r2 = 1.0 - float(np.sum((ly - fit) ** 2)) / float(
            np.sum((ly - ly.mean()) ** 2))
        exponent, exponent_r2 = float(b), float(exp_r2)
    else:
        exponent, exponent_r2 = float("nan"), float("nan")

    return ImpactCalibration(ticker=stats.ticker, gamma=gamma, eta=eta,
                             epsilon=epsilon, exponent=exponent,
                             exponent_r2=exponent_r2, linear_r2=linear_r2,
                             rows=rows)


__all__ = ["VenueCalibration", "ImpactCalibration", "scale_flow",
           "kyle_lambda_guess", "calibrate_venue", "measure_impact"]
