"""Work a parent order through the simulated book and measure what it cost.

This module is where the two halves of the project meet. An algorithm from
``schedules.py`` decides how many shares to send each slice; the simulator in
``lob/simulator.py`` decides what those shares actually fill at. Neither knows
about the other, which is what makes the comparison meaningful: the algorithms
were not written against this venue's quirks, and the venue does not know which
algorithm is trading.

**Implementation shortfall** is the metric, defined the way Perold (1988)
defined it - the difference between what was paid and the mid at the moment the
decision was made:

    IS = side * (average execution price - arrival mid) * shares filled
       + side * (final mid - arrival mid) * shares not filled

The second term is the one that gets left out. An algorithm that fails to
complete has not saved money; it has swapped execution cost for opportunity
cost, and it must be charged for the shares it did not buy at wherever the
price ended up. Without that term, "trade less when it is expensive" looks like
free alpha and POV in a quiet market looks like a genius.

**Timing convention.** Each slice's child order is sent at the slice boundary,
then anonymous flow runs for the rest of the slice. A child order therefore
never trades against volume that has not happened yet, and the volume an
algorithm reacts to is always strictly in the past. That look-ahead guard is
enforced structurally by this loop rather than by a comment in a notebook.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..data.market import NameStats
from ..lob.book import BUY
from ..lob.simulator import MarketSimulator, Snapshot
from ..utils.config import BookConfig, FlowConfig
from .schedules import Algo, ExecState


@dataclass
class SliceRecord:
    """One slice of the parent order, for the trajectory chart."""

    k: int
    t_start: float
    target_shares: float
    filled_shares: int
    avg_price: float
    mid_start: float
    mid_end: float
    market_shares: int
    remaining: float


@dataclass
class BaselinePath:
    """The session that would have happened if the parent order had not.

    Used two ways. As the counterfactual in the impact calibration, and as a
    control variate for the cost estimates - see :func:`control_variate`.
    """

    arrival: float
    mids: List[float]           # one per slice boundary, excluding t=0
    final_mid: float
    seed: int


@dataclass
class ExecResult:
    """Everything one parent order execution produced."""

    algo: str
    side: int
    target_shares: int
    filled_shares: int
    arrival_mid: float
    final_mid: float
    avg_price: float
    shortfall_usd: float
    shortfall_bps: float
    opportunity_usd: float
    market_vwap: float
    vs_vwap_bps: float
    participation: float
    n_child_orders: int
    passive_fill_frac: float
    forced_shares: float
    cv_bps: float = float("nan")
    shortfall_adj_bps: float = float("nan")
    slices: List[SliceRecord] = field(default_factory=list)
    snapshots: List[Snapshot] = field(default_factory=list)
    mid_path: List[float] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.filled_shares >= self.target_shares


def _round_lots(shares: float, lot: int) -> int:
    return int(max(0, math.floor(shares / lot)) * lot)


def run_parent(stats: NameStats, book_cfg: BookConfig, flow_cfg: FlowConfig,
               algo: Algo, X_shares: int, seed: int,
               horizon: float, n_slices: int, side: int = BUY,
               kyle_lambda: float = 0.0,
               sigma_exo: Optional[float] = None,
               vol_multiplier: float = 1.0,
               start_fraction: float = 0.25,
               seconds_per_day: float = 23400.0,
               child_type: str = "market",
               limit_offset_ticks: int = 0,
               passive_fraction: float = 0.8,
               record_every: Optional[float] = None,
               max_ticks_through: Optional[int] = 8,
               chunks_per_slice: int = 6) -> ExecResult:
    """Execute ``X_shares`` with ``algo`` on one simulated path.

    ``seed`` fixes the anonymous flow. Running two algorithms on the same seed
    is deliberate - it is the common-random-numbers trick, and it removes most
    of the path noise from an algorithm comparison. Without it, separating a
    2 bp difference between two algorithms from luck takes an order of
    magnitude more paths.
    """
    rng = np.random.default_rng(seed)
    sim = MarketSimulator(stats, book_cfg, flow_cfg, rng,
                          sigma_exo_per_sec=sigma_exo,
                          kyle_lambda=kyle_lambda,
                          start_fraction=start_fraction,
                          seconds_per_day=seconds_per_day,
                          vol_multiplier=vol_multiplier,
                          record_every=record_every)
    lot = book_cfg.lot_size
    tau = horizon / n_slices
    arrival = sim.mid
    algo.reset(float(X_shares), n_slices, horizon)

    remaining = float(X_shares)
    filled = 0
    notional = 0.0
    passive_shares = 0
    n_children = 0
    last_slice_volume = 0.0
    fill_cursor = 0
    slices: List[SliceRecord] = []
    mid_path: List[float] = [arrival]
    market_shares_total = 0

    for k in range(n_slices):
        t_start = k * tau
        mid_start = sim.mid
        st = ExecState(k=k, n_slices=n_slices, remaining=remaining,
                       elapsed=t_start, horizon=horizon, mid=mid_start,
                       arrival=arrival, last_slice_volume=last_slice_volume,
                       side=side)
        want = float(min(max(algo.child_shares(st), 0.0), remaining))
        # The last slice has to finish, so it takes the whole remaining lot
        # count; every other slice rounds down and rolls the remainder forward.
        child = (_round_lots(remaining, lot) if k == n_slices - 1
                 else _round_lots(want, lot))

        before_lots = sim.market_lots_traded

        if child > 0:
            n_children += 1
            if child_type == "market":
                # A slice is not sent as one order. The displayed book holds a
                # few hundred shares at the touch, so a whole slice fired at
                # once would sweep ten levels, pay the tail of the book and -
                # worse - stop when it runs out of quotes, leaving the parent
                # unfilled. Real algorithms cut the slice into child orders
                # spaced through the interval so that liquidity has time to
                # replenish between them. ``chunks_per_slice`` is that spacing,
                # and the unfilled remainder of each chunk rolls into the next.
                per = tau / chunks_per_slice
                pending = child
                for c in range(chunks_per_slice):
                    if pending >= lot:
                        want_chunk = (pending if c == chunks_per_slice - 1
                                      else _round_lots(pending / (chunks_per_slice - c), lot))
                        if want_chunk >= lot:
                            got = sim.agent_market(side, want_chunk,
                                                   max_ticks_through=max_ticks_through)
                            pending -= sum(f.shares for f in got)
                    sim.run_until(min(t_start + (c + 1) * per, t_start + tau))
            elif child_type == "limit_then_market":
                # Rest inside the slice, then clean up aggressively. This is
                # the cheap version of a passive/aggressive algo, and it is
                # enough to show the tradeoff: the resting order earns the
                # spread when it fills, and gets adversely selected when the
                # price runs away from it.
                before = sim.agent_shares_done
                sim.agent_limit(side, limit_offset_ticks, child)
                sim.run_until(t_start + passive_fraction * tau)
                done_passively = sim.agent_shares_done - before
                sim.cancel_agent_orders()
                leftover = child - done_passively
                if leftover >= lot:
                    sim.agent_market(side, leftover,
                                     max_ticks_through=max_ticks_through)
                passive_shares += done_passively
            else:
                raise ValueError(f"unknown child_type {child_type!r}")

        sim.run_until(t_start + tau)

        slice_filled = 0
        slice_notional = 0.0
        for f in sim.agent_fills[fill_cursor:]:
            slice_filled += f.shares
            slice_notional += f.price * f.shares
        fill_cursor = len(sim.agent_fills)

        filled += slice_filled
        notional += slice_notional
        remaining = max(0.0, float(X_shares) - filled)

        market_shares = (sim.market_lots_traded - before_lots) * lot
        market_shares_total += market_shares
        last_slice_volume = float(market_shares)
        mid_end = sim.mid
        mid_path.append(mid_end)
        slices.append(SliceRecord(
            k=k, t_start=t_start, target_shares=want,
            filled_shares=slice_filled,
            avg_price=(slice_notional / slice_filled) if slice_filled
            else float("nan"),
            mid_start=mid_start, mid_end=mid_end,
            market_shares=market_shares, remaining=remaining))

    final_mid = sim.mid
    avg_price = notional / filled if filled else float("nan")
    exec_cost = side * (avg_price - arrival) * filled if filled else 0.0
    not_done = float(X_shares) - filled
    opportunity = side * (final_mid - arrival) * not_done
    shortfall = exec_cost + opportunity
    shortfall_bps = (1e4 * shortfall / (X_shares * arrival)
                     if X_shares else float("nan"))

    mkt_vwap = (sim.anon_notional / (sim.market_lots_traded * lot)
                if sim.market_lots_traded else float("nan"))
    vs_vwap = (1e4 * side * (avg_price - mkt_vwap) / mkt_vwap
               if (filled and np.isfinite(mkt_vwap) and mkt_vwap > 0)
               else float("nan"))
    participation = (filled / (filled + market_shares_total)
                     if (filled + market_shares_total) else float("nan"))

    return ExecResult(
        algo=algo.name, side=side, target_shares=int(X_shares),
        filled_shares=int(filled), arrival_mid=arrival, final_mid=final_mid,
        avg_price=avg_price, shortfall_usd=shortfall,
        shortfall_bps=shortfall_bps, opportunity_usd=opportunity,
        market_vwap=mkt_vwap, vs_vwap_bps=vs_vwap, participation=participation,
        n_child_orders=n_children,
        passive_fill_frac=(passive_shares / filled if filled else 0.0),
        forced_shares=float(getattr(algo, "forced_shares", 0.0)),
        slices=slices, snapshots=sim.snapshots, mid_path=mid_path)


def baseline_path(stats: NameStats, book_cfg: BookConfig, flow_cfg: FlowConfig,
                  seed: int, horizon: float, n_slices: int,
                  kyle_lambda: float = 0.0, sigma_exo: Optional[float] = None,
                  vol_multiplier: float = 1.0, start_fraction: float = 0.25,
                  seconds_per_day: float = 23400.0) -> BaselinePath:
    """Run the same session with no parent order in it.

    Because the exogenous price path is indexed by time rather than by event
    (see ``MarketSimulator._exogenous``), this run and any execution run on the
    same seed see the same price. That is what makes it a counterfactual rather
    than just another path.
    """
    rng = np.random.default_rng(seed)
    sim = MarketSimulator(stats, book_cfg, flow_cfg, rng,
                          sigma_exo_per_sec=sigma_exo, kyle_lambda=kyle_lambda,
                          start_fraction=start_fraction,
                          seconds_per_day=seconds_per_day,
                          vol_multiplier=vol_multiplier)
    arrival = sim.mid
    tau = horizon / n_slices
    mids: List[float] = []
    for k in range(n_slices):
        sim.run_until((k + 1) * tau)
        mids.append(sim.mid)
    return BaselinePath(arrival=arrival, mids=mids, final_mid=sim.mid,
                        seed=int(seed))


def control_variate(base: BaselinePath, side: int = BUY) -> float:
    """The counterfactual cost of a flat schedule, in bp - mean zero, and
    strongly correlated with every algorithm's realised cost.

    Almost all of the variance in a single execution's shortfall is the price
    path, not the algorithm: over half an hour a large cap moves tens of basis
    points while the entire cost being measured is a few. Subtracting a
    quantity with mean zero and correlation ~0.95 to that noise cuts the
    standard error by an order of magnitude, which is the difference between
    "TWAP and AC differ by 1.4 bp, t = 0.3" and a usable answer.

    Two properties make this legitimate rather than a thumb on the scale:

    * The weights are flat and fixed in advance, identical for every
      algorithm. They are *not* each algorithm's own fill weights - using those
      would subtract exactly the timing skill that the adaptive algorithm is
      trying to demonstrate.
    * The expectation is zero by construction, since the baseline mid is a
      martingale. The adjustment therefore changes the estimator's variance and
      not its mean.

    The raw shortfall, not this, is what the risk statistics and the histograms
    on the page are computed from: an execution desk's cost variance genuinely
    does include the market's move, and removing it would understate the tail
    the risk-aversion parameter exists to price.
    """
    if not base.mids or base.arrival <= 0:
        return float("nan")
    mean_mid = float(np.mean(base.mids))
    return 1e4 * side * (mean_mid - base.arrival) / base.arrival


def apply_control_variate(res: ExecResult, base: BaselinePath) -> ExecResult:
    """Attach the control variate and the adjusted shortfall to a result."""
    cv = control_variate(base, res.side)
    res.cv_bps = cv
    res.shortfall_adj_bps = (res.shortfall_bps - cv
                             if np.isfinite(cv) else res.shortfall_bps)
    return res


__all__ = ["run_parent", "ExecResult", "SliceRecord", "BaselinePath",
           "baseline_path", "control_variate", "apply_control_variate"]
