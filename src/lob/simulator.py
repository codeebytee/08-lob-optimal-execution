"""The venue: a limit order book driven by stochastic order flow.

Three intensities drive the book, in the spirit of Cont-Stoikov-Talreja (2010):

* **limit orders** arrive at distance ``k`` ticks from the reference price with
  intensity ``limit_k / k**limit_alpha`` lots per second,
* **market orders** arrive as a two-sided Hawkes process, so trades cluster,
* **cancellations** hit each resting lot at distance ``k`` with hazard
  ``cancel_theta / k**cancel_alpha`` per second, so far-from-touch depth is
  stickier than depth at the touch.

and one extra ingredient that pure flow models do not have:

* **a latent efficient price** ``S_t``, an arithmetic Brownian motion plus a
  Kyle-style permanent impact term ``lambda * (signed lots traded)``.

That last piece is the design decision worth defending. A book driven by
symmetric order flow alone has a *mean-reverting* mid: depth accumulates on
whichever side has been eaten, and the price is pulled back. Real prices are
close to martingales, and an execution study whose price mean-reverts will
report absurdly optimistic costs for slow schedules, because waiting is free
when the price comes back to you. Quoting around a latent random walk fixes
that at the cost of one extra assumption, and the assumption is testable: the
simulated mid must have the variance ratio and the realised volatility that the
calibration says it should. ``src/flow/calibrate.py`` checks exactly that.

Placement rule: buy limits go at ``floor(S/tick) - (k-1)``, sell limits at
``floor(S/tick) + 1 + (k-1)``. When ``S`` drifts through a tick, newly arriving
quotes on one side cross the stale quotes on the other and execute against them
- which is how the book follows the efficient price, and is also where a
passive resting order gets adversely selected. That mechanism is not bolted on;
it falls out of quoting around a moving reference.

Timing is exact, not discretised. Inter-event times come from Ogata thinning:
the limit and cancel intensities are constant between events, the Hawkes
intensity only decays, so the intensity immediately after the previous event
bounds the next interval. The intraday U-shape is applied piecewise-constantly
on a grid, and the loop steps to grid boundaries so that piecewise-constant is
literally true rather than approximately true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data.market import NameStats
from ..flow.hawkes import BivariateHawkes
from ..utils.config import BookConfig, FlowConfig, u_shape
from .book import BUY, SELL, Fill, OrderBook


@dataclass
class AgentFill:
    """One of the agent's own fills, in shares and dollars."""

    t: float
    price: float
    shares: int
    side: int
    passive: bool


@dataclass
class Snapshot:
    """One recorded frame of the book, for the animation."""

    t: float
    latent: float
    best_bid: float
    best_ask: float
    bid_lots: List[int]
    ask_lots: List[int]
    bid_px: List[float]
    ask_px: List[float]
    agent_done: int
    last_trade: Optional[float]


class MarketSimulator:
    """A single simulated trading session for one name.

    The execution layer drives it: :meth:`run_until` advances anonymous flow to
    a wall-clock time, and the agent methods inject the parent order's child
    orders in between.
    """

    def __init__(self, stats: NameStats, book_cfg: BookConfig,
                 flow_cfg: FlowConfig, rng: np.random.Generator,
                 sigma_exo_per_sec: Optional[float] = None,
                 kyle_lambda: Optional[float] = None,
                 start_fraction: float = 0.25,
                 seconds_per_day: float = 23400.0,
                 vol_multiplier: float = 1.0,
                 record_every: Optional[float] = None,
                 n_record_levels: int = 8,
                 exo_seed: Optional[int] = None):
        self.stats = stats
        self.bcfg = book_cfg
        self.fcfg = flow_cfg
        self.rng = rng
        self.tick = float(stats.tick_size)
        self.lot = int(book_cfg.lot_size)
        self.seconds_per_day = float(seconds_per_day)
        self.start_fraction = float(start_fraction)
        self.vol_multiplier = float(vol_multiplier)

        target_sigma = stats.sigma_per_second(seconds_per_day) * self.vol_multiplier
        self.sigma_exo = (target_sigma if sigma_exo_per_sec is None
                          else float(sigma_exo_per_sec) * self.vol_multiplier)
        self.kyle_lambda = float(flow_cfg.kyle_lambda if kyle_lambda is None
                                 else kyle_lambda)

        self.t = 0.0
        self.latent = float(stats.price)
        self.arrival_price = float(stats.price)

        self.book = OrderBook(tick_size=self.tick, lot_size=self.lot)
        b_ref = int(math.floor(self.latent / self.tick))
        for k in range(book_cfg.n_levels):
            lots = max(1, int(round(book_cfg.initial_depth_lots
                                    * math.exp(-book_cfg.depth_decay * k))))
            self.book.add_limit(BUY, b_ref - k, lots, ts=0.0)
            self.book.add_limit(SELL, b_ref + 1 + k, lots, ts=0.0)

        self.hawkes = BivariateHawkes(mu0=flow_cfg.market_rate,
                                      alpha=flow_cfg.hawkes_alpha if flow_cfg.hawkes_enabled else 0.0,
                                      beta=flow_cfg.hawkes_beta,
                                      cross=flow_cfg.hawkes_cross)
        self.hawkes.reset(0.0)

        # --- accounting -------------------------------------------------
        self.market_lots_traded = 0        # anonymous volume only
        self.anon_notional = 0.0           # for the market VWAP benchmark
        self.signed_lots = 0
        self.agent_fills: List[AgentFill] = []
        self.agent_shares_done = 0
        self.n_events = 0
        self.mid_path: List[Tuple[float, float]] = []
        self.trade_prints: List[Tuple[float, float, int]] = []
        self.last_trade_price: Optional[float] = None
        self._agent_orders: Dict[int, int] = {}     # oid -> lots submitted

        self._record_every = record_every
        self._n_rec = int(n_record_levels)
        self.snapshots: List[Snapshot] = []
        self._next_record = 0.0 if record_every else math.inf

        # Precompute the limit-order level distribution: p(k) ~ 1/k**alpha.
        ks = np.arange(1, book_cfg.n_levels + 1, dtype=float)
        w = flow_cfg.limit_k / ks ** flow_cfg.limit_alpha
        self._limit_rate_total = float(w.sum())
        self._limit_cdf = list(np.cumsum(w) / w.sum())
        self._cancel_w = list(flow_cfg.cancel_theta / ks ** flow_cfg.cancel_alpha)
        self._u_grid = 30.0                 # seconds; U-shape held constant here
        self._u_cell = -1
        self._u_value = 1.0
        self._uniforms = np.zeros(0)
        self._uniform_i = 0
        # The exogenous price path lives on its own generator and its own time
        # grid - see _exogenous() for why that matters.
        # Defaulting the seed to a draw from the flow generator gives the
        # property the paired experiment needs: two simulators built from the
        # same flow seed see the same price path, without the caller having to
        # remember to line them up.
        if exo_seed is None:
            exo_seed = int(rng.integers(0, 2 ** 62))
        self._exo_rng = np.random.default_rng(exo_seed)
        self.exo_seed = int(exo_seed)
        self._exo_dt = 0.25
        self._exo_path = np.zeros(0)
        self._exo_t_max = -1.0
        self._exo_last = 0.0
        self._pending_e: Optional[float] = None

    # -- helpers -----------------------------------------------------------

    def _multiplier(self, t: float) -> float:
        """Intraday activity multiplier, held constant on a 30-second grid.

        Cached per grid cell and evaluated with scalar arithmetic rather than
        through the array version in ``utils.config``: this is called once per
        loop iteration, hundreds of thousands of times per simulated session,
        and the numpy round trip was 5% of total runtime. The two are pinned to
        each other in ``tests/test_simulator.py``.
        """
        cell = int((self.start_fraction * self.seconds_per_day + t) // self._u_grid)
        if cell == self._u_cell:
            return self._u_value
        u = min(max(cell * self._u_grid / self.seconds_per_day, 0.0), 1.0)
        c = self.fcfg
        shape = c.u_a + c.u_b * ((1.0 - u) ** c.u_p + u ** c.u_p)
        mean = c.u_a + 2.0 * c.u_b / (c.u_p + 1.0)
        self._u_cell = cell
        self._u_value = shape / mean
        return self._u_value

    @property
    def mid(self) -> float:
        m = self.book.mid
        return self.latent if m is None else m

    def _draw_size(self, mean_lots: float) -> int:
        """Geometric order sizes: mostly one lot, occasionally very large.

        A fixed size would make queue position deterministic and the depth
        profile far too smooth. Geometric is the crude version of the empirical
        power law - right shape, one parameter, nothing to overfit. Inverted
        from a uniform rather than drawn from ``rng.geometric`` because this is
        inner-loop code; the distribution is identical.
        """
        p = 1.0 / max(mean_lots, 1.0)
        if p >= 1.0:
            return 1
        return int(math.log(self._uniform()) / math.log1p(-p)) + 1

    def _ref_ticks(self) -> Tuple[int, int]:
        b_ref = int(math.floor(self.latent / self.tick))
        return b_ref, b_ref + 1

    def _cancel_weights(self) -> Tuple[List[Tuple[int, int]], List[float], float]:
        """Per-level cancellation rates, measured from the *reference* price.

        Distance is taken from the efficient price rather than from the current
        best, and quotes further out than the quoted book get a much higher
        hazard ``stale_cancel_theta``. Both choices are load-bearing. A large
        cap like MSFT moves hundreds of ticks in half an hour while its quoted
        book is a handful of ticks deep, so quotes that the price has walked
        away from have to leave - by cancellation, since nobody will trade with
        them. Without this the book fills with stale liquidity, the measured
        spread drifts out to fifteen ticks, and every execution cost in the
        project is wrong.
        """
        b_ref, a_ref = self._ref_ticks()
        keys: List[Tuple[int, int]] = []
        weights: List[float] = []
        total = 0.0
        n = self.bcfg.n_levels
        for (side, price), qty in self.book._level_qty.items():
            if qty <= 0:
                continue
            d = (b_ref - price + 1) if side == BUY else (price - a_ref + 1)
            if d < 1:
                # Inside the reference price: this quote is about to be crossed
                # by the next arrival on the other side, and is the most
                # exposed of all. Treat it as stale.
                w = self.fcfg.stale_cancel_theta
            elif d <= n:
                w = self._cancel_w[d - 1]
            else:
                w = self.fcfg.stale_cancel_theta
            r = w * qty
            keys.append((side, price))
            weights.append(r)
            total += r
        return keys, weights, total

    def _cancel_rate(self) -> float:
        return self._cancel_weights()[2]

    # -- events ------------------------------------------------------------

    def _do_limit(self, t: float) -> None:
        side = BUY if self._uniform() < 0.5 else SELL
        u = self._uniform()
        k = 1
        for i, c in enumerate(self._limit_cdf):
            if u <= c:
                k = i + 1
                break
        else:
            k = len(self._limit_cdf)
        b_ref, a_ref = self._ref_ticks()
        price = b_ref - (k - 1) if side == BUY else a_ref + (k - 1)
        size = self._draw_size(self.fcfg.limit_size_mean_lots)
        if self.book.qty_at(side, price) > self.bcfg.max_queue_lots:
            return
        _, fills = self.book.add_limit(side, price, size, ts=t)
        if fills:
            self._on_trades(t, fills, aggressor_side=side)

    def _do_market(self, t: float, side: int) -> None:
        size = self._draw_size(self.fcfg.market_size_mean_lots)
        # Anonymous flow is not infinitely aggressive: cap the walk at
        # ``sweep_limit_ticks`` from the touch. Without a cap, one large market
        # order in a thin book prints a trade a dollar away and the volatility
        # of the session is set by that single event.
        best = self.book.best_ask if side == BUY else self.book.best_bid
        if best is None:
            return
        cap = best + self.fcfg.sweep_limit_ticks * side
        fills = self.book.submit_market(side, size, ts=t, limit_ticks=cap)
        if fills:
            self._on_trades(t, fills, aggressor_side=side)
            self.hawkes.excite(t, side)

    def _do_cancel(self, keys, weights, total: float) -> None:
        if not keys or total <= 0:
            return
        target = self._uniform() * total
        acc = 0.0
        side, price = keys[-1]
        for (s, p), w in zip(keys, weights):
            acc += w
            if acc >= target:
                side, price = s, p
                break
        self.book.cancel_random(side, price, 1, self._uniform())

    def _uniform(self) -> float:
        """One uniform draw, taken from a buffer.

        Drawing 4096 at a time and handing them out is roughly four times
        cheaper per draw than calling the generator each time, and the inner
        loop takes three or four draws per event. The stream is identical
        either way, so seeds still reproduce runs exactly.
        """
        if self._uniform_i >= self._uniforms.size:
            self._uniforms = self.rng.random(4096)
            self._uniform_i = 0
        v = self._uniforms[self._uniform_i]
        self._uniform_i += 1
        return float(v)

    def _on_trades(self, t: float, fills: List[Fill], aggressor_side: int) -> None:
        lots = 0
        for f in fills:
            px = self.book.to_price(f.price)
            self.last_trade_price = px
            self.trade_prints.append((t, px, f.qty * f.aggressor_side))
            if f.passive_agent:
                self.agent_fills.append(AgentFill(t=t, price=px,
                                                  shares=f.qty * self.lot,
                                                  side=-f.aggressor_side,
                                                  passive=True))
                self.agent_shares_done += f.qty * self.lot
            else:
                lots += f.qty
                self.anon_notional += px * f.qty * self.lot
        if lots:
            self.market_lots_traded += lots
            self.signed_lots += lots * aggressor_side
        # Permanent impact: every trade, anonymous or ours, moves the efficient
        # price. Applying it to the agent's fills too is the point - it is what
        # makes trading fast expensive in a way that waiting cannot undo.
        traded = sum(f.qty for f in fills)
        if traded and self.kyle_lambda:
            self.latent += self.kyle_lambda * aggressor_side * traded

    # -- time --------------------------------------------------------------

    def _exogenous(self, t: float) -> float:
        """The exogenous component of the efficient price at time ``t``.

        Drawn on a fixed time grid from a *dedicated* generator and
        interpolated, rather than accumulated event by event. That is the
        difference between a paired experiment that works and one that does
        not.

        The measurement this project depends on - how much did the parent order
        move the price - is the difference between two runs of the same session,
        one with the order and one without. Over half an hour the price moves
        by tens of times the impact being measured, so the two runs have to see
        the *same* price path or the impact is invisible under path noise. If
        the path is built from draws taken at event times, adding the parent
        order changes the event stream, the draws desynchronise, and the pairing
        buys nothing. Indexing the path by wall-clock time instead makes it
        identical across any two runs with the same seed, whatever the agent
        does.

        The grid is fine enough (a quarter of a second) that linear
        interpolation inside a cell is invisible next to the tick size, and the
        path is extended in blocks so a session never allocates more than it
        uses.
        """
        if self.sigma_exo <= 0:
            return 0.0
        while t > self._exo_t_max:
            self._extend_exo()
        i = int(t / self._exo_dt)
        frac = (t - i * self._exo_dt) / self._exo_dt
        a = self._exo_path[i]
        b = self._exo_path[i + 1]
        return float(a + frac * (b - a))

    def _extend_exo(self) -> None:
        block = 4096
        step = self.sigma_exo * math.sqrt(self._exo_dt)
        incr = self._exo_rng.normal(size=block) * step
        tail = self._exo_path[-1] if self._exo_path.size else 0.0
        new = tail + np.cumsum(incr)
        self._exo_path = (np.concatenate([self._exo_path, new])
                          if self._exo_path.size else
                          np.concatenate([[0.0], new]))
        self._exo_t_max = (self._exo_path.size - 2) * self._exo_dt

    def _advance_price(self, dt: float) -> None:
        """Move the clock. The exogenous part is a lookup, so all this does is
        keep ``latent`` equal to ``start + exogenous(t) + accumulated impact``.
        """
        if dt <= 0:
            return
        new_exo = self._exogenous(self.t + dt)
        self.latent += new_exo - self._exo_last
        self._exo_last = new_exo

    def _maybe_record(self, t: float) -> None:
        while t >= self._next_record:
            self._record(self._next_record)
            self._next_record += self._record_every

    def _record(self, t: float) -> None:
        bt, bq, at, aq = self.book.depth_profile(self._n_rec)
        self.snapshots.append(Snapshot(
            t=t, latent=self.latent,
            best_bid=float(bt[0]) * self.tick, best_ask=float(at[0]) * self.tick,
            bid_lots=[int(x) for x in bq], ask_lots=[int(x) for x in aq],
            bid_px=[float(x) * self.tick for x in bt],
            ask_px=[float(x) * self.tick for x in at],
            agent_done=int(self.agent_shares_done),
            last_trade=self.last_trade_price))

    def run_until(self, t_end: float) -> None:
        """Advance anonymous order flow to ``t_end``."""
        while self.t < t_end:
            m = self._multiplier(self.t)
            # Step no further than the next U-shape boundary, so the
            # piecewise-constant intensity assumption is exact.
            grid_next = (math.floor(self.t / self._u_grid) + 1) * self._u_grid
            horizon = min(t_end, grid_next)

            rate_limit = m * self._limit_rate_total * 2.0
            keys, weights, cancel_total = self._cancel_weights()
            rate_cancel = m * cancel_total
            lam_b, lam_s = self.hawkes.intensities(self.t)
            bound = rate_limit + rate_cancel + m * (lam_b + lam_s)
            if bound <= 0:
                self._advance_price(horizon - self.t)
                self.t = horizon
                self._maybe_record(self.t)
                continue

            # The unused part of an exponential wait is carried across calls
            # rather than thrown away. Discarding it and redrawing would be
            # statistically harmless - exponential waits are memoryless - but
            # it makes the event stream depend on *how the caller chopped up
            # time*, and this simulator is driven with different chopping by
            # the baseline run (slice boundaries) and by the execution run
            # (chunk boundaries within each slice). Carrying the residual in
            # units of the standard exponential, so that it stays valid when
            # the rate changes, makes run_until a pure function of its end
            # time: two runs stay in lockstep for exactly as long as the agent
            # has not done anything.
            e = self._pending_e if self._pending_e is not None else -math.log(self._uniform())
            self._pending_e = None
            t_new = self.t + e / bound

            if t_new >= horizon:
                used = (horizon - self.t) * bound
                self._pending_e = max(e - used, 0.0)
                self._advance_price(horizon - self.t)
                self.t = horizon
                self._maybe_record(self.t)
                continue

            self._advance_price(t_new - self.t)
            self.t = t_new
            self._maybe_record(self.t)

            lam_b, lam_s = self.hawkes.intensities(self.t)
            r_mkt_b, r_mkt_s = m * lam_b, m * lam_s
            actual = rate_limit + rate_cancel + r_mkt_b + r_mkt_s
            u = self._uniform() * bound
            if u >= actual:
                continue                      # thinning rejection
            self.n_events += 1
            if u < rate_limit:
                self._do_limit(self.t)
            elif u < rate_limit + rate_cancel:
                self._do_cancel(keys, weights, cancel_total)
            elif u < rate_limit + rate_cancel + r_mkt_b:
                self._do_market(self.t, BUY)
            else:
                self._do_market(self.t, SELL)

    # -- agent -------------------------------------------------------------

    def agent_market(self, side: int, shares: int,
                     max_ticks_through: Optional[int] = None) -> List[AgentFill]:
        """Send an aggressive child order for ``shares`` shares.

        Returns only this order's fills. Odd share counts below one lot are
        rounded down, which is the honest behaviour: a lot-based book cannot
        express a 37-share child order, and silently rounding *up* would fill
        more than the parent.
        """
        lots = int(shares // self.lot)
        if lots <= 0:
            return []
        best = self.book.best_ask if side == BUY else self.book.best_bid
        if best is None:
            return []
        cap = None if max_ticks_through is None else best + max_ticks_through * side
        fills = self.book.submit_market(side, lots, ts=self.t, agent=True,
                                        limit_ticks=cap)
        out: List[AgentFill] = []
        for f in fills:
            px = self.book.to_price(f.price)
            af = AgentFill(t=self.t, price=px, shares=f.qty * self.lot,
                           side=side, passive=False)
            out.append(af)
            self.agent_fills.append(af)
            self.agent_shares_done += af.shares
            self.last_trade_price = px
            self.trade_prints.append((self.t, px, f.qty * side))
        traded = sum(f.qty for f in fills)
        if traded:
            self.hawkes.excite(self.t, side)
            if self.kyle_lambda:
                self.latent += self.kyle_lambda * side * traded
        return out

    def agent_limit(self, side: int, offset_ticks: int, shares: int) -> int:
        """Rest a child order ``offset_ticks`` inside/behind the touch.

        ``offset_ticks = 0`` joins the near touch, positive numbers step away
        from it (more passive). Returns the order id, or 0 if it filled
        immediately.
        """
        lots = int(shares // self.lot)
        if lots <= 0:
            return 0
        bb, ba = self.book.best_bid, self.book.best_ask
        if bb is None or ba is None:
            return 0
        price = bb - offset_ticks if side == BUY else ba + offset_ticks
        oid, fills = self.book.add_limit(side, price, lots, ts=self.t, agent=True)
        if fills:
            for f in fills:
                px = self.book.to_price(f.price)
                af = AgentFill(t=self.t, price=px, shares=f.qty * self.lot,
                               side=side, passive=False)
                self.agent_fills.append(af)
                self.agent_shares_done += af.shares
            if self.kyle_lambda:
                self.latent += self.kyle_lambda * side * sum(f.qty for f in fills)
        if oid:
            self._agent_orders[oid] = lots
        return oid

    def cancel_agent_orders(self) -> int:
        """Pull every resting agent order. Returns lots cancelled."""
        n = 0
        for oid in list(self._agent_orders):
            n += self.book.cancel(oid)
            self._agent_orders.pop(oid, None)
        return n

    def agent_open_shares(self) -> int:
        return sum(self.book._orders[oid].qty * self.lot
                   for oid in self._agent_orders
                   if oid in self.book._orders) if self._agent_orders else 0


__all__ = ["MarketSimulator", "AgentFill", "Snapshot"]
