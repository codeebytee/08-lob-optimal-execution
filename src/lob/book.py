"""A price-time-priority limit order book, at the level of individual orders.

This is the piece that most "LOB projects" fake by keeping a depth *profile* -
an array of sizes per price - and treating a market order as an arithmetic
subtraction. That version cannot answer the question an execution trader
actually asks, which is *where is my order in the queue and will it fill*. So
this book stores real orders in real FIFO queues, and the agent's own orders sit
in those queues behind whatever arrived before them.

Design notes that matter:

**Integer ticks.** Prices are integers (ticks), never floats. Floating-point
prices in a matching engine produce orders that fail to match because
``1.15 - 0.01 != 1.14``. Conversion to dollars happens exactly once, at the
edge, in :meth:`OrderBook.to_price`.

**Crossing limits are executed, not rejected.** ``add_limit`` walks the
opposite side first and rests only the remainder. A book that refuses crossing
orders is not a matching engine, and it makes the "aggressive child order"
strategies impossible to express.

**One clock, no wall time.** Every mutation takes a simulation timestamp from
the caller. Sequence numbers break ties, so priority is total and
deterministic: same seed, same book, byte for byte.

**Empty-side handling is explicit.** A book can legitimately be one-sided in a
simulation (a market order that eats every ask). ``best_ask`` then returns
``None`` and the callers have to say what that means, rather than silently
reading a stale price.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

BUY = 1
SELL = -1


@dataclass
class Order:
    """A resting order. ``qty`` is in lots and shrinks as it fills."""

    oid: int
    side: int          # BUY or SELL
    price: int         # ticks
    qty: int           # lots remaining
    agent: bool = False
    ts: float = 0.0
    seq: int = 0

    @property
    def is_live(self) -> bool:
        return self.qty > 0


@dataclass(frozen=True)
class Fill:
    """One (aggressor, resting) match.

    ``passive_agent`` and ``aggressive_agent`` are the two flags the execution
    layer cares about: a fill where the agent was passive earned the spread, and
    one where it was aggressive paid it.
    """

    price: int
    qty: int
    ts: float
    aggressor_side: int
    passive_agent: bool
    aggressive_agent: bool

    @property
    def signed_qty(self) -> int:
        """Positive when the aggressor bought. This is the sign convention the
        impact model uses: buyer-initiated volume pushes the price up."""
        return self.qty * self.aggressor_side


@dataclass
class BookStats:
    """Cheap running counters, so a simulation can report activity without a
    second pass over the tape."""

    n_limit: int = 0
    n_market: int = 0
    n_cancel: int = 0
    lots_traded: int = 0
    signed_lots: int = 0
    agent_passive_lots: int = 0
    agent_aggressive_lots: int = 0


class OrderBook:
    """Price-time priority book over integer tick prices.

    Parameters
    ----------
    tick_size, lot_size
        Conversion factors to dollars and shares. The engine itself never uses
        them; they exist so callers do not have to carry them separately.
    """

    __slots__ = ("tick_size", "lot_size", "_bids", "_asks", "_orders",
                 "_best_bid", "_best_ask", "_next_oid", "_seq", "stats",
                 "_level_qty")

    def __init__(self, tick_size: float = 0.01, lot_size: int = 100):
        self.tick_size = float(tick_size)
        self.lot_size = int(lot_size)
        self._bids: Dict[int, Deque[Order]] = {}
        self._asks: Dict[int, Deque[Order]] = {}
        self._level_qty: Dict[Tuple[int, int], int] = {}
        self._orders: Dict[int, Order] = {}
        self._best_bid: Optional[int] = None
        self._best_ask: Optional[int] = None
        self._next_oid = 1
        self._seq = 0
        self.stats = BookStats()

    # -- conversions -------------------------------------------------------

    def to_price(self, ticks: int) -> float:
        return ticks * self.tick_size

    def to_ticks(self, price: float) -> int:
        return int(round(price / self.tick_size))

    # -- inspection --------------------------------------------------------

    @property
    def best_bid(self) -> Optional[int]:
        return self._best_bid

    @property
    def best_ask(self) -> Optional[int]:
        return self._best_ask

    @property
    def mid(self) -> Optional[float]:
        if self._best_bid is None or self._best_ask is None:
            return None
        return 0.5 * (self._best_bid + self._best_ask) * self.tick_size

    @property
    def spread_ticks(self) -> Optional[int]:
        if self._best_bid is None or self._best_ask is None:
            return None
        return self._best_ask - self._best_bid

    def qty_at(self, side: int, price: int) -> int:
        return self._level_qty.get((side, price), 0)

    def queue_ahead(self, oid: int) -> int:
        """Lots ahead of order ``oid`` in its queue - the number that decides
        whether a passive order fills. O(queue length), and only ever called on
        the agent's own orders."""
        o = self._orders.get(oid)
        if o is None or not o.is_live:
            return 0
        book = self._bids if o.side == BUY else self._asks
        q = book.get(o.price)
        if q is None:
            return 0
        ahead = 0
        for other in q:
            if other.oid == oid:
                return ahead
            ahead += other.qty
        return ahead

    def depth_profile(self, n_levels: int) -> Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
        """``(bid_ticks, bid_lots, ask_ticks, ask_lots)`` for the top
        ``n_levels`` of each side, best first. Missing levels report zero size
        at their notional price so the animation has a stable x-axis."""
        bb = self._best_bid if self._best_bid is not None else (
            (self._best_ask - 1) if self._best_ask is not None else 0)
        ba = self._best_ask if self._best_ask is not None else bb + 1
        bt = np.arange(bb, bb - n_levels, -1, dtype=np.int64)
        at = np.arange(ba, ba + n_levels, dtype=np.int64)
        bq = np.array([self.qty_at(BUY, int(p)) for p in bt], dtype=np.int64)
        aq = np.array([self.qty_at(SELL, int(p)) for p in at], dtype=np.int64)
        return bt, bq, at, aq

    def total_depth(self, side: int, n_levels: int) -> int:
        bt, bq, at, aq = self.depth_profile(n_levels)
        return int(bq.sum()) if side == BUY else int(aq.sum())

    @property
    def n_orders(self) -> int:
        return sum(1 for o in self._orders.values() if o.is_live)

    def live_order_ids(self, agent_only: bool = False) -> List[int]:
        return [oid for oid, o in self._orders.items()
                if o.is_live and (o.agent or not agent_only)]

    # -- mutation ----------------------------------------------------------

    def add_limit(self, side: int, price: int, qty: int, ts: float = 0.0,
                  agent: bool = False) -> Tuple[int, List[Fill]]:
        """Submit a limit order. Crosses first, rests the remainder.

        Returns ``(oid, fills)``. ``oid`` is 0 when the order filled completely
        on arrival and never rested, which is the marketable-limit case.
        """
        if qty <= 0:
            return 0, []
        fills: List[Fill] = []
        remaining = qty

        # Cross against the opposite side while the limit price allows it.
        if side == BUY:
            while remaining > 0 and self._best_ask is not None and price >= self._best_ask:
                remaining, f = self._consume_level(SELL, self._best_ask, remaining,
                                                   ts, BUY, agent)
                fills.extend(f)
        else:
            while remaining > 0 and self._best_bid is not None and price <= self._best_bid:
                remaining, f = self._consume_level(BUY, self._best_bid, remaining,
                                                   ts, SELL, agent)
                fills.extend(f)

        if remaining <= 0:
            self.stats.n_limit += 1
            return 0, fills

        self._seq += 1
        oid = self._next_oid
        self._next_oid += 1
        order = Order(oid=oid, side=side, price=price, qty=remaining,
                      agent=agent, ts=ts, seq=self._seq)
        book = self._bids if side == BUY else self._asks
        q = book.get(price)
        if q is None:
            q = deque()
            book[price] = q
        q.append(order)
        self._orders[oid] = order
        key = (side, price)
        self._level_qty[key] = self._level_qty.get(key, 0) + remaining

        if side == BUY and (self._best_bid is None or price > self._best_bid):
            self._best_bid = price
        elif side == SELL and (self._best_ask is None or price < self._best_ask):
            self._best_ask = price

        self.stats.n_limit += 1
        return oid, fills

    def submit_market(self, side: int, qty: int, ts: float = 0.0,
                      agent: bool = False,
                      limit_ticks: Optional[int] = None) -> List[Fill]:
        """Marketable order for ``qty`` lots, walking the book.

        ``limit_ticks`` caps how far it will walk (an aggressive *limit* order,
        which is what a real execution algo sends - an uncapped market order
        into a thin book is how you print a trade 3% away from the mid). The
        unfilled remainder is simply not executed; the caller decides whether to
        retry.
        """
        self.stats.n_market += 1
        fills: List[Fill] = []
        remaining = int(qty)
        while remaining > 0:
            opp = SELL if side == BUY else BUY
            best = self._best_ask if side == BUY else self._best_bid
            if best is None:
                break
            if limit_ticks is not None:
                if side == BUY and best > limit_ticks:
                    break
                if side == SELL and best < limit_ticks:
                    break
            remaining, f = self._consume_level(opp, best, remaining, ts, side, agent)
            fills.extend(f)
        return fills

    def cancel(self, oid: int, qty: Optional[int] = None) -> int:
        """Cancel all or part of a resting order. Returns lots removed."""
        o = self._orders.get(oid)
        if o is None or not o.is_live:
            return 0
        take = o.qty if qty is None else min(int(qty), o.qty)
        o.qty -= take
        key = (o.side, o.price)
        self._level_qty[key] = self._level_qty.get(key, 0) - take
        if o.qty == 0:
            self._remove_dead(o)
        if self._level_qty.get(key, 0) <= 0:
            self._level_qty.pop(key, None)
            self._drop_level_if_empty(o.side, o.price)
        self.stats.n_cancel += 1
        return take

    def cancel_random(self, side: int, price: int, lots: int,
                      u: float) -> int:
        """Cancel ``lots``, chosen uniformly at random over the resting lots at
        a level - the cancellation model in Cont-Stoikov-Talreja.

        Uniform-over-lots is the right null: it makes the cancellation hazard
        proportional to displayed size, which is the empirical regularity the
        intensity function encodes. Picking uniformly over *orders* instead
        would make one 500-lot order as fragile as a 1-lot order.

        The agent's own orders are excluded - the flow model cancels other
        people's liquidity, not ours. ``u`` is a uniform draw supplied by the
        caller; taking it as an argument keeps this function free of any RNG
        machinery, which is what makes it cheap enough for the inner loop.
        """
        if lots <= 0:
            return 0
        book = self._bids if side == BUY else self._asks
        q = book.get(price)
        if not q:
            return 0
        anon = 0
        for o in q:
            if o.is_live and not o.agent:
                anon += o.qty
        if anon <= 0:
            return 0
        removed = 0
        for _ in range(min(lots, anon)):
            target = int(u * anon)
            if target >= anon:
                target = anon - 1
            acc = 0
            victim = None
            for o in q:
                if not o.is_live or o.agent:
                    continue
                acc += o.qty
                if acc > target:
                    victim = o
                    break
            if victim is None:
                break
            removed += self.cancel(victim.oid, 1)
            anon -= 1
            u = (u * 997.0) % 1.0        # cheap decorrelation for repeat lots
        return removed

    # -- internals ---------------------------------------------------------

    def _consume_level(self, resting_side: int, price: int, remaining: int,
                       ts: float, aggressor_side: int,
                       aggressor_agent: bool) -> Tuple[int, List[Fill]]:
        """Match against one price level in FIFO order."""
        book = self._bids if resting_side == BUY else self._asks
        q = book.get(price)
        fills: List[Fill] = []
        if not q:
            self._drop_level_if_empty(resting_side, price)
            return remaining, fills
        while q and remaining > 0:
            head = q[0]
            if not head.is_live:
                q.popleft()
                self._orders.pop(head.oid, None)
                continue
            take = min(head.qty, remaining)
            head.qty -= take
            remaining -= take
            key = (resting_side, price)
            self._level_qty[key] = self._level_qty.get(key, 0) - take
            fills.append(Fill(price=price, qty=take, ts=ts,
                              aggressor_side=aggressor_side,
                              passive_agent=head.agent,
                              aggressive_agent=aggressor_agent))
            self.stats.lots_traded += take
            self.stats.signed_lots += take * aggressor_side
            if head.agent:
                self.stats.agent_passive_lots += take
            if aggressor_agent:
                self.stats.agent_aggressive_lots += take
            if head.qty == 0:
                q.popleft()
                self._orders.pop(head.oid, None)
        if not q:
            self._level_qty.pop((resting_side, price), None)
            self._drop_level_if_empty(resting_side, price)
        return remaining, fills

    def _remove_dead(self, o: Order) -> None:
        book = self._bids if o.side == BUY else self._asks
        q = book.get(o.price)
        if q is None:
            return
        # Cheap when the dead order is at the front (the common case after a
        # fill); the linear scan only happens for a mid-queue cancel.
        if q and q[0].oid == o.oid:
            q.popleft()
        else:
            try:
                q.remove(o)
            except ValueError:
                pass
        self._orders.pop(o.oid, None)

    def _drop_level_if_empty(self, side: int, price: int) -> None:
        book = self._bids if side == BUY else self._asks
        q = book.get(price)
        if q is not None:
            while q and not q[0].is_live:
                dead = q.popleft()
                self._orders.pop(dead.oid, None)
            if not q:
                book.pop(price, None)
                self._level_qty.pop((side, price), None)
        if side == BUY and self._best_bid is not None and price >= self._best_bid:
            self._best_bid = max(self._bids) if self._bids else None
        if side == SELL and self._best_ask is not None and price <= self._best_ask:
            self._best_ask = min(self._asks) if self._asks else None

    # -- construction helpers ---------------------------------------------

    def seed_symmetric(self, mid_ticks: int, spread_ticks: int, n_levels: int,
                       depth_lots: int, decay: float = 0.0,
                       ts: float = 0.0) -> None:
        """Fill a fresh book with a symmetric ladder around ``mid_ticks``.

        A simulation that starts from an empty book spends its first minutes
        building one, and every statistic collected over that period is a
        statistic about the burn-in rather than about the market. Seeding a
        ladder and then letting the flow model reshape it is faster and more
        honest than pretending the first minute is stationary.
        """
        half = max(1, spread_ticks // 2)
        for k in range(1, n_levels + 1):
            lots = max(1, int(round(depth_lots * np.exp(-decay * (k - 1)))))
            self.add_limit(BUY, mid_ticks - half - (k - 1), lots, ts=ts)
            self.add_limit(SELL, mid_ticks + half + (k - 1), lots, ts=ts)

    def check_invariants(self) -> None:
        """Assert the book is internally consistent. Used by the tests and by
        the simulator under ``--paranoid``; too slow for the inner loop."""
        if self._best_bid is not None and self._best_ask is not None:
            assert self._best_bid < self._best_ask, (
                f"crossed book: bid {self._best_bid} >= ask {self._best_ask}")
        for side, book in ((BUY, self._bids), (SELL, self._asks)):
            for price, q in book.items():
                assert q, f"empty deque left at {side} {price}"
                total = sum(o.qty for o in q)
                assert total == self._level_qty.get((side, price), 0), (
                    f"level cache mismatch at {side} {price}")
                seqs = [o.seq for o in q]
                assert seqs == sorted(seqs), f"queue out of time order at {price}"
        if self._bids:
            assert self._best_bid == max(self._bids)
        if self._asks:
            assert self._best_ask == min(self._asks)


__all__ = ["OrderBook", "Order", "Fill", "BookStats", "BUY", "SELL"]
