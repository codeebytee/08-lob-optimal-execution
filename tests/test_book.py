"""The matching engine. If these fail, nothing downstream means anything."""

from __future__ import annotations

import numpy as np
import pytest

from src.lob.book import BUY, SELL, OrderBook


@pytest.fixture
def book() -> OrderBook:
    return OrderBook(tick_size=0.01, lot_size=100)


def test_best_prices_track_additions(book):
    book.add_limit(BUY, 9990, 5)
    book.add_limit(SELL, 10010, 5)
    assert book.best_bid == 9990
    assert book.best_ask == 10010
    book.add_limit(BUY, 9995, 3)
    assert book.best_bid == 9995
    book.add_limit(SELL, 10005, 3)
    assert book.best_ask == 10005
    assert book.mid == pytest.approx(0.5 * (9995 + 10005) * 0.01)


def test_empty_book_has_no_mid(book):
    assert book.best_bid is None and book.best_ask is None
    assert book.mid is None and book.spread_ticks is None


def test_time_priority_is_fifo(book):
    first, _ = book.add_limit(SELL, 10000, 2, ts=0.0)
    second, _ = book.add_limit(SELL, 10000, 2, ts=1.0)
    fills = book.submit_market(BUY, 2, ts=2.0)
    assert sum(f.qty for f in fills) == 2
    # The first order is gone, the second is untouched.
    assert book.queue_ahead(second) == 0
    assert book.qty_at(SELL, 10000) == 2
    assert first not in book.live_order_ids()


def test_price_priority_beats_time(book):
    old, _ = book.add_limit(SELL, 10010, 5, ts=0.0)
    new, _ = book.add_limit(SELL, 10000, 5, ts=9.0)
    fills = book.submit_market(BUY, 3, ts=10.0)
    assert all(f.price == 10000 for f in fills)
    assert book.qty_at(SELL, 10010) == 5


def test_queue_ahead_counts_lots_not_orders(book):
    book.add_limit(BUY, 9990, 7, ts=0.0)
    book.add_limit(BUY, 9990, 3, ts=1.0)
    mine, _ = book.add_limit(BUY, 9990, 1, ts=2.0, agent=True)
    assert book.queue_ahead(mine) == 10


def test_market_order_walks_levels(book):
    for k, px in enumerate((10000, 10001, 10002)):
        book.add_limit(SELL, px, 2)
    fills = book.submit_market(BUY, 5, ts=1.0)
    assert [(f.price, f.qty) for f in fills] == [(10000, 2), (10001, 2), (10002, 1)]
    assert book.best_ask == 10002
    assert book.qty_at(SELL, 10002) == 1


def test_market_order_respects_price_cap(book):
    book.add_limit(SELL, 10000, 1)
    book.add_limit(SELL, 10005, 10)
    fills = book.submit_market(BUY, 6, ts=1.0, limit_ticks=10002)
    assert sum(f.qty for f in fills) == 1        # stopped rather than sweeping
    assert book.qty_at(SELL, 10005) == 10


def test_market_order_into_empty_side_is_a_no_op(book):
    book.add_limit(BUY, 9990, 5)
    assert book.submit_market(BUY, 3, ts=1.0) == []


def test_crossing_limit_executes_then_rests(book):
    book.add_limit(SELL, 10000, 2)
    oid, fills = book.add_limit(BUY, 10000, 5, ts=1.0)
    assert sum(f.qty for f in fills) == 2
    assert oid != 0
    assert book.qty_at(BUY, 10000) == 3
    assert book.best_ask is None
    book.check_invariants()


def test_fully_filled_crossing_limit_never_rests(book):
    book.add_limit(SELL, 10000, 5)
    oid, fills = book.add_limit(BUY, 10000, 5, ts=1.0)
    assert oid == 0
    assert sum(f.qty for f in fills) == 5
    assert book.qty_at(BUY, 10000) == 0


def test_partial_cancel_leaves_priority(book):
    a, _ = book.add_limit(BUY, 9990, 5, ts=0.0)
    b, _ = book.add_limit(BUY, 9990, 5, ts=1.0)
    assert book.cancel(a, 3) == 3
    assert book.qty_at(BUY, 9990) == 7
    assert book.queue_ahead(b) == 2


def test_cancel_removes_level_and_repricing_best(book):
    book.add_limit(BUY, 9990, 5)
    oid, _ = book.add_limit(BUY, 9995, 5)
    book.cancel(oid)
    assert book.best_bid == 9990
    book.check_invariants()


def test_cancel_random_never_touches_agent_orders(book):
    book.add_limit(BUY, 9990, 4, agent=False)
    mine, _ = book.add_limit(BUY, 9990, 4, agent=True)
    for u in np.linspace(0.01, 0.99, 8):
        book.cancel_random(BUY, 9990, 1, float(u))
    assert book.qty_at(BUY, 9990) == 4          # only the anonymous lots went
    assert mine in book.live_order_ids(agent_only=True)


def test_cancel_random_is_uniform_over_lots(book):
    """A 9-lot order should be hit nine times as often as a 1-lot order."""
    hits_big = 0
    trials = 400
    rng = np.random.default_rng(3)
    for _ in range(trials):
        b = OrderBook()
        big, _ = b.add_limit(BUY, 100, 9)
        small, _ = b.add_limit(BUY, 100, 1)
        b.cancel_random(BUY, 100, 1, float(rng.random()))
        hits_big += 1 if b._orders[big].qty == 8 else 0
    # Expected 90% of hits on the big order; allow a wide band.
    assert 0.82 < hits_big / trials < 0.97


def test_fill_flags_identify_the_agent_side(book):
    book.add_limit(SELL, 10000, 3, agent=True)
    fills = book.submit_market(BUY, 2, ts=1.0, agent=False)
    assert fills[0].passive_agent and not fills[0].aggressive_agent

    other = OrderBook()
    other.add_limit(BUY, 9990, 2, agent=False)
    out = other.submit_market(SELL, 1, ts=2.0, agent=True)
    assert out and out[0].aggressive_agent and not out[0].passive_agent


def test_signed_qty_convention(book):
    book.add_limit(SELL, 10000, 1)
    f = book.submit_market(BUY, 1, ts=0.0)[0]
    assert f.signed_qty == 1
    book.add_limit(BUY, 9990, 1)
    g = book.submit_market(SELL, 1, ts=0.0)[0]
    assert g.signed_qty == -1


def test_stats_accumulate(book):
    book.add_limit(SELL, 10000, 4)
    book.submit_market(BUY, 3, ts=0.0)
    assert book.stats.lots_traded == 3
    assert book.stats.signed_lots == 3
    assert book.stats.n_market == 1
    assert book.stats.n_limit == 1


def test_seed_symmetric_is_uncrossed_and_deep(book):
    book.seed_symmetric(mid_ticks=10000, spread_ticks=2, n_levels=5,
                        depth_lots=10)
    book.check_invariants()
    assert book.best_bid < book.best_ask
    bt, bq, at, aq = book.depth_profile(5)
    assert bq.sum() == aq.sum() == 50


def test_depth_profile_reports_zero_for_missing_levels(book):
    book.add_limit(BUY, 9990, 2)
    book.add_limit(SELL, 10000, 2)
    bt, bq, at, aq = book.depth_profile(4)
    assert list(bq) == [2, 0, 0, 0]
    assert list(aq) == [2, 0, 0, 0]
    assert bt[0] == 9990 and at[0] == 10000


def test_invariants_catch_a_crossed_book(book):
    """Force an inconsistent state and confirm the checker sees it."""
    book.add_limit(BUY, 9990, 1)
    book._asks[9980] = book._bids[9990]     # deliberately corrupt
    book._best_ask = 9980
    with pytest.raises(AssertionError):
        book.check_invariants()


def test_random_flow_keeps_the_book_consistent():
    """Fuzz: a few thousand random operations must not break an invariant."""
    rng = np.random.default_rng(11)
    b = OrderBook()
    b.seed_symmetric(10000, 2, 8, 10)
    for _ in range(3000):
        u = rng.random()
        if u < 0.45:
            side = BUY if rng.random() < 0.5 else SELL
            offset = int(rng.integers(0, 6))
            px = (b.best_bid or 9999) - offset if side == BUY else (b.best_ask or 10001) + offset
            b.add_limit(side, int(px), int(rng.integers(1, 5)), ts=float(_))
        elif u < 0.75:
            side = BUY if rng.random() < 0.5 else SELL
            b.submit_market(side, int(rng.integers(1, 4)), ts=float(_))
        else:
            live = b.live_order_ids()
            if live:
                b.cancel(int(rng.choice(live)), int(rng.integers(1, 3)))
    b.check_invariants()
