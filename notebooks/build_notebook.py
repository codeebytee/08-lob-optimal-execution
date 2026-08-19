"""Generate ``notebooks/execution_story.ipynb``.

    python notebooks/build_notebook.py

The notebook is generated rather than hand-authored so that it cannot drift
away from the library: every cell imports from ``src/`` and recomputes, and if
an API changes the notebook stops running and this script has to be fixed. A
hand-edited notebook with stale outputs pasted in is worse than no notebook.

It is deliberately the *only* notebook in the repo, and it tells the story once:
here is the book, here is what an order does to it, here is the model, here is
where the model is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "execution_story.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Working a parent order: the book, the schedule, and the gap between them

This notebook builds the project's result in the order it was actually
discovered. Everything imports from `src/`; nothing is reimplemented here.

1. A limit order book with queue priority, driven by stochastic order flow.
2. What a large buy order does to it.
3. The Almgren-Chriss closed form, and its two limits.
4. Measuring the impact parameters *from the simulator* rather than assuming
   them.
5. The comparison that matters: model cost against simulated cost, and where
   the model stops being true.
"""),
    code("""
import sys, math
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.utils.config import AppConfig
from src.data.market import load_snapshot
from src.lob.book import OrderBook, BUY, SELL
from src.lob.simulator import MarketSimulator
from src.execution.almgren_chriss import ACParams, solve, frontier, bps, schedule_cost
from src.execution.schedules import TWAP, VWAP, POV, AlmgrenChriss, Adaptive
from src.execution.runner import run_parent, baseline_path, control_variate
from src.flow.calibrate import calibrate_venue, measure_impact
from src.utils.stats import cost_stats, paired_test

cfg = AppConfig.load(ROOT / "config.yaml")
snap = load_snapshot(cfg.market)
stats = snap[cfg.market.default_ticker]
print(f"{stats.ticker}: ${stats.price:.2f}, {stats.sigma_annual:.1%} annual vol, "
      f"{stats.adv_shares/1e6:.1f}M shares/day, data as of {snap.as_of} ({snap.source})")
"""),
    md("""
## 1. The matching engine

Price-time priority, in a few lines. The point of the assertions is that this is
a real queue: the order that arrived first is the order that fills first, and a
crossing limit order executes rather than resting.
"""),
    code("""
b = OrderBook(tick_size=0.01, lot_size=100)
first, _  = b.add_limit(SELL, 10000, 3, ts=0.0)   # 300 shares at $100.00
second, _ = b.add_limit(SELL, 10000, 5, ts=1.0)   # 500 shares, behind it
b.add_limit(BUY, 9999, 4, ts=1.5)

print("before:", b.qty_at(SELL, 10000), "lots offered at", b.to_price(10000))
print("queue ahead of the second order:", b.queue_ahead(second), "lots")

fills = b.submit_market(BUY, 4, ts=2.0)           # buy 400 shares
for f in fills:
    print(f"  filled {f.qty} lots at {b.to_price(f.price):.2f}")
print("after: ", b.qty_at(SELL, 10000), "lots left;",
      "queue ahead of the second order is now", b.queue_ahead(second))
b.check_invariants()
"""),
    md("""
## 2. A session, and what it looks like

The simulator adds three order-flow intensities and a latent efficient price.
The calibration step scales every rate by one factor so the session prints the
name's real daily volume - see `src/flow/calibrate.py` for why one factor is
enough.
"""),
    code("""
cal = calibrate_venue(stats, cfg.book, cfg.flow, T=900.0,
                      seconds_per_day=cfg.market.seconds_per_day)
print(f"rate scale {cal.rate_scale:.3f}")
print(f"simulated ADV {cal.sim_adv_shares/1e6:.1f}M vs real {stats.adv_shares/1e6:.1f}M")
print(f"simulated vol/s {cal.sim_sigma_per_sec:.5f} vs target {cal.target_sigma_per_sec:.5f}")
print(f"emergent spread {cal.median_spread_ticks:.1f} ticks, "
      f"depth at touch {cal.mean_touch_lots:.1f} lots "
      f"({cal.mean_touch_lots*cfg.book.lot_size:.0f} shares)")
"""),
    code("""
sim = MarketSimulator(stats, cfg.book, cal.flow, np.random.default_rng(7),
                      kyle_lambda=cal.kyle_lambda, sigma_exo_per_sec=cal.sigma_exo,
                      record_every=2.0, seconds_per_day=cfg.market.seconds_per_day)
sim.run_until(1800.0)

t   = [s.t for s in sim.snapshots]
mid = [0.5*(s.best_bid + s.best_ask) for s in sim.snapshots]
spr = [round((s.best_ask - s.best_bid)/stats.tick_size) for s in sim.snapshots]
dep = [s.bid_lots[0] + s.ask_lots[0] for s in sim.snapshots]

fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t, mid, lw=0.8); ax[0].set_ylabel("mid ($)")
ax[1].plot(t, spr, lw=0.6);  ax[1].set_ylabel("spread (ticks)")
ax[2].plot(t, dep, lw=0.6);  ax[2].set_ylabel("depth at touch (lots)")
ax[2].set_xlabel("seconds")
fig.suptitle(f"one simulated half hour in {stats.ticker}")
fig.tight_layout()
print(f"{sim.n_events:,} events, {sim.market_lots_traded*cfg.book.lot_size:,} shares printed")
"""),
    md("""
## 3. The closed form, and its limits

Almgren-Chriss minimises `E[cost] + lambda Var[cost]`. Two limits are worth
checking before trusting anything it says: at zero risk aversion the optimal
trajectory must be a straight line (TWAP), and as risk aversion grows the order
must be pulled forward.
"""),
    code("""
X = int(round(0.005 * stats.adv_shares / 100) * 100)
p = ACParams(X=float(X), T=1800.0, N=30,
             sigma=stats.sigma_per_second(cfg.market.seconds_per_day),
             eta=1.75e-4, gamma=2.6e-7, epsilon=0.01)   # replaced below by fitted values

fig, ax = plt.subplots(figsize=(7, 4))
for lam in (0.0, 1e-7, 1e-6, 1e-5):
    s = solve(p, lam)
    label = "lambda = 0 (TWAP)" if lam == 0 else f"lambda = {lam:.0e}"
    ax.plot(s["t"]/60, s["x"]/1e3, label=label)
ax.set_xlabel("minutes"); ax.set_ylabel("shares remaining (thousands)")
ax.legend(); ax.set_title("optimal trajectories at four urgencies")

zero = solve(p, 0.0)
print("zero risk aversion is exactly TWAP:",
      np.allclose(zero["n"], p.X/p.N))
"""),
    md("""
## 4. Measuring the impact parameters instead of assuming them

`gamma` and `eta` are not inputs here. They are estimated by running parent
orders through the simulator and comparing against **the identical session with
no order in it**. That counterfactual is what separates a price move you caused
from one that was going to happen anyway, and without it the estimate is buried
under a price path ten to a hundred times larger than the effect.

This cell runs a few hundred simulated sessions and takes a couple of minutes.
"""),
    code("""
ic = measure_impact(stats, cfg.book, cal,
                    participations=(0.01, 0.02, 0.05, 0.10, 0.20),
                    paths=40, horizon=1800.0, n_slices=30,
                    seconds_per_day=cfg.market.seconds_per_day)
print(f"gamma   = {ic.gamma:.4g} $/share per share")
print(f"eta     = {ic.eta:.4g} $/share per (share/second)")
print(f"epsilon = {ic.epsilon:.4g} $/share")
print(f"fitted impact exponent = {ic.exponent:.2f} (the model assumes 1.00)")
print(f"injected Kyle lambda   = {cal.kyle_lambda/cfg.book.lot_size:.4g} $/share")
print()
for r in ic.rows:
    print(f"  participation {r['participation']:5.1%}  "
          f"perm ${r['perm_usd']:6.3f} +/- {r['perm_se']:.3f}  "
          f"cost/share ${r['cost_per_share']:7.4f}  fill {r['fill_rate']:5.1%}")
"""),
    md("""
## 5. The tournament

Five algorithms, the same simulated sessions, common random numbers. The
control variate is the counterfactual session's flat-schedule cost: mean zero
by construction, and strongly correlated with every algorithm's outcome, so
subtracting it cuts the standard error by roughly an order of magnitude without
touching the estimate.
"""),
    code("""
p = ACParams(X=float(X), T=1800.0, N=30,
             sigma=stats.sigma_per_second(cfg.market.seconds_per_day),
             eta=ic.eta, gamma=ic.gamma, epsilon=ic.epsilon)
lam = 1e-6
algos = lambda: [TWAP(), VWAP(cal.flow), POV(cfg.execution.pov_rate),
                 AlmgrenChriss(p, lam), Adaptive(p, lam)]

raw = {a.name: [] for a in algos()}
adj = {a.name: [] for a in algos()}
for seed in range(60):
    base = baseline_path(stats, cfg.book, cal.flow, seed, 1800.0, 30,
                         kyle_lambda=cal.kyle_lambda, sigma_exo=cal.sigma_exo,
                         seconds_per_day=cfg.market.seconds_per_day)
    cv = control_variate(base)
    for a in algos():
        r = run_parent(stats, cfg.book, cal.flow, a, X, seed, 1800.0, 30,
                       kyle_lambda=cal.kyle_lambda, sigma_exo=cal.sigma_exo,
                       seconds_per_day=cfg.market.seconds_per_day)
        raw[a.name].append(r.shortfall_bps)
        adj[a.name].append(r.shortfall_bps - cv)

for name in raw:
    s_raw, s_adj = cost_stats(raw[name]), cost_stats(adj[name])
    print(f"{name:<9} mean {s_adj.mean:6.2f} +/- {s_adj.stderr:4.2f} bp "
          f"(raw se {s_raw.stderr:5.2f})  sd {s_raw.stdev:6.2f}  "
          f"CVaR95 {s_raw.cvar95:7.2f}")
"""),
    code("""
print("paired against TWAP (negative = cheaper):")
for name in raw:
    t = paired_test(adj[name], adj["TWAP"])
    print(f"  {name:<9} {t.mean_diff:+6.2f} bp   t = {t.t_stat:+5.2f}   "
          f"wins {t.win_rate:.0%}")
"""),
    md("""
## 6. Model against venue

The closed form says a schedule costs a particular number of basis points. The
simulator says what it actually cost. The gap is the interesting quantity, and
it is not constant - it grows with size, because the model assumes a market that
can always supply the shares and the book cannot.
"""),
    code("""
sizes = [0.001, 0.0025, 0.005, 0.01, 0.02]
model_bps, sim_bps, fills = [], [], []
for size in sizes:
    Xi = int(round(size * stats.adv_shares / 100) * 100)
    pi = ACParams(X=float(Xi), T=1800.0, N=30, sigma=p.sigma, eta=ic.eta,
                  gamma=ic.gamma, epsilon=ic.epsilon)
    model_bps.append(bps(schedule_cost(pi, np.full(30, Xi/30))["expected_cost"],
                         Xi, stats.price))
    costs, fr = [], []
    for seed in range(30):
        base = baseline_path(stats, cfg.book, cal.flow, seed, 1800.0, 30,
                             kyle_lambda=cal.kyle_lambda, sigma_exo=cal.sigma_exo,
                             seconds_per_day=cfg.market.seconds_per_day)
        r = run_parent(stats, cfg.book, cal.flow, TWAP(), Xi, seed, 1800.0, 30,
                       kyle_lambda=cal.kyle_lambda, sigma_exo=cal.sigma_exo,
                       seconds_per_day=cfg.market.seconds_per_day)
        costs.append(r.shortfall_bps - control_variate(base))
        fr.append(r.filled_shares / Xi)
    sim_bps.append(np.mean(costs)); fills.append(np.mean(fr))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot([100*s for s in sizes], model_bps, "o-", label="Almgren-Chriss")
ax[0].plot([100*s for s in sizes], sim_bps, "s-", label="order book simulation")
ax[0].set_xlabel("parent size (% of ADV)"); ax[0].set_ylabel("TWAP cost (bp)")
ax[0].legend()
ax[1].plot([100*s for s in sizes], [100*f for f in fills], "o-")
ax[1].set_xlabel("parent size (% of ADV)"); ax[1].set_ylabel("fill rate (%)")
fig.tight_layout()
for s, m, v, f in zip(sizes, model_bps, sim_bps, fills):
    print(f"  {s:6.2%} of ADV: model {m:7.2f} bp   simulated {v:7.2f} bp   "
          f"fill {f:5.1%}")
"""),
    md("""
## What to take away

- Half the cost of a large order is permanent impact, and no schedule touches
  it. The argument worth having is about parent size, not about algorithm
  choice.
- Between algorithms the differences are small and mostly not significant on a
  single order. They are real, and they are worth having, but only over many
  orders - which is why execution quality is a statistical claim rather than an
  anecdote.
- The closed form is a good description of the venue up to the point where the
  book stops being able to supply the shares. Past that its cost estimate is
  fiction, because it has no term for an order that cannot be filled.

The full grid, the figures and the numbers quoted in `DEEP_DIVE.md` come from
`python scripts/make_results.py`; this notebook is a smaller version of the same
pipeline.
"""),
]


def main() -> int:
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
