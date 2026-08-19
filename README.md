# Optimal execution against a simulated limit order book

A limit order book simulator with Hawkes-driven order flow, the Almgren–Chriss
closed form solved on top of it, and an honest measurement of where the closed
form stops describing the market it is supposed to describe.

**[→ Open the interactive interface](https://codeebytee.github.io/08-lob-optimal-execution/)**
(runs offline; no server, no install — or double-click `docs/index.html`)

![the interface](results/interface.gif)

---

## The problem

A portfolio manager decides to buy 250,000 shares of Microsoft. Trade fast and
you walk up the book and reveal your intent; the price moves against you and
stays moved. Trade slowly and you avoid that, but you are exposed to volatility
for half an hour. Almgren–Chriss (2000) is the canonical answer: minimise
`E[cost] + λ·Var[cost]`.

Implementing that closed form is fifteen lines. The interesting question is
whether it describes anything real — its assumptions (linear impact, an
infinitely elastic market, a schedule that can always be executed) are all false
in ways that matter. So this project builds the market the model is supposed to
describe, calibrates the model to that market, and measures the gap.

## What it does

- **A matching engine** (`src/lob/book.py`) with FIFO priority within a price,
  price priority across prices, partial cancels that preserve queue position.
- **A venue around it** (`src/lob/simulator.py`): power-law limit order arrival
  by distance from touch, two-sided Hawkes market orders, hazard-rate cancels,
  stale-quote handling, and quotes anchored to a latent efficient price with a
  Kyle-λ impact term.
- **Calibration** (`src/flow/calibrate.py`) of that venue to real names from a
  daily-bar snapshot — flow rates to ADV, Kyle λ to volatility-over-volume.
- **Five execution algorithms** (TWAP, VWAP, POV, Almgren–Chriss, Adaptive)
  compared on common random numbers with paired counterfactuals.
- **An interactive page** (`docs/`) that recomputes the real solver in the
  browser on every slider move, with the JS port pinned to the Python by a
  parity test.

## Headline results

Calibrated to 8 names (SPY, AAPL, MSFT, XOM, KO, TSLA, GME, IWM) as of
2026-08-17. Every number below is produced by `scripts/make_results.py` and
stored in `results/`.

**Half of a large order's cost is untouchable.** The permanent-impact term is
identical across algorithms, so once parent size is fixed the entire argument
between execution algos is over the temporary part. The highest-value
conversation is with the PM about size, not with the vendor about algorithms.

**Between algorithms, differences are small, real, and easily mistaken for
noise.** On a single order the spread of outcomes is an order of magnitude
larger than the gap between algorithms — which is why execution quality is
judged over hundreds of orders and never over one. The paired tests in
`results/paired_vs_twap.csv` are what separate signal from luck.

**The model understates the cost of urgency.** The closed form charges `ηv` per
share — linear — while walking a finite book is convex. Agreement is good for
patient schedules and deteriorates as `λ` rises.

**Past a certain size, the model stops describing anything.** Below roughly 0.5%
of a day's volume in half an hour, everything fills and costs grow smoothly.
Past that the venue cannot supply the shares inside the horizon, and the closed
form — which has no term for an unfillable order — gives an answer that is not
merely inaccurate but meaningless.

**Across names, cost is volatility over liquidity.** What separates expensive
names from cheap ones is not price or sector but the ratio of dollar volatility
to volume — the same quantity Kyle's λ is built from.

## Validation

148 tests (`pytest`). The engine is fuzz-tested over 3,000 random operations
against its invariants. Hawkes MLE recovers the branching ratio to ±0.12 and
time-rescaling residuals are unit-exponential under the true model. The closed
form reproduces TWAP exactly as `λ→0`, satisfies its defining `cosh` equation to
1e-12, and matches a 40,000-path Monte Carlo to 2–3%. The measured impact
coefficient recovers the Kyle λ the venue was built with to ~85–90%. The
JavaScript port matches `src/` to 6e-15.

Two spread estimators (Corwin–Schultz, Abdi–Ranaldo) are validated against a
synthetic Roll model where the true spread is known, then both fail on real
large caps by one to two orders of magnitude in opposite directions. That
failure is a reported result, and it is why the project does not take its spread
from daily bars.

## Running it

```bash
pip install -r requirements.txt

pytest                                  # 148 tests
python scripts/make_results.py --quick  # ~2 min smoke test
python scripts/make_results.py          # full run, ~3.5 min on 8 cores
python scripts/check_page.py            # verify docs/ ships and the JS matches src/
```

Everything is seeded — two runs produce identical numbers.
`scripts/refresh_data.py` is the only script that touches the network; without
it the committed snapshot in `data/` is used.

## Where to start

| If you want | Read |
|---|---|
| No finance background | [`PREREQUISITES.md`](PREREQUISITES.md) |
| The picture, in 30 seconds | [`docs/index.html`](docs/index.html) — the "Live book" tab, press play |
| The maths and the failure modes | [`DEEP_DIVE.md`](DEEP_DIVE.md) |
| The results built up step by step | [`notebooks/execution_story.ipynb`](notebooks/execution_story.ipynb) |
| The code | `src/lob/book.py` → `simulator.py` → `execution/almgren_chriss.py` → `execution/runner.py` → `flow/calibrate.py` |

## References

Almgren & Chriss (2000), *Optimal execution of portfolio transactions*.
Cont, Stoikov & Talreja (2010), *A stochastic model for order book dynamics*.
Kyle (1985), *Continuous auctions and insider trading*.
Full list in [`DEEP_DIVE.md`](DEEP_DIVE.md#9-references).

## License

MIT — see [`LICENSE`](LICENSE).
