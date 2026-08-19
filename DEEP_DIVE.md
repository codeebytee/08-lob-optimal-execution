# Optimal execution against a simulated limit order book

## 1. Problem statement

A portfolio manager decides to buy 250,000 shares of Microsoft. That decision is
made at a price — the mid at the moment of the decision — and everything that
happens between that moment and the completion of the order is execution, which
is somebody's job and somebody's P&L. On a large order, execution cost is
routinely larger than the fee, the commission and the spread combined, and it is
the one cost that scales with size rather than with count.

The trader's problem is a trade-off with two sides that pull in opposite
directions:

- **Trade fast** and you consume the visible book, walk up several price levels
  per child order, and reveal your intent to everyone watching; the price moves
  against you and stays moved.
- **Trade slowly** and you avoid all that, but the order sits unfinished for
  half an hour while the price does whatever it was going to do anyway.
  Volatility over that window is your risk, and it scales as √T.

Almgren and Chriss (2000) is the canonical answer: model both effects, and
choose the trading trajectory minimising `E[cost] + λ·Var[cost]`. Every
execution desk runs a variant of it, and every execution-algo interview asks
about it.

The interesting question is not whether one can implement that closed form —
it is fifteen lines. It is whether the closed form describes anything real. Its
assumptions (linear impact, an infinitely elastic market, a schedule that can
always be executed) are all false in a way that matters. So this project builds
the market it is supposed to describe, calibrates the model to that market, and
then measures the gap. The desk-relevant question — *when can I trust the
number my cost model prints?* — has an answer here, and the answer is not
"always".

## 2. Mathematical formulation

### 2.1 The venue

The book is driven by three intensities, in the spirit of Cont, Stoikov and
Talreja (2010). At distance `k` ticks from the reference price:

- limit orders arrive at rate `λ(k) = K / k^α` lots per second per side;
- market orders arrive per side as a Hawkes process (below);
- each resting lot is cancelled with hazard `θ(k) = Θ / k^{α_c}` per second,
  and any quote outside the top `n` levels is treated as stale and cancelled at
  the much higher hazard `Θ_stale`.

Trade arrivals are a two-sided Hawkes process with exponential kernel,

$$\lambda_t = \mu_0 + \sum_{t_i < t} \alpha\, e^{-\beta (t - t_i)},$$

whose branching ratio `n = α/β` is the expected number of trades each trade
provokes, and whose stationary intensity is `E[λ] = μ₀/(1−n)`. Trades cluster;
a Poisson model would badly understate the probability of the ten quiet minutes
in which your order needs liquidity and there is none.

Quotes are placed around a **latent efficient price**

$$S_t = S_0 + \sigma W_t + \lambda_{\text{Kyle}} \sum_{\text{trades}} \epsilon_i q_i,$$

an arithmetic Brownian motion plus a Kyle-style permanent impact term in signed
volume. That last ingredient is not in the pure flow model, and it is the most
consequential modelling decision in the repo — see §3.1.

### 2.2 The cost model

Split `[0,T]` into `N` intervals of length `τ = T/N`. Let `x_k` be shares
remaining at `t_k = kτ` (so `x_0 = X`, `x_N = 0`), `n_k = x_{k-1} - x_k` the
shares traded in interval `k`, and `v_k = n_k/τ` the rate. Prices evolve as

$$S_k = S_{k-1} + \sigma\sqrt{\tau}\,\xi_k - \tau\, g(v_k), \qquad
\tilde S_k = S_{k-1} - h(v_k),$$

with permanent impact `g(v) = γv` and temporary impact
`h(v) = ε·sgn(v) + ηv`. The implementation shortfall then has

$$\mathbb{E}[C] = \underbrace{\tfrac{1}{2}\gamma X^2}_{\text{permanent}}
+ \underbrace{\epsilon \sum_k |n_k|}_{\text{spread}}
+ \underbrace{\frac{\tilde\eta}{\tau}\sum_k n_k^2}_{\text{temporary}},
\qquad
\mathbb{V}[C] = \sigma^2 \tau \sum_k x_k^2,$$

with `η̃ = η − γτ/2`. Three observations before any optimisation:

1. **`½γX²` does not depend on the schedule.** Half the cost of a large order
   is decided the moment somebody decides to trade it. No slicing touches it.
   That is the single most useful thing the model says, and it is why desks
   argue about parent size rather than about algorithm choice.
2. **`ε∑|n_k|` is also schedule-independent** for a one-directional programme.
   It stops being so the moment the schedule is allowed to change sign, which
   is exactly why unconstrained versions with drift produce those suspicious
   buy-then-sell trajectories.
3. **`η̃` can go negative** if `τ` is large enough. That is the discrete model
   announcing it has left its domain: it would then "prove" that infinitely
   fast trading is free. `ACParams.validate` raises instead of returning a
   number, and the web page shows the refusal rather than a chart.

Minimising `E[C] + λV[C]` subject to `x_0 = X`, `x_N = 0` gives a linear
second-order difference equation whose solution is

$$x_j = X\,\frac{\sinh\!\big(\kappa (T - t_j)\big)}{\sinh(\kappa T)},
\qquad
\cosh(\kappa\tau) = 1 + \frac{\lambda \sigma^2 \tau^2}{2\tilde\eta}.$$

`1/κ` is the urgency time scale. Two limits check the algebra and are asserted
in `tests/test_almgren_chriss.py`:

- `λ → 0` gives `κ → 0` and `x_j → X(1 - t_j/T)` — the straight line, i.e.
  **TWAP**. A risk-neutral trader should trade uniformly, and the model agrees.
- `λ → ∞` gives `κT ≫ 1` and an exponential decay — everything up front.

**Where the assumptions fail.** Linear temporary impact is the first casualty:
walking a book with finite depth is convex, not linear, and §5 measures the
exponent. Infinite elasticity is the second: the model has no term for an order
that *cannot* be filled, and past a certain size that is precisely what happens.
Constant σ is the third, though over half an hour it is the least damaging.

## 3. Method, and why this method

### 3.1 A latent price, rather than pure order flow

A book driven only by symmetric order flow has a mean-reverting mid: depth
accumulates on whichever side has been eaten, and the price is pulled back. Real
equity prices are close to martingales. This matters enormously for an execution
study, because if the price mean-reverts then *waiting is free* — the price
comes back to you — and every slow schedule looks brilliant for a reason that
has nothing to do with execution.

So quotes are placed around a latent random walk. The cost is one extra
assumption; the benefit is that the assumption is testable, and
`tests/test_simulator.py` tests it: the variance ratio between 10-second and
40-second mid changes must be near 1 (it is 0.6–1.6 across seeds), and the
realised volatility must match the name's calibration.

The alternative — a genuinely martingale book built purely from order flow,
as in the queue-reactive literature — requires state-dependent intensities that
would need intraday data to calibrate, and this project is restricted to free
daily data.

### 3.2 One scale factor per name

Every intensity is multiplied by a single factor `c` chosen so the simulated
session prints the name's real median daily volume. Uniform scaling is a pure
time change for the queueing system: multiply all arrival, cancellation and
trade rates by `c` and the stationary book shape is unchanged while volume
scales exactly by `c`. This turns per-name calibration into a one-dimensional
fixed point rather than a search over three parameters, and it is why the depth
profile does not have to be retuned per name. `tests/test_calibrate.py` asserts
the invariance directly.

What genuinely differs across names, then, is the ratio of trading speed to
price speed — which is the ratio that decides execution cost anyway.

### 3.3 Measuring impact instead of assuming it

`γ` and `η` are *not* inputs. They are estimated from the venue, and the
identification rests on a counterfactual: each session is run twice on the same
seed, once with the parent order and once without. Because the exogenous price
path is indexed by wall-clock time rather than by event count
(`MarketSimulator._exogenous`), the two runs see the identical price path even
though the order changes the event stream. Then

$$\hat\gamma = \frac{\mathbb{E}\big[\,\text{side}\cdot(S^{\text{with}}_T - S^{\text{without}}_T)\,\big]}{X}.$$

Without the pairing this is hopeless: over half an hour the price moves tens of
times the impact being measured. With it, a hundred paths suffice — the
standard errors in `results/impact_calibration.csv` are a few percent of the
estimates.

`η` then follows from the shortfall identity for a TWAP parent,
`E[IS]/X = ε + γX/2 + ηv`, by subtracting the two known terms. Note that this
*sequencing* is essential rather than stylistic: with a TWAP schedule `v = X/T`,
so size and rate are perfectly collinear and no regression of shortfall alone
can separate `γ` from `η`. A single regression reporting both has not separated
them; it has reported one number twice.

### 3.4 Common random numbers and a control variate

The differences between execution algorithms are one to a few basis points; the
path-to-path noise is tens. Two variance reductions make the comparison
possible:

- **Common random numbers.** Every algorithm in a grid cell runs on the same
  seed, so it sees the same anonymous flow and the same price path. Comparisons
  are paired, and the paired standard error is several times smaller.
- **A control variate.** The counterfactual session's flat-schedule cost has
  mean zero exactly (the baseline mid is a martingale) and correlation ≈ 0.95
  with every algorithm's realised cost. Subtracting it cuts the standard error
  of the mean by roughly an order of magnitude without touching the estimate.
  The weights are flat and identical for every algorithm, deliberately: using
  each algorithm's own fill weights would subtract exactly the timing skill the
  adaptive algorithm is supposed to demonstrate.

The raw shortfall, not the adjusted one, is what the dispersion, CVaR and
histograms report — an execution desk's cost variance genuinely does include the
market's move, and removing it would understate the tail that `λ` exists to
price.

## 4. Implementation walkthrough

### 4.1 The matching engine (`src/lob/book.py`)

Prices are integers (ticks), never floats — floating-point prices in a matching
engine produce orders that fail to match because `1.15 - 0.01 != 1.14`. Orders
rest in FIFO deques per price level, so the agent's own orders sit behind
whatever arrived first, and `queue_ahead` reports exactly how many lots are in
front:

```python
def queue_ahead(self, oid: int) -> int:
    o = self._orders.get(oid)
    book = self._bids if o.side == BUY else self._asks
    ahead = 0
    for other in book[o.price]:
        if other.oid == oid:
            return ahead
        ahead += other.qty
```

A crossing limit order executes and rests only the remainder; a market order
walks levels with an optional price cap, because an uncapped market order into a
thin book is how you print a trade 3% away from the mid. `check_invariants()`
asserts no crossed book, no stale level cache and no out-of-order queue, and a
3,000-operation fuzz test runs it.

### 4.2 Exact event timing (`src/lob/simulator.py`)

Inter-event times come from Ogata thinning rather than a discretised clock: the
limit and cancel intensities are constant between events, and the Hawkes
intensity only decays, so the intensity immediately after the last event bounds
the next interval. The subtlety worth documenting is what happens at the end of
a `run_until` call:

```python
e = self._pending_e if self._pending_e is not None else -math.log(self._uniform())
t_new = self.t + e / bound
if t_new >= horizon:
    self._pending_e = max(e - (horizon - self.t) * bound, 0.0)   # carry the residual
    ...
```

The unused part of the exponential wait is carried across calls, in units of the
standard exponential so that it stays valid when the rate changes. Discarding
and redrawing would be statistically harmless — exponential waits are
memoryless — but it would make the event stream depend on *how the caller
chopped up time*, and the baseline run (slice boundaries) and the execution run
(chunk boundaries within each slice) chop it differently. Carrying the residual
is what keeps a paired experiment paired. `test_stepping_in_pieces_equals_one_step`
pins it.

### 4.3 Stale quotes

The first working version of the simulator produced a 15-tick spread on a name
that quotes 2. The cause: cancellation intensity was measured from the current
best rather than from the efficient price, and it *decreases* with distance, so
quotes the price had walked away from lived forever. A $495 name moves hundreds
of ticks in half an hour while its quoted book is a handful of ticks deep, so
that stale liquidity accumulated until the book was mostly fiction. Measuring
distance from the reference price and giving out-of-book quotes a high hazard
fixed it — the spread fell to 1–2 ticks and depth at the touch to 300–600
shares, both of which are right for these names, and neither of which was
targeted.

### 4.4 Child orders are not slices

A slice is not sent as one order:

```python
per = tau / chunks_per_slice
for c in range(chunks_per_slice):
    if pending >= lot:
        got = sim.agent_market(side, want_chunk, max_ticks_through=...)
        pending -= sum(f.shares for f in got)
    sim.run_until(min(t_start + (c + 1) * per, t_start + tau))
```

The displayed book holds a few hundred shares at the touch. A whole one-minute
slice fired at once sweeps ten levels, pays the tail of the book, and then stops
when it runs out of quotes — leaving the parent unfilled for reasons that are an
artefact of the implementation rather than of the market. Real algorithms space
child orders so that liquidity replenishes between them, and the remainder of
each chunk rolls into the next.

### 4.5 The JavaScript port

The web page recomputes the Almgren–Chriss solution live rather than looking it
up, which means a second implementation of the same maths — a liability unless
it is pinned. `scripts/check_page.py` extracts the port from `index.html`, runs
it under Node against 20 parameter sets, and compares against the Python
library. Worst relative error at the time of writing: **5.8e-15**, i.e. floating
point noise. `κ` is computed through `acosh` in both, not the
`κ² ≈ λσ²/η̃` shortcut, which differs by several percent at these interval
lengths.

## 5. Validation

Every claim below is a number produced by `python scripts/make_results.py` and
stored in `results/`.

**The matching engine.** 21 tests in `tests/test_book.py`: FIFO within a price,
price priority across prices, crossing limits, partial cancels preserving
priority, uniform-over-lots cancellation (a 9-lot order is hit ~9× as often as a
1-lot order), and a fuzz test that runs 3,000 random operations and then asserts
the invariants.

**The Hawkes process.** Simulated event rate matches the theoretical
`μ₀/(1−n)` within sampling error; `α = 0` collapses to Poisson with
coefficient of variation 1; the recursive log-likelihood matches a direct
O(n²) computation to 1e-9; the true parameters beat perturbed ones on a long
sample; MLE recovers the branching ratio to ±0.12; and time-rescaling residuals
are unit-exponential under the true model and detectably not under a wrong one.

**The closed form.** `λ→0` reproduces TWAP exactly. `κ` satisfies its defining
`cosh` equation to 1e-12. The closed-form mean and standard deviation match a
40,000-path Monte Carlo of the model's own dynamics to 2% and 3%. The optimum
beats 25 random endpoint-preserving perturbations of the trajectory. The
frontier is convex and monotone. Degenerate inputs (zero shares, negative time,
zero `η`, negative `η̃`) raise rather than returning numbers.

**The venue against the real names.** See `results/venue_calibration.csv`. After
calibration each name's simulated session reproduces its median daily volume to
within a few percent and its realised volatility to within a few percent, while
the spread (1–2 ticks) and the depth at the touch (roughly 300–600 shares)
*emerge* from the flow model rather than being targeted — and land where these
names actually quote.

**The impact measurement.** The recovered `γ` is compared against the Kyle
coefficient the venue was built with; the ratio is reported in the run log and
is around 85–90%, with the shortfall attributable to measuring the mid rather
than the latent price. `tests/test_calibrate.py` asserts this recovery
end-to-end as a test. The linear cost model fits the participation sweep with
R² ≈ 0.9, and the fitted temporary-impact exponent is reported alongside — the
model assumes 1.

**The spread estimators.** Both Corwin–Schultz and Abdi–Ranaldo are validated
against a synthetic Roll model where the true spread is known, and both recover
it. Both then fail on real large caps by one to two orders of magnitude, in
opposite directions. That failure is a reported result (it is a panel on the web
page), and it is why the project does not take its spread from daily bars.

## 6. Results and interpretation

The headline numbers are in `results/tournament_summary.csv`, and the run log
`results/make_results.log` prints the comparison table. Read them as follows.

**Half of a large order's cost is untouchable.** The permanent-impact term is
the same for every algorithm, so once the parent size is fixed, the entire
argument between execution algorithms is over the temporary part. On the cost
decomposition panel, at a normal institutional size the permanent share is the
largest single block. The operational consequence is that the highest-value
conversation is with the portfolio manager about size, not with the vendor about
algorithms.

**Between algorithms, the differences are small, real, and easily mistaken for
noise.** On a single order the spread of outcomes is an order of magnitude
larger than the difference between algorithms — which is exactly why execution
quality is judged over hundreds of orders and never over one. This is visible in
the histogram panel, where the distributions overlap almost completely. The
paired tests in `results/paired_vs_twap.csv` are what separate signal from luck,
and they only work because of the common random numbers.

**The model understates the cost of urgency.** Comparing the Almgren–Chriss
prediction against the simulated cost of the *same* schedule
(`results/frontier_sim.csv`), the agreement is good for patient schedules and
deteriorates as `λ` rises. The reason is structural: the closed form charges
`ηv` per share for trading at rate `v`, a linear function, while walking a
finite book is convex. An aggressive schedule is therefore cheaper on paper than
in the book, and the gap grows with urgency.

**Past a certain size, the model stops describing anything.** The capacity
panel shows fill rate against parent size. Below roughly half a percent of a
day's volume in half an hour everything completes and costs grow smoothly. Past
that, the venue cannot supply the shares inside the horizon: the fill rate
falls, the unfilled remainder is charged at whatever the price did, and the
closed form — which has no term for an unfillable order — gives an answer that
is not merely inaccurate but meaningless.

**Across names, cost is a story about volatility over liquidity.** The
cross-sectional panel executes the same fraction of each name's ADV in each
name's calibrated venue. What separates the expensive names from the cheap ones
is not price or sector but the ratio of dollar volatility to volume — which is
the same quantity Kyle's λ is built from, and a reassuring consistency check on
the whole apparatus.

## 7. Limitations and failure modes

1. **The venue is a model, not a tape.** It reproduces daily volume,
   volatility, a plausible spread and a plausible depth profile, but it has no
   intraday data behind it. Its book is single-venue, its flow has no informed
   traders, and there is no queue-position gaming, no hidden liquidity, no odd
   lots and no auctions. Numbers from it are indicative of magnitude, not
   forecasts of a specific desk's costs.
2. **Impact is linear-permanent by construction.** The Kyle term is built into
   the simulator, so the measured `γ` partly reflects a modelling choice. The
   honest reading of §5's recovery test is "the estimator works", not "real
   permanent impact is linear". The empirical literature says it is closer to a
   square root in size.
3. **The impact parameters are measured on one name and scaled to the others**
   by `Pσ/ADV`. That is a dimensional argument, not a measurement.
4. **Cross-sectional and time-series σ are constant** within a run. Real
   execution risk is worst precisely when volatility spikes, and the vol
   multiplier on the grid is a coarse stand-in for that.
5. **The passive child-order mode is simplistic**: post at the touch, then
   clean up aggressively. It has no repricing logic and no queue-position
   management, so it will understate what a good passive algorithm achieves and
   overstate its adverse selection.
6. **No signal, no alpha decay.** Real execution decisions are made against an
   alpha that decays; urgency in production is usually set by how fast the
   signal decays, not by a risk-aversion parameter. That is the most important
   thing missing from the objective function here.

How you would know in production: monitor realised shortfall against the model's
prediction, bucketed by participation rate. A widening gap in the high-
participation buckets is exactly the failure mode §6 describes, and it shows up
long before anything else does.

## 8. Extensions

- **Queue-reactive intensities** (Huang, Lehalle and Rosenbaum, 2015): make
  arrival and cancellation rates depend on the current queue sizes, which
  produces a martingale mid without needing the latent-price device in §3.1.
- **Almgren–Lorenz adaptive execution solved properly** as a dynamic program,
  rather than the re-solve-and-tilt heuristic implemented here, so the value of
  adaptivity can be measured against its true optimum.
- **A square-root impact model** fitted to the venue instead of the linear one,
  and the corresponding numerically-solved trajectory, which would let the
  model track the venue into the high-participation region where it currently
  breaks.
- **Alpha-aware execution**: add a decaying signal to the objective, which is
  what actually sets urgency on a real desk.
- **Multi-venue routing**, with a fee/rebate structure — where a large part of
  real execution engineering effort goes.
- **Calibration against a real tape** (LOBSTER sample data, or a day of Nasdaq
  ITCH), which would replace the assumed order-size and arrival-rate parameters
  with measured ones.

## 9. References

- Almgren, R. and Chriss, N. (2000). "Optimal execution of portfolio
  transactions." *Journal of Risk* 3(2), 5–39.
- Almgren, R. and Lorenz, J. (2007). "Adaptive arrival price." *Algorithmic
  Trading III*, Institutional Investor.
- Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005). "Direct estimation of
  equity market impact." *Risk* 18(7), 58–62.
- Bacry, E., Mastromatteo, I. and Muzy, J.-F. (2015). "Hawkes processes in
  finance." *Market Microstructure and Liquidity* 1(1).
- Bouchaud, J.-P., Bonart, J., Donier, J. and Gould, M. (2018). *Trades, Quotes
  and Prices: Financial Markets Under the Microscope.* Cambridge University
  Press.
- Cont, R., Stoikov, S. and Talreja, R. (2010). "A stochastic model for order
  book dynamics." *Operations Research* 58(3), 549–563.
- Corwin, S. and Schultz, P. (2012). "A simple way to estimate bid-ask spreads
  from daily high and low prices." *Journal of Finance* 67(2), 719–760.
- Abdi, F. and Ranaldo, A. (2017). "A simple estimation of bid-ask spreads from
  daily close, high, and low prices." *Review of Financial Studies* 30(12),
  4437–4480.
- Hasbrouck, J. (2007). *Empirical Market Microstructure.* Oxford University
  Press.
- Huang, W., Lehalle, C.-A. and Rosenbaum, M. (2015). "Simulating and analyzing
  order book data: the queue-reactive model." *JASA* 110(509), 107–122.
- Kyle, A. (1985). "Continuous auctions and insider trading." *Econometrica*
  53(6), 1315–1335.
- Ogata, Y. (1981). "On Lewis' simulation method for point processes." *IEEE
  Transactions on Information Theory* 27(1), 23–31.
- Perold, A. (1988). "The implementation shortfall: paper versus reality."
  *Journal of Portfolio Management* 14(3), 4–9.
- Roll, R. (1984). "A simple implicit measure of the effective bid-ask spread in
  an efficient market." *Journal of Finance* 39(4), 1127–1139.
