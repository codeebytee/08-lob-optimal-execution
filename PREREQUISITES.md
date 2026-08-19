# Prerequisites — what you need to know to read this repo

Written for someone who can program but has never worked in finance. No prior
knowledge of markets is assumed. If you can follow "a queue where people wait
their turn", you can follow all of it.

---

## 1. The idea, in one paragraph

When a pension fund decides to buy a million shares of Microsoft, it cannot
just buy them. There is nobody standing there offering a million shares at
today's price; there are a few hundred shares offered at the best price, a few
hundred more slightly higher, and so on. Buy aggressively and you walk up that
ladder, paying more and more, and other traders notice and move their prices
away from you before you finish. Buy slowly and you avoid that, but now you are
exposed for hours to the price simply drifting away from you for reasons that
have nothing to do with your order. Every large trade is that trade-off: **pay
to go fast, or take risk by going slow.** This project builds a working
simulation of the market you would be trading into, implements the standard
mathematical solution to the trade-off, and then checks — honestly — whether
that solution's predictions survive contact with the simulation.

---

## 2. An analogy

Think of buying every ticket for a concert from a resale site.

The site lists a handful of tickets at $100, a few more at $102, some at $105.
That list is the **order book**. If you buy fifty tickets right now, you clear
out the $100 offers, then the $102 offers, and end up averaging $103 — you have
"walked the book", and the difference between $103 and the $100 you saw when
you decided to buy is your **execution cost**.

But you could instead buy five tickets an hour for ten hours. Sellers keep
posting new tickets at around $100, so you mostly get the cheap ones and you
never visibly clear out a price level. That is cheaper — but over ten hours the
band might announce a stadium tour and every remaining ticket reprices to $130.
You saved a few dollars a ticket and lost thirty.

That is the entire problem. The mathematics in this repo is a precise version
of "how fast should I buy", given how thin the listings are and how jumpy the
price is.

---

## 3. Concepts, in dependency order

Each of these is needed by the one after it.

**Bid, ask, mid, spread.** At any moment the market shows the highest price
someone will pay (the **bid**) and the lowest price someone will sell at (the
**ask**). The gap between them is the **spread**, and the average of the two is
the **mid**. Nobody trades at the mid; a buyer pays the ask, a seller receives
the bid, so crossing the spread costs you half of it either way.
*Why it matters here:* half the spread is the irreducible cost of trading
immediately, and it is the first term in every cost model in this repo.

**Limit order and market order.** A **limit order** says "buy me 200 shares, but
never above $100" — it waits in a queue until someone trades with it, and it
might wait forever. A **market order** says "buy me 200 shares now, whatever
they cost" — it executes immediately against whatever limit orders are sitting
there. Limit orders supply liquidity; market orders consume it.
*Why it matters here:* an execution algorithm's basic choice is how much to
demand immediately versus how much to sit and wait for.

**The limit order book, and queue priority.** All the resting limit orders,
organised by price, are the **book**. Within one price, orders are filled in the
order they arrived — **price-time priority**, exactly like a queue at a counter.
Get there first and you get filled first.
*Why it matters here:* the difference between a real book and a cartoon one is
the queue. If you post an order behind 5,000 shares, you will probably never
trade, and any simulation that ignores queue position will tell you passive
trading is far easier than it is. `src/lob/book.py` implements the real thing.

**ADV, and order size as a fraction of it.** **Average daily volume** is how
many shares change hands in a day. Order size is always quoted relative to it,
because 100,000 shares is nothing in Apple and enormous in a small company.
*Why it matters here:* every result in this project is stated as a percentage
of ADV, so it transfers between names.

**Volatility.** How much the price moves, per unit of time, measured as a
standard deviation. It scales with the square root of time: if a price moves 1%
in a day, it moves about 1%/√13 ≈ 0.28% in the last half hour, because there
are roughly thirteen half-hours in a trading day.
*Why it matters here:* this square-root law is exactly why waiting is risky,
and it is one side of the central trade-off.

**Market impact, permanent and temporary.** Your buying pushes the price up.
Part of that push is **temporary** — you exhausted the offers at the touch, and
liquidity refills a moment later, so the price comes back. Part of it is
**permanent** — other participants infer that somebody wants to buy and reprice
accordingly, and that part never comes back.
*Why it matters here:* the two behave completely differently. The permanent part
depends only on how much you trade in total, so no clever scheduling touches it.
The temporary part depends on how fast you trade, and is the only thing a
schedule can control. Confusing them is the single most common error in this
area.

**Implementation shortfall.** The standard measure of execution quality: the
average price you actually paid, minus the mid price at the moment you decided
to trade, times the number of shares. Positive means the trade cost you money
relative to the decision. It includes **opportunity cost** — shares you failed
to buy are charged at wherever the price ended up, so failing to complete is not
a way of looking cheap.
*Why it matters here:* it is the number every chart in this project reports.

**TWAP, VWAP, POV.** Three standard schedules. **TWAP** buys the same amount
every minute (time-weighted). **VWAP** buys more when the market is normally
busier — mornings and afternoons — to match the volume-weighted average price.
**POV** (percent of volume) buys a fixed fraction of whatever is trading, so it
speeds up when the market is active.
*Why it matters here:* these are the benchmarks. A model-based schedule that
cannot beat TWAP is not worth running.

**The mean-variance trade-off, and risk aversion λ.** You cannot minimise both
expected cost and cost uncertainty. So you minimise `expected cost + λ ×
variance of cost`, where λ says how much you dislike uncertainty. λ = 0 means
you only care about the average; large λ means you want to be done and will pay
for it.
*Why it matters here:* λ is the single knob in the Almgren–Chriss solution, and
the "urgency" slider on the web page is exactly this.

**The efficient frontier.** Sweep λ from 0 to large, plot expected cost against
cost standard deviation, and you get a curve. Every point on it is optimal for
somebody. Nothing exists below and to the left of it.
*Why it matters here:* it is how the answer is presented, and it is what lets
you compare schedules that were never designed with λ in mind.

**Poisson and Hawkes arrivals.** A **Poisson process** models events arriving
independently at a constant average rate — raindrops. A **Hawkes process**
models events that make more events more likely: each arrival temporarily
raises the rate, so events come in bursts. Real trades cluster like the second,
not the first.
*Why it matters here:* clustered flow means the quiet ten minutes when your
order needs liquidity and there is none are far more likely than a Poisson model
would suggest. That is the tail risk an execution desk cares about.

**Monte Carlo, and common random numbers.** **Monte Carlo** means: simulate the
thing thousands of times and look at the distribution of outcomes. **Common
random numbers** means running every algorithm on the *same* thousand simulated
days, so when you compare them you are not comparing luck. It is the same idea
as an A/B test on the same users rather than on different ones.
*Why it matters here:* the difference between two execution algorithms is a
basis point or two, while the day-to-day noise is fifty. Without pairing, the
comparison is hopeless.

**Control variate.** A variance-reduction trick: if you know a quantity that is
correlated with your noisy estimate and whose true average you know exactly,
subtract it. The noise cancels and the average is unchanged.
*Why it matters here:* it turns "these two algorithms differ by 1.4 bp, t =
0.3" into a usable measurement, with no extra simulation.

---

## 4. Glossary

| Term | Meaning |
|---|---|
| ADV | Average daily volume, in shares |
| Ask / offer | Lowest price anyone is currently willing to sell at |
| Basis point (bp) | One hundredth of a percent. 10 bp on $100 is one cent |
| Bid | Highest price anyone is currently willing to buy at |
| Child order | One slice of a large parent order, sent to the market |
| CVaR 95% | Average cost on the worst 5% of outcomes |
| Efficient (or latent) price | The "true" price the market is quoting around, unobservable |
| Fill | An executed trade against your order |
| Implementation shortfall | Average paid minus arrival mid, times shares; the cost measure |
| Lot | 100 shares; the unit of size in the simulated book |
| Mid | Midpoint of bid and ask |
| Parent order | The whole order to be worked, e.g. "buy 250,000 shares" |
| Participation rate | Your volume divided by total market volume over a period |
| Queue position | How many shares are ahead of yours at the same price |
| Slice / interval | One time step of the schedule; here, one minute |
| Spread | Ask minus bid |
| Tick | The minimum price increment; one cent for US stocks above $1 |
| Touch | The best bid and best ask; "depth at the touch" is size there |
| VWAP | Volume-weighted average price over a period |

Symbols used in `DEEP_DIVE.md`:

| Symbol | Meaning |
|---|---|
| X | Total shares in the parent order |
| T | Time allowed, in seconds |
| N, τ | Number of intervals, and their length T/N |
| xₖ | Shares still to trade at the start of interval k |
| nₖ | Shares traded during interval k |
| v | Trading rate, shares per second |
| σ | Volatility, in dollars per square root of a second |
| γ | Permanent impact: dollars of price move per share traded |
| η | Temporary impact: extra dollars per share, per share/second of rate |
| ε | Fixed cost per share — half the spread, plus fees |
| λ | Risk aversion, in inverse dollars |
| κ | Urgency scale, in inverse seconds; 1/κ is roughly how long the order takes |
| μ, α, β | Hawkes baseline rate, excitation size, and decay rate |

---

## 5. The maths, honestly labelled

**Essential:**
- Expectation and variance, and that variance of a sum of independent things
  adds. Everything in the cost model is built from this.
- The idea that a random walk's standard deviation grows like √t.
- Being comfortable reading a formula and asking "what happens when this term
  goes to zero".

**Helpful but skippable on a first pass:**
- Solving a linear difference equation. This is where the `sinh` in the optimal
  trajectory comes from. You can accept the answer and check its limits instead
  — the tests do exactly that.
- Lagrange multipliers, for how the endpoint constraint (finish the order)
  enters the optimisation.
- Point processes and intensities, for the Hawkes model. The one-sentence
  version — "each event raises the arrival rate, and the rise decays away" — is
  enough to follow everything the simulator does.

**Not needed at all:** stochastic calculus. The whole model here is discrete;
there is not an Itô integral anywhere in the code.

---

## 6. Where to learn more

1. **Almgren & Chriss, "Optimal execution of portfolio transactions" (2000)** —
   the source paper, and unusually readable. Sections 1–3 contain everything
   used here. Free from the authors' pages.
2. **Robert Almgren, "Execution costs" (2009, *Encyclopedia of Quantitative
   Finance*)** — six pages, no prerequisites, and the best short description of
   what impact actually is.
3. **Cont, Stoikov & Talreja, "A stochastic model for order book dynamics"
   (2010)** — the flow model this simulator's arrival intensities come from.
   Read Section 2 for the setup.
4. **Bouchaud, Bonart, Donier & Gould, *Trades, Quotes and Prices* (2018),
   Chapter 3** — the modern reference on market microstructure. Chapter 3 alone
   covers the order book mechanics used in `src/lob/`.
5. **Hasbrouck, *Empirical Market Microstructure* (2007), Chapters 3 and 5** —
   for where Kyle's λ comes from and why price impact is linear in the simplest
   models.
6. **Bacry, Mastromatteo & Muzy, "Hawkes processes in finance" (2015)** — a
   survey; Section 2 is the gentlest introduction to self-exciting arrivals.

---

## 7. How to read the rest of this repo

1. **This file**, which you have just finished.
2. **The interface** — open `docs/index.html` by double-clicking it. Start on
   the "Live book" tab and press play; that is the whole problem in one
   picture. Then move the urgency slider on the "Schedules" tab.
3. **`README.md`** for the headline results and the design decisions.
4. **`DEEP_DIVE.md`** for the mathematics, the validation evidence, and the
   places the model fails.
5. **The code**, in this order: `src/lob/book.py` (the matching engine),
   `src/lob/simulator.py` (the market around it), `src/execution/almgren_chriss.py`
   (the closed form), `src/execution/runner.py` (how a cost gets measured), and
   `src/flow/calibrate.py` (how the two halves are tied together).
6. **`notebooks/execution_story.ipynb`** if you would rather see the results
   built up step by step than read about them.
