# LinkedIn post draft

Almgren–Chriss did not beat TWAP on cost. It was never supposed to.

I built an event-driven limit order book — FIFO matching, Hawkes order flow,
quotes anchored to a latent price with a Kyle impact term — calibrated it to
eight real names, and ran five execution algorithms through it on common random
numbers, so the comparisons are paired rather than anecdotal.

At half a percent of ADV over thirty minutes, TWAP was cheapest on average: 1.1
bp of shortfall against 5.7 bp for the most risk-averse Almgren–Chriss schedule,
a gap that survives a paired t-test. What Almgren–Chriss bought was the tail —
95% CVaR fell from 51.8 bp to 33.4 bp and dispersion nearly halved. That is the
trade the λ parameter actually makes, and a mean-shortfall table hides it.

The simulator also found the edge of the model: somewhere between 1% and 2% of a
day's volume in half an hour, the fill rate drops from 99% to 64%, and a closed
form with no term for an unfillable order returns a number that means nothing.

Everything runs in the browser, offline:
<!-- fill in after enabling Pages -->
https://codeebytee.github.io/08-lob-optimal-execution/
