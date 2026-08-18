"""Exponential-kernel Hawkes process: clustered market-order arrivals.

Trades do not arrive like raindrops. A trade makes the next trade more likely -
because a large parent order is being sliced, because the print triggers other
people's signals, because a level breaking pulls in momentum flow. Poisson
arrivals miss all of that, and they miss it in the direction that matters for
execution: they understate the probability of the ten minutes in which the
book empties out while you still have half your order to work.

A one-dimensional Hawkes process with an exponential kernel,

    lambda(t) = mu0 + sum_{t_i < t} alpha * exp(-beta (t - t_i)),

is the smallest model that produces that clustering, and it is analytically
tractable enough to check. Two properties do most of the work here:

**Branching ratio** ``n = alpha / beta`` is the expected number of children per
event. ``n < 1`` is required for stationarity, and

    E[lambda] = mu0 / (1 - n)

which is the identity :func:`stationary_intensity` returns and the tests pin
the simulator against. Empirical estimates of ``n`` on equity trade flow sit
around 0.6-0.9; the default here is 0.46, which is deliberately conservative -
this project uses the clustering to stress an execution schedule, not to make
the biggest number it can.

**The two-sided version.** Buy and sell trades excite themselves and, more
weakly, each other. :class:`BivariateHawkes` handles that with one
cross-excitation coefficient, which is enough to produce runs of same-side
trades without inventing a full 2x2 kernel matrix that nothing in this project
would calibrate.

Simulation is Ogata thinning, which is exact rather than a discretised
approximation: the exponential kernel means the intensity only *decreases*
between events, so the intensity at the last event is a valid upper bound for
the next inter-arrival draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ExpHawkes:
    """Univariate Hawkes with kernel ``alpha * exp(-beta t)``."""

    mu0: float
    alpha: float
    beta: float

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta if self.beta > 0 else np.inf

    @property
    def is_stationary(self) -> bool:
        return self.branching_ratio < 1.0

    def stationary_intensity(self) -> float:
        """``mu0 / (1 - n)``. Infinite for an explosive process."""
        n = self.branching_ratio
        if n >= 1.0:
            return float("inf")
        return self.mu0 / (1.0 - n)

    def intensity(self, t: float, history: np.ndarray) -> float:
        h = np.asarray(history, dtype=float)
        h = h[h < t]
        if h.size == 0:
            return self.mu0
        return float(self.mu0 + self.alpha * np.exp(-self.beta * (t - h)).sum())

    def simulate(self, T: float, rng: np.random.Generator,
                 t0: float = 0.0) -> np.ndarray:
        """Event times on ``[t0, t0+T)`` by Ogata thinning.

        The running state ``excite`` is the summed kernel, decayed forward
        between events, so the whole simulation is O(number of events) rather
        than O(events^2) - the difference between a second and a minute at the
        rates this project runs at.
        """
        out: List[float] = []
        t = t0
        end = t0 + T
        excite = 0.0        # summed kernel, always decayed to exactly ``t``
        while t < end:
            bound = self.mu0 + excite     # intensity cannot rise before the
            if bound <= 0:                # next event, so this bounds it
                break
            dt = rng.exponential(1.0 / bound)
            t = t + dt
            if t >= end:
                break
            excite = excite * np.exp(-self.beta * dt)
            if rng.random() <= (self.mu0 + excite) / bound:
                out.append(t)
                excite += self.alpha
        return np.asarray(out, dtype=float)

    def loglik(self, times: np.ndarray, T: float, t0: float = 0.0) -> float:
        """Exact log-likelihood, using the recursive form of the kernel sum.

            l = sum_i log lambda(t_i) - integral_{t0}^{t0+T} lambda(u) du

        The compensator term integrates the kernel in closed form; doing it
        numerically is the usual way this gets slightly, invisibly wrong.
        """
        t = np.asarray(times, dtype=float)
        if t.size == 0:
            return -self.mu0 * T
        if self.alpha < 0 or self.beta <= 0 or self.mu0 <= 0:
            return -np.inf
        r = 0.0
        ll = 0.0
        prev = t[0]
        ll += np.log(self.mu0)
        for i in range(1, t.size):
            r = (1.0 + r) * np.exp(-self.beta * (t[i] - prev))
            prev = t[i]
            ll += np.log(self.mu0 + self.alpha * r)
        compensator = self.mu0 * T + (self.alpha / self.beta) * float(
            np.sum(1.0 - np.exp(-self.beta * (t0 + T - t))))
        return float(ll - compensator)

    def residuals(self, times: np.ndarray, t0: float = 0.0) -> np.ndarray:
        """Time-rescaling residuals ``Lambda(t_i) - Lambda(t_{i-1})``.

        Under the true model these are i.i.d. Exponential(1). This is the
        goodness-of-fit test for a point process, and it is the one thing that
        distinguishes "I fitted a Hawkes process" from "I checked whether the
        data is one".
        """
        t = np.asarray(times, dtype=float)
        if t.size < 2:
            return np.zeros(0)
        out = np.empty(t.size - 1)
        # Lambda(t) = mu0 t + (alpha/beta) sum_i (1 - exp(-beta (t - t_i)))
        for k in range(1, t.size):
            prev, cur = t[k - 1], t[k]
            past = t[:k]
            comp_cur = self.mu0 * (cur - t0) + (self.alpha / self.beta) * float(
                np.sum(1.0 - np.exp(-self.beta * (cur - past))))
            past_prev = t[:k - 1]
            comp_prev = self.mu0 * (prev - t0) + (self.alpha / self.beta) * float(
                np.sum(1.0 - np.exp(-self.beta * (prev - past_prev))))
            out[k - 1] = comp_cur - comp_prev
        return out


def fit_mle(times: np.ndarray, T: float, t0: float = 0.0,
            x0: Optional[Tuple[float, float, float]] = None) -> ExpHawkes:
    """Maximum-likelihood fit of ``(mu0, alpha, beta)``.

    Optimised in log-space on all three parameters. That is not cosmetic: the
    parameters are positive and span orders of magnitude, the likelihood is
    badly scaled in ``beta``, and an unconstrained optimiser handed raw
    parameters will happily propose ``beta < 0`` and return the boundary. The
    stationarity constraint ``alpha < beta`` is imposed by parameterising
    ``alpha = n * beta`` with ``n = sigmoid(z)``.
    """
    from scipy.optimize import minimize

    t = np.asarray(times, dtype=float)
    if t.size < 10:
        raise ValueError("need at least 10 events to fit a Hawkes process")
    rate = t.size / T
    if x0 is None:
        x0 = (0.5 * rate, 0.5, 1.0)
    mu0_0, n_0, beta_0 = x0[0], min(0.9, max(0.05, x0[1] / max(x0[2], 1e-9))), x0[2]

    def unpack(z):
        mu0 = np.exp(z[0])
        n = 1.0 / (1.0 + np.exp(-z[1]))
        beta = np.exp(z[2])
        return mu0, n * beta, beta

    def nll(z):
        mu0, alpha, beta = unpack(z)
        if not np.isfinite(mu0 + alpha + beta):
            return 1e12
        val = ExpHawkes(mu0, alpha, beta).loglik(t, T, t0)
        return -val if np.isfinite(val) else 1e12

    z0 = np.array([np.log(mu0_0), np.log(n_0 / (1 - n_0)), np.log(beta_0)])
    res = minimize(nll, z0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    mu0, alpha, beta = unpack(res.x)
    return ExpHawkes(float(mu0), float(alpha), float(beta))


@dataclass
class BivariateHawkes:
    """Buy and sell trade arrivals that excite themselves and each other.

    ``cross`` in [0, 1] splits each event's excitation: ``1 - cross`` of it goes
    to the same side, ``cross`` to the other. ``cross = 0`` is two independent
    processes and produces long one-sided runs; ``cross = 1`` makes every trade
    provoke the opposite side and produces alternation. Equity data sits nearer
    the first, which is why the default is 0.25.

    This class carries mutable state (the two decayed excitation sums) because
    the simulator interleaves it with book events and cannot generate all the
    arrival times up front - the agent's own trades feed back into the
    intensity.
    """

    mu0: float
    alpha: float
    beta: float
    cross: float = 0.25
    _excite_buy: float = 0.0
    _excite_sell: float = 0.0
    _t: float = 0.0

    def reset(self, t0: float = 0.0) -> None:
        self._excite_buy = 0.0
        self._excite_sell = 0.0
        self._t = t0

    def decay_to(self, t: float) -> None:
        dt = t - self._t
        if dt <= 0:
            return
        f = np.exp(-self.beta * dt)
        self._excite_buy *= f
        self._excite_sell *= f
        self._t = t

    def intensities(self, t: float) -> Tuple[float, float]:
        self.decay_to(t)
        return self.mu0 + self._excite_buy, self.mu0 + self._excite_sell

    def excite(self, t: float, side: int) -> None:
        """Register a trade. ``side`` is +1 for a buy, -1 for a sell."""
        self.decay_to(t)
        same = self.alpha * (1.0 - self.cross)
        other = self.alpha * self.cross
        if side > 0:
            self._excite_buy += same
            self._excite_sell += other
        else:
            self._excite_sell += same
            self._excite_buy += other

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta if self.beta > 0 else np.inf

    def stationary_intensity(self) -> float:
        """Per side. Symmetry means the cross term redistributes excitation but
        does not change the total, so the univariate identity still holds."""
        n = self.branching_ratio
        return self.mu0 / (1.0 - n) if n < 1 else float("inf")


__all__ = ["ExpHawkes", "BivariateHawkes", "fit_mle"]
