"""Almgren-Chriss optimal execution, in the discrete form, solved exactly.

The problem. A parent order of ``X`` shares must be finished by time ``T``.
Trading it fast pays impact; trading it slowly leaves the unexecuted remainder
exposed to volatility. Almgren and Chriss (2000) make that tradeoff explicit by
choosing the schedule that minimises

    E[cost] + lambda * Var[cost].

Setup. Split ``[0, T]`` into ``N`` intervals of length ``tau = T/N``; let
``x_k`` be the shares still to trade at time ``t_k = k tau`` (so ``x_0 = X``,
``x_N = 0``) and ``n_k = x_{k-1} - x_k`` the shares traded in interval ``k`` at
rate ``v_k = n_k / tau``. The price follows

    S_k = S_{k-1} + sigma sqrt(tau) xi_k - tau * g(v_k)          (permanent)
    Shat_k = S_{k-1} - h(v_k)                                    (execution)

with linear impact ``g(v) = gamma v`` and ``h(v) = epsilon sgn(v) + eta v``.
Then the implementation shortfall relative to the arrival price has

    E[C] = (gamma/2) X^2 + epsilon sum |n_k| + (eta_tilde / tau) sum n_k^2
    Var[C] = sigma^2 tau sum x_k^2 ,     eta_tilde = eta - gamma tau / 2

Three things are worth noticing before any optimisation happens.

*The permanent-impact term ``gamma X^2 / 2`` does not depend on the schedule.*
Half of what a large order costs is decided the moment you decide to trade it;
no amount of clever slicing touches it. That is the single most useful thing
this model says, and it is why desks argue about parent size, not just about
algorithms.

*``epsilon sum |n_k|`` is also schedule-independent* for a pure-buy programme -
crossing the spread on every share costs the same whatever order you do it in.
It stops being independent the moment the schedule is allowed to change sign,
which is exactly why unconstrained Almgren-Chriss with drift produces those
suspicious buy-then-sell trajectories.

*``eta_tilde = eta - gamma tau / 2`` can go negative* if you make the intervals
long enough. That is the discrete model telling you it has left its domain of
validity: it would then "prove" that infinitely fast trading is optimal. The
code refuses rather than returning a number.

The solution. Minimising ``E + lambda Var`` subject to the endpoint conditions
gives a linear second-order difference equation whose solution is a hyperbolic
sine,

    x_j = X * sinh(kappa (T - t_j)) / sinh(kappa T),

where ``kappa`` solves ``cosh(kappa tau) = 1 + lambda sigma^2 tau^2 / (2
eta_tilde)``. ``1/kappa`` is the *urgency time scale*: the order is
substantially finished after a few multiples of it. Two limits check the
algebra, and both are asserted in the tests:

* ``lambda -> 0``: ``kappa -> 0`` and ``sinh`` degenerates to its argument, so
  ``x_j -> X (1 - t_j/T)`` - the linear trajectory, i.e. TWAP. A risk-neutral
  trader should trade uniformly, and the model agrees.
* ``lambda -> infinity``: ``kappa T >> 1`` and the trajectory decays like
  ``e^{-kappa t}`` - everything up front.

Everything in this module is closed form and cheap, which is why it is the half
of the project that the web page recomputes live in JavaScript rather than
looking up from a precomputed grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ACParams:
    """Parameters in absolute units: shares, seconds, dollars.

    ``sigma`` is *arithmetic* volatility in dollars per square-root-second, not
    a percentage. Mixing the two is the most common way to get an
    Almgren-Chriss implementation silently wrong by a factor of the price.
    """

    X: float                 # shares to execute (positive for a buy)
    T: float                 # horizon, seconds
    N: int                   # number of intervals
    sigma: float             # $ per sqrt(second)
    eta: float               # temporary impact, $ per share per (share/second)
    gamma: float             # permanent impact, $ per share per share
    epsilon: float = 0.0     # fixed cost per share (half spread + fees)

    @property
    def tau(self) -> float:
        return self.T / self.N

    @property
    def eta_tilde(self) -> float:
        return self.eta - 0.5 * self.gamma * self.tau

    def validate(self) -> None:
        if self.X <= 0:
            raise ValueError("X must be positive; sell orders are handled by "
                             "the sign convention in the caller")
        if self.T <= 0 or self.N < 1:
            raise ValueError("need T > 0 and N >= 1")
        if self.sigma < 0 or self.eta <= 0 or self.gamma < 0 or self.epsilon < 0:
            raise ValueError("sigma, gamma, epsilon must be >= 0 and eta > 0")
        if self.eta_tilde <= 0:
            raise ValueError(
                f"eta_tilde = eta - gamma*tau/2 = {self.eta_tilde:.3g} <= 0: "
                "the discrete model is invalid at this interval length. "
                "Shorten tau (raise N) or check the impact calibration.")


def kappa(p: ACParams, lam: float) -> float:
    """Solve ``cosh(kappa tau) = 1 + lambda sigma^2 tau^2 / (2 eta_tilde)``.

    Computed via ``acosh`` rather than the ``kappa^2 ~ lambda sigma^2 /
    eta_tilde`` continuous approximation, because the two differ by several
    percent at the interval lengths this project uses, and the difference shows
    up directly in the cost. ``lam = 0`` short-circuits to zero: ``acosh(1)`` is
    fine, but dividing by ``tau`` after it loses precision.
    """
    if lam <= 0:
        return 0.0
    p.validate()
    tau = p.tau
    arg = 1.0 + lam * p.sigma ** 2 * tau ** 2 / (2.0 * p.eta_tilde)
    return float(math.acosh(arg) / tau)


def trajectory(p: ACParams, lam: float) -> Dict[str, np.ndarray]:
    """Optimal holdings ``x_j`` and trades ``n_k``.

    Returns ``t`` (N+1 grid times), ``x`` (N+1 shares remaining), ``n`` (N
    shares traded per interval) and ``v`` (N trading rates).
    """
    p.validate()
    k = kappa(p, lam)
    t = np.linspace(0.0, p.T, p.N + 1)
    if k <= 0:
        x = p.X * (1.0 - t / p.T)
    else:
        x = p.X * np.sinh(k * (p.T - t)) / math.sinh(k * p.T)
    x[-1] = 0.0
    n = -np.diff(x)
    return {"t": t, "x": x, "n": n, "v": n / p.tau, "kappa": k}


def expected_cost(p: ACParams, n: np.ndarray) -> float:
    """``E[shortfall]`` in dollars for an arbitrary schedule ``n``.

    Works for *any* schedule, not just the optimal one - that is what makes it
    usable as the yardstick for TWAP, VWAP and POV as well.
    """
    p.validate()
    n = np.asarray(n, dtype=float)
    perm = 0.5 * p.gamma * p.X ** 2
    fixed = p.epsilon * float(np.abs(n).sum())
    temp = (p.eta_tilde / p.tau) * float((n ** 2).sum())
    return perm + fixed + temp


def cost_variance(p: ACParams, x: np.ndarray) -> float:
    """``Var[shortfall]`` in dollars-squared, given the holdings path.

    Uses ``x_1..x_{N-1}`` - the holdings *after* each trade. The first element
    ``x_0 = X`` is excluded because it is held for zero time inside the model's
    convention, and including it is a classic off-by-one that inflates the
    variance by one interval's worth.
    """
    p.validate()
    x = np.asarray(x, dtype=float)
    return float(p.sigma ** 2 * p.tau * (x[1:] ** 2).sum())


def solve(p: ACParams, lam: float) -> Dict[str, object]:
    """Full solution at one risk aversion: schedule, cost, variance, kappa."""
    traj = trajectory(p, lam)
    ec = expected_cost(p, traj["n"])
    vc = cost_variance(p, traj["x"])
    k = float(traj["kappa"])
    return {"t": traj["t"], "x": traj["x"], "n": traj["n"], "v": traj["v"],
            "kappa": k,
            "half_life": (math.log(2.0) / k if k > 0 else math.inf),
            "expected_cost": ec, "variance": vc,
            "stdev": math.sqrt(max(vc, 0.0)),
            "objective": ec + lam * vc}


def frontier(p: ACParams, lambdas: np.ndarray) -> Dict[str, np.ndarray]:
    """The efficient frontier of expected cost against cost standard deviation.

    Every point is optimal for *some* trader; the frontier is the statement
    that you cannot have less of both. Its slope at a point is the price of
    risk that trader is implicitly paying, which is the only defensible way to
    choose lambda: pick the tradeoff you are willing to make, not the number
    that makes the backtest look best.
    """
    lam = np.asarray(lambdas, dtype=float)
    ec, sd, kap = [], [], []
    for l in lam:
        s = solve(p, float(l))
        ec.append(s["expected_cost"])
        sd.append(s["stdev"])
        kap.append(s["kappa"])
    return {"lambda": lam, "expected_cost": np.asarray(ec),
            "stdev": np.asarray(sd), "kappa": np.asarray(kap)}


def schedule_cost(p: ACParams, n: np.ndarray) -> Dict[str, float]:
    """Model cost and variance of a *given* schedule, e.g. TWAP or POV."""
    n = np.asarray(n, dtype=float)
    x = p.X - np.concatenate([[0.0], np.cumsum(n)])
    ec = expected_cost(p, n)
    vc = cost_variance(p, x)
    return {"expected_cost": ec, "variance": vc, "stdev": math.sqrt(max(vc, 0.0))}


def bps(cost_usd: float, X: float, price: float) -> float:
    """Dollars of shortfall -> basis points of the arrival notional."""
    denom = X * price
    return float("nan") if denom == 0 else 1e4 * cost_usd / denom


def implied_lambda(p: ACParams, target_kappa: float) -> float:
    """Invert ``kappa(lambda)``.

    Useful for stating urgency the way a trader does - "I want this half done
    in five minutes" - instead of in units of inverse dollars, which nobody has
    intuition for. This is the transform behind the urgency slider on the page.
    """
    p.validate()
    if target_kappa <= 0:
        return 0.0
    tau = p.tau
    return float(2.0 * p.eta_tilde * (math.cosh(target_kappa * tau) - 1.0)
                 / (p.sigma ** 2 * tau ** 2))


def temporary_impact_bps(p: ACParams, price: float,
                         rate_shares_per_sec: float) -> float:
    """Per-share temporary impact ``epsilon + eta v`` expressed in bp of price.

    Not used by the optimiser. It exists because ``eta``, in dollars per share
    per share-per-second, is impossible to eyeball, and a calibration error of
    three orders of magnitude is invisible until someone writes it in bp.
    """
    if price <= 0:
        return float("nan")
    return 1e4 * (p.epsilon + p.eta * rate_shares_per_sec) / price


__all__ = ["ACParams", "kappa", "trajectory", "expected_cost", "cost_variance",
           "solve", "frontier", "schedule_cost", "bps", "implied_lambda",
           "temporary_impact_bps"]
