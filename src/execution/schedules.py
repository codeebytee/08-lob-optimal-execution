"""The five execution algorithms, behind one interface.

Each algorithm answers a single question once per slice: *how many shares do I
send now?* Everything else - how those shares reach the book, what they fill at,
what the market does in response - belongs to the simulator, and keeping that
boundary clean is what lets the same five objects be scored against a
microstructure simulation, against the Almgren-Chriss cost model, and (in
JavaScript) against a reduced-form model on the web page.

The five:

``TWAP``       Equal shares per slice. The benchmark that everyone beats in
               backtests and nobody beats reliably in production.
``VWAP``       Shares proportional to the *expected* volume curve. It is not a
               volume-tracking algorithm - it cannot be, since the curve is a
               forecast - so it inherits tracking error whenever realised
               volume differs from the forecast, which is the honest way to
               show why VWAP algorithms miss their benchmark.
``POV``        Participate at a fixed fraction of *realised* volume. The only
               algorithm here whose completion is not guaranteed, and the code
               does not hide that: it reports shortfall in slices where it had
               to force out a remainder.
``AC``         The Almgren-Chriss trajectory for a given risk aversion, fixed
               at the start and followed regardless of what happens.
``Adaptive``   Almgren-Chriss re-solved every slice on the remaining shares and
               remaining time, plus an aggressiveness-in-the-money tilt in the
               spirit of Almgren-Lorenz (2007): buy faster when the price has
               moved in your favour.

Look-ahead is the thing to watch here, and there is exactly one rule: an
algorithm may use information dated strictly before the slice it is sizing.
``POV`` therefore sizes off the volume printed in the *previous* slice, not the
one it is about to trade into - using the latter would let it participate in
volume that its own trading helps create, and it is worth roughly a basis point
of free money. :class:`ExecState` carries only lagged fields for that reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils.config import FlowConfig, u_shape
from .almgren_chriss import ACParams, trajectory


@dataclass
class ExecState:
    """What an algorithm is allowed to know when sizing slice ``k``.

    Every field is as of the *end of slice k-1*. Nothing here can be
    contaminated by the slice about to be traded.
    """

    k: int                          # slice index about to be traded, 0-based
    n_slices: int
    remaining: float                # shares still to do
    elapsed: float                  # seconds since the parent started
    horizon: float                  # total seconds
    mid: float                      # last observed mid
    arrival: float                  # mid when the parent order arrived
    last_slice_volume: float        # market shares printed in slice k-1
    side: int = 1                   # +1 buy, -1 sell

    @property
    def remaining_time(self) -> float:
        return max(self.horizon - self.elapsed, 1e-9)


class Algo:
    """Base class. ``child_shares`` is the whole interface."""

    name = "base"

    def reset(self, X: float, n_slices: int, horizon: float) -> None:
        self.X = float(X)
        self.n_slices = int(n_slices)
        self.horizon = float(horizon)

    def child_shares(self, st: ExecState) -> float:
        raise NotImplementedError

    def plan(self, X: float, n_slices: int, horizon: float) -> Optional[np.ndarray]:
        """Static schedule if the algorithm has one, else ``None``.

        The web page draws these, and the Almgren-Chriss cost model scores
        them; adaptive algorithms return ``None`` because their schedule is not
        knowable in advance, which is the point of them.
        """
        return None


class TWAP(Algo):
    name = "TWAP"

    def child_shares(self, st: ExecState) -> float:
        left = st.n_slices - st.k
        return st.remaining / left if left > 0 else st.remaining

    def plan(self, X, n_slices, horizon):
        return np.full(n_slices, X / n_slices)


class VWAP(Algo):
    """Trade the forecast volume curve.

    The forecast is the same U-shape the simulator uses to modulate its
    intensities, which looks like cheating and is not: a real VWAP algorithm is
    fitted to the same historical curve the market keeps repeating, and the
    interesting error is not curve mis-estimation but the difference between
    the curve and the *realisation* on the day. That difference is present here
    in full - the simulator's realised volume is stochastic around the curve.
    """

    name = "VWAP"

    def __init__(self, flow: FlowConfig, start_fraction: float = 0.25,
                 seconds_per_day: float = 23400.0):
        self.flow = flow
        self.start_fraction = start_fraction
        self.seconds_per_day = seconds_per_day
        self._w: Optional[np.ndarray] = None

    def _weights(self, n_slices: int, horizon: float) -> np.ndarray:
        if self._w is not None and len(self._w) == n_slices:
            return self._w
        tau = horizon / n_slices
        mid_times = (np.arange(n_slices) + 0.5) * tau
        u = self.start_fraction + mid_times / self.seconds_per_day
        w = np.asarray(u_shape(np.clip(u, 0.0, 1.0), self.flow), dtype=float)
        self._w = w / w.sum()
        return self._w

    def reset(self, X, n_slices, horizon):
        super().reset(X, n_slices, horizon)
        self._w = None

    def child_shares(self, st: ExecState) -> float:
        w = self._weights(st.n_slices, st.horizon)
        left = w[st.k:].sum()
        if left <= 0:
            return st.remaining
        return st.remaining * w[st.k] / left

    def plan(self, X, n_slices, horizon):
        return X * self._weights(n_slices, horizon)


class POV(Algo):
    """Participate at ``rate`` of realised volume, with a completion backstop.

    Two production details that toy implementations skip:

    * **The lag.** Slice ``k`` is sized from slice ``k-1``'s printed volume.
      A POV that sizes off contemporaneous volume is trading on information it
      does not have, and it flatters itself by participating in the very volume
      its own child order creates.
    * **The backstop.** POV is a rate instruction, not a completion
      instruction. If volume never shows up, a pure POV finishes the day short.
      Real orders have a deadline, so the last ``catchup_slices`` slices size
      to finish, and :attr:`forced_shares` records how much had to be forced -
      a number the results table reports rather than hides.
    """

    name = "POV"

    def __init__(self, rate: float, catchup_slices: int = 3,
                 max_multiple: float = 4.0):
        self.rate = float(rate)
        self.catchup_slices = int(catchup_slices)
        self.max_multiple = float(max_multiple)
        self.forced_shares = 0.0

    def reset(self, X, n_slices, horizon):
        super().reset(X, n_slices, horizon)
        self.forced_shares = 0.0

    def child_shares(self, st: ExecState) -> float:
        left = st.n_slices - st.k
        even = st.remaining / max(left, 1)
        if left <= self.catchup_slices:
            forced = max(0.0, even - self.rate * st.last_slice_volume)
            self.forced_shares += min(forced, st.remaining)
            return st.remaining if left == 1 else min(st.remaining,
                                                      max(even, self.rate * st.last_slice_volume))
        want = self.rate * st.last_slice_volume
        # Cap the child at a multiple of the even rate. Without it, one
        # clustered burst of volume drags the whole order into a single slice,
        # which is a real POV failure mode but not one worth simulating as
        # unbounded.
        return float(min(st.remaining, want, self.max_multiple * even))


class AlmgrenChriss(Algo):
    """The static optimal trajectory, computed once and followed."""

    name = "AC"

    def __init__(self, params: ACParams, lam: float):
        self.params = params
        self.lam = float(lam)
        self._n: Optional[np.ndarray] = None

    def reset(self, X, n_slices, horizon):
        super().reset(X, n_slices, horizon)
        p = ACParams(X=X, T=horizon, N=n_slices, sigma=self.params.sigma,
                     eta=self.params.eta, gamma=self.params.gamma,
                     epsilon=self.params.epsilon)
        self.params = p
        self._n = trajectory(p, self.lam)["n"]

    def child_shares(self, st: ExecState) -> float:
        assert self._n is not None, "reset() must be called first"
        if st.k >= len(self._n):
            return st.remaining
        planned = float(self._n[st.k])
        # The plan is in absolute shares; if earlier slices under-filled, the
        # remainder still has to go. Scale the tail rather than dumping it at
        # the end, which is what a real implementation shortfall algo does.
        planned_left = float(self._n[st.k:].sum())
        if planned_left <= 0:
            return st.remaining
        return min(st.remaining, st.remaining * planned / planned_left)

    def plan(self, X, n_slices, horizon):
        p = ACParams(X=X, T=horizon, N=n_slices, sigma=self.params.sigma,
                     eta=self.params.eta, gamma=self.params.gamma,
                     epsilon=self.params.epsilon)
        return trajectory(p, self.lam)["n"]


class Adaptive(Algo):
    """Almgren-Chriss re-solved each slice, plus an in-the-money tilt.

    Almgren and Lorenz (2007) show that the static trajectory is *not* optimal
    once you are allowed to condition on the price path: a mean-variance trader
    should speed up after a favourable move, because the remaining order is now
    a smaller expected loss and the marginal value of getting it done has
    risen. The exact solution requires solving a stochastic control problem
    slice by slice; this is the practical version desks actually run, which is
    a one-parameter tilt on top of a re-solved static schedule:

        n_k <- n_k * (1 + tilt * z),   z = -side * (S_t - S_0) / (sigma sqrt t)

    ``z`` is the price move so far in standard deviations, signed so that
    positive means "in your favour". The multiplier is clipped to
    ``[1 - tilt_cap, 1 + tilt_cap]`` because the linear form is only a local
    approximation and an unclipped version will try to trade a negative number
    of shares after a three-sigma move against.

    Whether this is worth anything after costs is exactly the sort of claim
    this project is built to test rather than assert; see ``results/`` for the
    answer, which is not a flattering one at small parent sizes.
    """

    name = "Adaptive"

    def __init__(self, params: ACParams, lam: float, tilt: float = 1.0,
                 tilt_cap: float = 0.8):
        self.params = params
        self.lam = float(lam)
        self.tilt = float(tilt)
        self.tilt_cap = float(tilt_cap)

    def child_shares(self, st: ExecState) -> float:
        left_slices = st.n_slices - st.k
        if left_slices <= 1 or st.remaining <= 0:
            return st.remaining
        p = ACParams(X=st.remaining, T=st.remaining_time, N=left_slices,
                     sigma=self.params.sigma, eta=self.params.eta,
                     gamma=self.params.gamma, epsilon=self.params.epsilon)
        try:
            n = trajectory(p, self.lam)["n"]
        except ValueError:
            return st.remaining / left_slices
        base = float(n[0])
        if st.elapsed > 0 and self.params.sigma > 0:
            z = -st.side * (st.mid - st.arrival) / (self.params.sigma
                                                    * math.sqrt(st.elapsed))
        else:
            z = 0.0
        mult = 1.0 + self.tilt * z
        mult = min(max(mult, 1.0 - self.tilt_cap), 1.0 + self.tilt_cap)
        return float(min(st.remaining, base * mult))


def build_algos(names: Sequence[str], params: ACParams, lam: float,
                flow: FlowConfig, pov_rate: float, tilt: float,
                start_fraction: float = 0.25,
                seconds_per_day: float = 23400.0) -> List[Algo]:
    """Instantiate algorithms by name, in the order given."""
    out: List[Algo] = []
    for nm in names:
        key = nm.upper()
        if key == "TWAP":
            out.append(TWAP())
        elif key == "VWAP":
            out.append(VWAP(flow, start_fraction, seconds_per_day))
        elif key == "POV":
            out.append(POV(pov_rate))
        elif key == "AC":
            out.append(AlmgrenChriss(params, lam))
        elif key == "ADAPTIVE":
            out.append(Adaptive(params, lam, tilt))
        else:
            raise ValueError(f"unknown algorithm {nm!r}")
    return out


__all__ = ["ExecState", "Algo", "TWAP", "VWAP", "POV", "AlmgrenChriss",
           "Adaptive", "build_algos"]
