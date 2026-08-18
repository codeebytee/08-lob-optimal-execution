"""Statistics for cost distributions, and for comparing two of them.

Execution quality is a distribution, not a number. A schedule that averages
6 bp with a 40 bp tail is worse for most desks than one averaging 7 bp with a
12 bp tail, and reporting only the mean hides exactly the thing the risk
aversion parameter exists to price. Everything here is built around that.

The comparison functions deserve a note. Algorithms are scored on *paired*
paths - the same seed drives the same anonymous order flow for every
algorithm - so the right test is a paired one on the differences. The paired
standard error is typically several times smaller than the unpaired one,
because the common price path cancels; using the unpaired version would make
every real difference look insignificant and every conclusion "needs more
paths".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CostStats:
    """Summary of one cost distribution, in basis points."""

    n: int
    mean: float
    stdev: float
    stderr: float
    median: float
    p05: float
    p25: float
    p75: float
    p95: float
    cvar95: float
    worst: float
    best: float

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "mean": self.mean, "stdev": self.stdev,
                "stderr": self.stderr, "median": self.median, "p05": self.p05,
                "p25": self.p25, "p75": self.p75, "p95": self.p95,
                "cvar95": self.cvar95, "worst": self.worst, "best": self.best}


def cost_stats(x: Sequence[float]) -> CostStats:
    """Summarise a shortfall sample. ``cvar95`` is the mean of the worst 5%.

    Conditional value at risk rather than the 95th percentile alone, because
    the interesting question about an execution algorithm is not "how bad is a
    bad day" but "how bad are the bad days on average" - a tail that is thin
    beyond the 95th percentile and one that is not look identical in a
    percentile and very different in a CVaR.
    """
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        nan = float("nan")
        return CostStats(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    tail = a[a >= np.percentile(a, 95)]
    return CostStats(
        n=int(n), mean=float(a.mean()),
        stdev=float(a.std(ddof=1)) if n > 1 else 0.0,
        stderr=float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        median=float(np.median(a)), p05=float(np.percentile(a, 5)),
        p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
        p95=float(np.percentile(a, 95)),
        cvar95=float(tail.mean()) if tail.size else float(a.max()),
        worst=float(a.max()), best=float(a.min()))


@dataclass(frozen=True)
class PairedTest:
    """Paired difference between two algorithms on the same paths."""

    n: int
    mean_diff: float
    stderr: float
    t_stat: float
    win_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "mean_diff": self.mean_diff,
                "stderr": self.stderr, "t_stat": self.t_stat,
                "win_rate": self.win_rate}


def paired_test(a: Sequence[float], b: Sequence[float]) -> PairedTest:
    """``a - b`` on matched paths. Negative mean means ``a`` was cheaper.

    ``win_rate`` is the fraction of paths on which ``a`` was cheaper, which is
    reported alongside the t-statistic because the two disagree more often than
    people expect: an algorithm can win 70% of paths and still lose on average
    if the 30% it loses are the paths where it loses badly.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    n = min(x.size, y.size)
    d = x[:n] - y[:n]
    d = d[np.isfinite(d)]
    if d.size < 2:
        return PairedTest(int(d.size), float("nan"), float("nan"),
                          float("nan"), float("nan"))
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    return PairedTest(n=int(d.size), mean_diff=float(d.mean()), stderr=se,
                      t_stat=float(d.mean() / se) if se > 0 else float("nan"),
                      win_rate=float(np.mean(d < 0)))


def histogram(x: Sequence[float], bins: int = 41,
              lo: Optional[float] = None,
              hi: Optional[float] = None) -> Dict[str, List[float]]:
    """Histogram on a fixed range, for shipping to the page.

    The range is an argument so that several algorithms can share identical
    bins - overlaid histograms on differently binned axes are a way to make any
    two distributions look however you want.
    """
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"edges": [], "counts": []}
    lo = float(np.min(a)) if lo is None else float(lo)
    hi = float(np.max(a)) if hi is None else float(hi)
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(a, bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts]}


def mean_variance_objective(cost_bps: Sequence[float], lam_bps: float) -> float:
    """``mean + lambda * variance`` of a cost sample, in bp units.

    ``lam_bps`` is the risk aversion expressed per basis point rather than per
    dollar, which is the only form in which the number is comparable across
    parent sizes and names.
    """
    a = np.asarray(list(cost_bps), dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    return float(a.mean() + lam_bps * a.var(ddof=1))


def bootstrap_ci(x: Sequence[float], stat=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap interval.

    Used for the tail statistics, where the normal approximation behind a
    standard error is not credible: a CVaR estimated from 5% of 400 paths is an
    average of twenty numbers from the fattest part of the distribution.
    """
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return {"lo": float("nan"), "hi": float("nan"), "point": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    draws = np.array([stat(a[i]) for i in idx])
    return {"point": float(stat(a)),
            "lo": float(np.percentile(draws, 100 * alpha / 2)),
            "hi": float(np.percentile(draws, 100 * (1 - alpha / 2)))}


__all__ = ["CostStats", "cost_stats", "PairedTest", "paired_test", "histogram",
           "mean_variance_objective", "bootstrap_ci"]
