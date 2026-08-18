"""The five algorithms, tested on what they promise rather than on internals."""

from __future__ import annotations

import numpy as np
import pytest

from src.execution.almgren_chriss import ACParams
from src.execution.schedules import (POV, TWAP, VWAP, Adaptive,
                                     AlmgrenChriss, ExecState, build_algos)
from src.utils.config import FlowConfig


X = 100_000.0
N = 20
T = 1200.0


def state(k, remaining, mid=100.0, arrival=100.0, last_vol=0.0, side=1):
    return ExecState(k=k, n_slices=N, remaining=remaining, elapsed=k * T / N,
                     horizon=T, mid=mid, arrival=arrival,
                     last_slice_volume=last_vol, side=side)


def drive(algo, x=X, n=N, horizon=T, volumes=None, mids=None) -> np.ndarray:
    """Run an algorithm through a whole parent order and return the schedule."""
    algo.reset(x, n, horizon)
    out = []
    remaining = x
    for k in range(n):
        vol = 0.0 if (volumes is None or k == 0) else float(volumes[k - 1])
        mid = 100.0 if mids is None else float(mids[k])
        s = algo.child_shares(state(k, remaining, mid=mid, last_vol=vol))
        s = min(max(s, 0.0), remaining)
        out.append(s)
        remaining -= s
    return np.asarray(out)


@pytest.fixture
def ac_params() -> ACParams:
    return ACParams(X=X, T=T, N=N, sigma=0.01, eta=1e-5, gamma=1e-7,
                    epsilon=0.005)


def test_twap_is_flat_and_completes():
    n = drive(TWAP())
    assert np.allclose(n, X / N)
    assert n.sum() == pytest.approx(X)


def test_twap_catches_up_after_an_underfill():
    a = TWAP()
    a.reset(X, N, T)
    # Pretend slice 0 filled nothing: the remaining shares spread over what is
    # left, so the next child is larger than the original slice.
    assert a.child_shares(state(1, X)) > X / N


def test_vwap_follows_the_u_shape_and_completes():
    a = VWAP(FlowConfig(), start_fraction=0.0, seconds_per_day=T)
    n = drive(a)
    assert n.sum() == pytest.approx(X)
    # Full day in this test, so both ends are heavier than the middle.
    assert n[0] > n[N // 2] and n[-1] > n[N // 2]


def test_vwap_weights_are_normalised():
    a = VWAP(FlowConfig())
    a.reset(X, N, T)
    plan = a.plan(X, N, T)
    assert plan.sum() == pytest.approx(X)
    assert np.all(plan > 0)


def test_vwap_is_flat_when_the_curve_is_flat():
    flat = FlowConfig(u_a=1.0, u_b=0.0)
    n = drive(VWAP(flat))
    assert np.allclose(n, X / N)


def test_pov_tracks_volume_with_a_lag():
    """Slice k trades a fraction of slice k-1's volume, never its own."""
    a = POV(rate=0.2)
    volumes = np.full(N, 10_000.0)
    n = drive(a, volumes=volumes)
    assert n[0] == 0.0                     # no volume has printed yet
    assert n[1] == pytest.approx(0.2 * 10_000.0)


def test_pov_completes_and_records_what_it_forced():
    """With almost no volume, POV must still finish - and admit that it had
    to force the remainder out."""
    a = POV(rate=0.1)
    n = drive(a, volumes=np.full(N, 100.0))
    assert n.sum() == pytest.approx(X)
    assert a.forced_shares > 0.5 * X


def test_pov_caps_a_volume_burst():
    a = POV(rate=0.5, max_multiple=2.0)
    volumes = np.zeros(N)
    volumes[2] = 10_000_000.0
    n = drive(a, volumes=volumes)
    # The cap is a multiple of the *even remaining* rate, not of the original
    # slice size: by slice 3 nothing has traded, so the even rate is X/17.
    assert n[3] == pytest.approx(2.0 * X / (N - 3))


def test_pov_zero_rate_still_completes():
    a = POV(rate=0.0)
    n = drive(a, volumes=np.full(N, 50_000.0))
    assert n.sum() == pytest.approx(X)


def test_ac_plan_matches_the_closed_form(ac_params):
    a = AlmgrenChriss(ac_params, lam=2e-6)
    n = drive(a)
    assert n.sum() == pytest.approx(X)
    assert np.all(np.diff(n) < 0)          # front-loaded
    plan = a.plan(X, N, T)
    assert np.allclose(n, plan, rtol=1e-9)


def test_ac_at_zero_lambda_is_twap(ac_params):
    n = drive(AlmgrenChriss(ac_params, lam=0.0))
    assert np.allclose(n, X / N)


def test_ac_rescales_the_tail_after_an_underfill(ac_params):
    a = AlmgrenChriss(ac_params, lam=2e-6)
    a.reset(X, N, T)
    plan = a.plan(X, N, T)
    short = a.child_shares(state(5, X))    # nothing filled so far
    assert short > plan[5]                 # the tail is scaled up, not dumped


def test_adaptive_speeds_up_on_a_favourable_move(ac_params):
    a = Adaptive(ac_params, lam=2e-6, tilt=1.0)
    a.reset(X, N, T)
    neutral = a.child_shares(state(5, 0.6 * X, mid=100.0))
    cheaper = a.child_shares(state(5, 0.6 * X, mid=99.5))   # buying, price fell
    dearer = a.child_shares(state(5, 0.6 * X, mid=100.5))
    assert cheaper > neutral > dearer


def test_adaptive_sign_flips_for_a_sell(ac_params):
    a = Adaptive(ac_params, lam=2e-6, tilt=1.0)
    a.reset(X, N, T)
    up = a.child_shares(state(5, 0.6 * X, mid=100.5, side=-1))
    down = a.child_shares(state(5, 0.6 * X, mid=99.5, side=-1))
    assert up > down                        # selling into strength is the good case


def test_adaptive_tilt_is_clipped(ac_params):
    a = Adaptive(ac_params, lam=2e-6, tilt=5.0, tilt_cap=0.5)
    a.reset(X, N, T)
    base = a.child_shares(state(5, 0.6 * X, mid=100.0))
    extreme = a.child_shares(state(5, 0.6 * X, mid=90.0))
    assert extreme <= 1.5 * base + 1e-9
    assert a.child_shares(state(5, 0.6 * X, mid=110.0)) >= 0.0


def test_adaptive_completes(ac_params):
    mids = 100.0 + np.linspace(-0.5, 0.5, N)
    n = drive(Adaptive(ac_params, lam=2e-6), mids=mids)
    assert n.sum() == pytest.approx(X, rel=1e-9)


def test_every_algo_completes_under_a_random_path(ac_params):
    rng = np.random.default_rng(2)
    mids = 100.0 + np.cumsum(rng.normal(scale=0.05, size=N))
    volumes = rng.lognormal(mean=np.log(20_000.0), sigma=0.6, size=N)
    algos = build_algos(("TWAP", "VWAP", "POV", "AC", "Adaptive"), ac_params,
                        2e-6, FlowConfig(), pov_rate=0.15, tilt=1.0)
    for a in algos:
        n = drive(a, volumes=volumes, mids=mids)
        assert n.sum() == pytest.approx(X, rel=1e-9), a.name
        assert np.all(n >= -1e-9), a.name


def test_build_algos_rejects_unknown_names(ac_params):
    with pytest.raises(ValueError, match="unknown algorithm"):
        build_algos(("MAGIC",), ac_params, 1e-6, FlowConfig(), 0.1, 1.0)


def test_only_lagged_volume_reaches_the_algorithm():
    """The state object is the look-ahead guard: there is no field on it that
    could carry the current slice's volume."""
    fields = set(ExecState.__dataclass_fields__)
    assert "last_slice_volume" in fields
    assert not {"current_volume", "future_volume", "next_mid"} & fields
