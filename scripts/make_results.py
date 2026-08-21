"""The whole research run, in one command.

    python scripts/make_results.py            # full run, uses every core
    python scripts/make_results.py --quick    # ~2 minutes, for a smoke test
    python scripts/make_results.py --workers 8

Order of operations, because each stage depends on the one before it:

1. **Calibrate the venue** to every name in the snapshot - flow rates to ADV,
   Kyle lambda to volatility over volume, exogenous vol to the residual.
2. **Measure the impact parameters** of the calibrated venue by executing TWAP
   parents at a range of participation rates against paired counterfactual
   sessions. This is what produces gamma, eta and the impact exponent.
3. **Run the algorithm tournament** over a grid of parent size, volatility
   regime and risk aversion, on common random numbers.
4. **Compare the model to the venue**: the Almgren-Chriss efficient frontier
   computed from the fitted parameters, against the cost and cost dispersion
   the same schedules actually incurred in the simulator.
5. **Record one execution in full** for the animated book on the web page.
6. Write figures, tables and ``results/summary.json``.

Everything is seeded. Two runs of this script produce identical numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.market import NameStats, load_snapshot  # noqa: E402
from src.execution.almgren_chriss import (ACParams, bps, frontier,  # noqa: E402
                                          schedule_cost, solve)
from src.execution.runner import (apply_control_variate, baseline_path,  # noqa: E402
                                  run_parent)
from src.execution.schedules import (POV, TWAP, VWAP, Adaptive,  # noqa: E402
                                     AlmgrenChriss)
from src.flow.calibrate import (calibrate_venue, fit_impact,  # noqa: E402
                                impact_sample, measure_impact,
                                summarise_impact_samples)
from src.utils.config import AppConfig  # noqa: E402
from src.utils.stats import cost_stats, histogram, paired_test  # noqa: E402

RESULTS = ROOT / "results"

# The worker processes need these; they are set once per process by _init.
_CTX: Dict[str, object] = {}


# --------------------------------------------------------------------------
# the tournament worker
# --------------------------------------------------------------------------

_POOL_PROBE_SECONDS = 90.0


def _probe() -> bool:
    """A task that does nothing, used to prove a worker process is alive."""
    return True


def _init(payload: Dict[str, object]) -> None:
    """Runs once per worker process. Everything here is read-only afterwards."""
    _CTX.update(payload)


def _parallel_map(fn, tasks, ctx: Dict[str, object], workers: int,
                  chunksize: int = 1, log=None):
    """Map ``fn`` over ``tasks``, yielding results as they complete.

    Process pools are a convenience here, not a requirement: every task is
    independent and the serial path produces identical numbers, only slower.
    Windows occasionally refuses to start a pool at all - the child fails in
    ``spawn_main`` with ``PermissionError: [WinError 5]`` while duplicating the
    parent's pipe handle, which is an environment condition (handle pressure,
    an interfering security product) rather than anything wrong with the work.
    Losing an hour of completed stages to that is not acceptable, so pool
    creation is retried with fewer workers and then abandoned for the serial
    path, with a line in the log saying which one ran.
    """
    if workers > 1:
        attempts = [w for w in (workers, max(2, workers // 2), 2) if w > 1]
        seen = []
        for w in attempts:
            if w in seen:
                continue
            seen.append(w)
            pool = None
            try:
                pool = mp.Pool(w, initializer=_init, initargs=(ctx,))
                # The constructor returns before the children have finished
                # starting, and a child that dies during spawn shows up only
                # as work that never comes back. Round-trip one trivial task
                # so a broken pool fails here, in a few seconds, rather than
                # hanging the stage.
                pool.apply_async(_probe).get(timeout=_POOL_PROBE_SECONDS)
            except Exception as exc:                # noqa: BLE001 - see above
                if log is not None:
                    log(f"  process pool with {w} workers failed to start "
                        f"({type(exc).__name__}: {exc}); retrying smaller")
                if pool is not None:
                    pool.terminate()
                continue
            # Past this point results have been handed to the caller, so a
            # failure is a real one and must not be retried behind its back.
            with pool:
                yield from pool.imap_unordered(fn, tasks, chunksize=chunksize)
            return

        if log is not None:
            log("  no process pool available; running this stage serially")

    _init(ctx)
    for task in tasks:
        yield fn(task)


def _make_algo(name: str, lam: float, params: ACParams, cfg: AppConfig, flow):
    if name == "TWAP":
        return TWAP()
    if name == "VWAP":
        return VWAP(flow, _CTX["start_fraction"], cfg.market.seconds_per_day)
    if name == "POV":
        return POV(cfg.execution.pov_rate)
    if name == "AC":
        return AlmgrenChriss(params, lam)
    if name == "Adaptive":
        return Adaptive(params, lam, cfg.execution.adaptive_tilt)
    raise ValueError(name)


def _run_cell(task: Tuple[float, float, int]) -> List[Dict[str, float]]:
    """One (parent size, volatility regime, seed): every algorithm, one path.

    All algorithms in a cell share the seed, so they see the same anonymous
    order flow and the same exogenous price path. The baseline - the session
    with no parent order at all - is run once for the cell and reused as the
    control variate for each algorithm.
    """
    size_pct, vol_mult, seed = task
    cfg: AppConfig = _CTX["cfg"]           # type: ignore[assignment]
    stats: NameStats = _CTX["stats"]       # type: ignore[assignment]
    flow = _CTX["flow"]
    kyle = float(_CTX["kyle"])             # type: ignore[arg-type]
    sigma_exo = float(_CTX["sigma_exo"])   # type: ignore[arg-type]
    params: ACParams = _CTX["ac_params"]   # type: ignore[assignment]
    horizon = float(_CTX["horizon"])       # type: ignore[arg-type]
    n_slices = int(_CTX["n_slices"])       # type: ignore[arg-type]
    lambdas = list(_CTX["lambdas"])        # type: ignore[arg-type]
    lot = cfg.book.lot_size

    X = int(round(size_pct * stats.adv_shares / lot) * lot)
    base = baseline_path(stats, cfg.book, flow, seed, horizon, n_slices,
                         kyle_lambda=kyle, sigma_exo=sigma_exo,
                         vol_multiplier=vol_mult,
                         seconds_per_day=cfg.market.seconds_per_day)

    out: List[Dict[str, float]] = []
    plans = [("TWAP", None), ("VWAP", None), ("POV", None)]
    plans += [("AC", lam) for lam in lambdas]
    plans += [("Adaptive", lam) for lam in lambdas]

    p_cell = ACParams(X=float(X), T=horizon, N=n_slices, sigma=params.sigma * vol_mult,
                      eta=params.eta, gamma=params.gamma, epsilon=params.epsilon)

    for name, lam in plans:
        algo = _make_algo(name, lam or 0.0, p_cell, cfg, flow)
        res = run_parent(stats, cfg.book, flow, algo, X, seed, horizon,
                         n_slices, kyle_lambda=kyle, sigma_exo=sigma_exo,
                         vol_multiplier=vol_mult,
                         seconds_per_day=cfg.market.seconds_per_day)
        apply_control_variate(res, base)
        out.append({"size_pct": size_pct, "vol_mult": vol_mult, "seed": seed,
                    "algo": name, "lambda": (lam if lam is not None else float("nan")),
                    "X": X, "shortfall_bps": res.shortfall_bps,
                    "shortfall_adj_bps": res.shortfall_adj_bps,
                    "cv_bps": res.cv_bps, "vs_vwap_bps": res.vs_vwap_bps,
                    "fill_rate": res.filled_shares / X,
                    "participation": res.participation,
                    "forced_shares": res.forced_shares,
                    "final_move_bps": 1e4 * (res.final_mid - res.arrival_mid)
                    / res.arrival_mid})
    return out


def impact_scale(name, base) -> float:
    """Carry impact parameters measured on one name to another.

    Both coefficients are scaled by the ratio of *dollar volatility per unit of
    volume*, ``P sigma / ADV``. That is the same dimensional argument the Kyle
    coefficient rests on - a day's volume moves the price by about a day's
    volatility - and it is the only scaling in this project that is applied
    across names, so it is worth being explicit that it is an assumption rather
    than a measurement. The alternative would be to repeat the whole impact
    calibration for all eight names, which costs eight times the compute for a
    cross-sectional chart that is illustrative either way.
    """
    def unit(s):
        return s.price * s.sigma_annual / s.adv_shares
    return float(unit(name) / unit(base))


def _impact_task(task):
    """One paired impact sample. Top-level so the process pool can pickle it."""
    rho, X, seed = task
    cfg: AppConfig = _CTX["cfg"]           # type: ignore[assignment]
    return (rho, X, impact_sample(_CTX["stats"], cfg.book, _CTX["cal"], X, seed,
                                  float(_CTX["horizon"]), int(_CTX["n_slices"]),
                                  cfg.market.seconds_per_day))


def _run_cross_section(task: Tuple[str, int]) -> List[Dict[str, float]]:
    """One name, one seed, every algorithm - for the cross-sectional chart."""
    ticker, seed = task
    cfg: AppConfig = _CTX["cfg"]           # type: ignore[assignment]
    cross = _CTX["cross"]                  # type: ignore[assignment]
    entry = cross[ticker]
    stats: NameStats = entry["stats"]
    flow = entry["flow"]
    params: ACParams = entry["ac_params"]
    horizon = float(_CTX["horizon"])       # type: ignore[arg-type]
    n_slices = int(_CTX["n_slices"])       # type: ignore[arg-type]
    lot = cfg.book.lot_size
    size_pct = float(_CTX["cross_size"])   # type: ignore[arg-type]
    X = int(round(size_pct * stats.adv_shares / lot) * lot)

    base = baseline_path(stats, cfg.book, flow, seed, horizon, n_slices,
                         kyle_lambda=entry["kyle"], sigma_exo=entry["sigma_exo"],
                         seconds_per_day=cfg.market.seconds_per_day)
    p_cell = ACParams(X=float(X), T=horizon, N=n_slices, sigma=params.sigma,
                      eta=params.eta, gamma=params.gamma, epsilon=params.epsilon)
    out = []
    for name, lam in (("TWAP", None), ("VWAP", None), ("POV", None),
                      ("AC", cfg.execution.risk_aversion),
                      ("Adaptive", cfg.execution.risk_aversion)):
        algo = _make_algo(name, lam or 0.0, p_cell, cfg, flow)
        res = run_parent(stats, cfg.book, flow, algo, X, seed, horizon,
                         n_slices, kyle_lambda=entry["kyle"],
                         sigma_exo=entry["sigma_exo"],
                         seconds_per_day=cfg.market.seconds_per_day)
        apply_control_variate(res, base)
        out.append({"ticker": ticker, "seed": seed, "algo": name,
                    "shortfall_bps": res.shortfall_bps,
                    "shortfall_adj_bps": res.shortfall_adj_bps,
                    "fill_rate": res.filled_shares / X, "X": X})
    return out


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_calibrate(cfg: AppConfig, snap, quick: bool, log) -> Dict[str, object]:
    log("calibrating the venue to each name")
    T = 300.0 if quick else 900.0
    cals = {}
    for t in snap.tickers:
        t0 = time.time()
        cal = calibrate_venue(snap[t], cfg.book, cfg.flow, T=T,
                              seconds_per_day=cfg.market.seconds_per_day)
        cals[t] = cal
        log(f"  {t:<5} scale={cal.rate_scale:6.3f}  ADV sim/real="
            f"{cal.sim_adv_shares/1e6:6.2f}/{snap[t].adv_shares/1e6:6.2f}M  "
            f"sigma sim/target={cal.sim_sigma_per_sec:.5f}/"
            f"{cal.target_sigma_per_sec:.5f}  spread={cal.median_spread_ticks:.1f}t"
            f"  touch={cal.mean_touch_lots:.1f} lots  [{time.time()-t0:.0f}s]")
    df = pd.DataFrame([{**c.to_dict(),
                        "real_adv_shares": snap[t].adv_shares,
                        "price": snap[t].price,
                        "sigma_annual": snap[t].sigma_annual}
                       for t, c in cals.items()]).set_index("ticker")
    df.to_csv(RESULTS / "venue_calibration.csv", float_format="%.8g")
    return cals


def stage_impact(cfg: AppConfig, snap, cals, ticker: str, quick: bool,
                 workers: int, log):
    """Measure gamma, eta and the impact exponent, across every core.

    The samples are independent, so this parallelises exactly; it is the
    single most expensive stage in the pipeline because each sample is two full
    simulated sessions.
    """
    log(f"measuring impact on {ticker} against paired counterfactuals")
    t0 = time.time()
    stats = snap[ticker]
    cal = cals[ticker]
    parts = (0.01, 0.05, 0.15) if quick else cfg.impact.calib_participations
    paths = 8 if quick else cfg.impact.calib_paths
    horizon, n_slices = cfg.execution.horizon_seconds, cfg.execution.n_slices
    lot = cfg.book.lot_size
    expected_volume = cal.sim_adv_shares * horizon / cfg.market.seconds_per_day

    tasks = []
    for rho in parts:
        X = int(round(max(rho * expected_volume, lot) / lot) * lot)
        tasks += [(rho, X, 5000 + i) for i in range(paths)]

    ctx = {"cfg": cfg, "stats": stats, "cal": cal, "horizon": horizon,
           "n_slices": n_slices}
    bucket: Dict[float, List[Dict[str, float]]] = {rho: [] for rho in parts}
    sizes: Dict[float, int] = {}
    for rho, X, sample in _parallel_map(_impact_task, tasks, ctx, workers,
                                       chunksize=2, log=log):
        bucket[rho].append(sample)
        sizes[rho] = X

    rows = [summarise_impact_samples(rho, sizes[rho], horizon, bucket[rho])
            for rho in parts]
    ic = fit_impact(stats, cal, rows)
    pd.DataFrame(ic.rows).to_csv(RESULTS / "impact_calibration.csv",
                                 index=False, float_format="%.8g")
    for r in ic.rows:
        log(f"  participation {r['participation']:6.1%}  X={r['X']:>9,.0f} sh  "
            f"permanent ${r['perm_usd']:6.3f} +/- {r['perm_se']:.3f}  "
            f"cost/share ${r['cost_per_share']:7.4f} +/- {r['cost_se']:.4f}  "
            f"fill {r['fill_rate']:6.1%}")
    log(f"  gamma={ic.gamma:.4g} $/share^2   eta={ic.eta:.4g} $/share/(share/s)"
        f"   epsilon={ic.epsilon:.4g} $/share")
    log(f"  injected Kyle lambda = {cal.kyle_lambda / lot:.4g} $/share; "
        f"recovered gamma is {ic.gamma / (cal.kyle_lambda / lot):.0%} of it")
    log(f"  temporary impact exponent = {ic.exponent:.3f} (R2 {ic.exponent_r2:.3f});"
        f"  linear model R2 = {ic.linear_r2:.3f}   [{time.time()-t0:.0f}s]")
    return ic


def stage_tournament(cfg: AppConfig, ctx: Dict[str, object], sizes, vols,
                     seeds, workers: int, log) -> pd.DataFrame:
    tasks = [(s, v, seed) for s in sizes for v in vols for seed in seeds]
    log(f"algorithm tournament: {len(tasks)} cells x "
        f"{3 + 2 * len(ctx['lambdas'])} runs")
    t0 = time.time()
    rows: List[Dict[str, float]] = []
    for i, part in enumerate(_parallel_map(_run_cell, tasks, ctx, workers,
                                          chunksize=1, log=log)):
        rows.extend(part)
        if (i + 1) % max(1, len(tasks) // 20) == 0:
            done = (i + 1) / len(tasks)
            el = time.time() - t0
            log(f"  {done:5.0%}  [{el:5.0f}s elapsed, "
                f"{el / done - el:5.0f}s left]")
    log(f"  done in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def summarise_tournament(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    out = []
    for (size, vol, algo, lam), g in df.groupby(
            ["size_pct", "vol_mult", "algo", "lambda"], dropna=False):
        raw = cost_stats(g["shortfall_bps"])
        adj = cost_stats(g["shortfall_adj_bps"])
        out.append({"size_pct": size, "vol_mult": vol, "algo": algo,
                    "lambda": lam, "n": raw.n,
                    "mean_bps": adj.mean, "stderr_bps": adj.stderr,
                    "raw_mean_bps": raw.mean, "raw_stderr_bps": raw.stderr,
                    "stdev_bps": raw.stdev, "median_bps": raw.median,
                    "p05_bps": raw.p05, "p95_bps": raw.p95,
                    "cvar95_bps": raw.cvar95, "worst_bps": raw.worst,
                    "fill_rate": g["fill_rate"].mean(),
                    "participation": g["participation"].mean(),
                    "vs_vwap_bps": g["vs_vwap_bps"].mean(),
                    "forced_frac": (g["forced_shares"] / g["X"]).mean()})
    return pd.DataFrame(out).sort_values(["size_pct", "vol_mult", "algo", "lambda"])


def stage_frontier(cfg: AppConfig, ctx, summary: pd.DataFrame, ic, stats,
                   size_pct: float, log) -> pd.DataFrame:
    """Model frontier vs what the venue actually charged for the same schedules."""
    lot = cfg.book.lot_size
    X = int(round(size_pct * stats.adv_shares / lot) * lot)
    p = ACParams(X=float(X), T=cfg.execution.horizon_seconds,
                 N=cfg.execution.n_slices, sigma=ctx["ac_params"].sigma,
                 eta=ic.eta, gamma=ic.gamma, epsilon=ic.epsilon)
    lams = np.logspace(math.log10(cfg.execution.lambda_lo),
                       math.log10(cfg.execution.lambda_hi),
                       cfg.execution.lambda_n)
    f = frontier(p, lams)
    rows = [{"lambda": float(l), "model_cost_bps": bps(c, p.X, stats.price),
             "model_stdev_bps": bps(s, p.X, stats.price),
             "kappa": float(k), "half_life_s": (math.log(2) / k if k > 0 else np.inf)}
            for l, c, s, k in zip(f["lambda"], f["expected_cost"], f["stdev"],
                                  f["kappa"])]
    df = pd.DataFrame(rows)

    sim = summary[(summary["size_pct"] == size_pct) & (summary["vol_mult"] == 1.0)]
    sim_rows = []
    for _, r in sim.iterrows():
        if r["algo"] in ("AC", "Adaptive") and np.isfinite(r["lambda"]):
            lam = float(r["lambda"])
        elif r["algo"] == "TWAP":
            lam = 0.0
        else:
            lam = float("nan")
        sim_rows.append({"algo": r["algo"], "lambda": lam,
                         "sim_cost_bps": r["mean_bps"],
                         "sim_stderr_bps": r["stderr_bps"],
                         "sim_stdev_bps": r["stdev_bps"],
                         "model_cost_bps": (bps(solve(p, lam)["expected_cost"],
                                                p.X, stats.price)
                                            if np.isfinite(lam) else float("nan")),
                         "model_stdev_bps": (bps(solve(p, lam)["stdev"], p.X,
                                                 stats.price)
                                             if np.isfinite(lam) else float("nan"))})
    sim_df = pd.DataFrame(sim_rows)
    df.to_csv(RESULTS / "frontier_model.csv", index=False, float_format="%.8g")
    sim_df.to_csv(RESULTS / "frontier_sim.csv", index=False, float_format="%.8g")
    log("  model vs simulated cost for the same schedules:")
    for _, r in sim_df.iterrows():
        log(f"    {r['algo']:<9} lam={r['lambda']:.1e}  model="
            f"{r['model_cost_bps']:7.2f} bp   sim={r['sim_cost_bps']:7.2f} "
            f"+/- {r['sim_stderr_bps']:.2f} bp")
    return df, sim_df


def stage_cross_section(cfg, snap, cals, cal, ic, stats, workers: int,
                        quick: bool, log) -> pd.DataFrame:
    """The same order, sized to each name's own ADV, in each name's venue."""
    cross_ctx = {"cfg": cfg, "horizon": cfg.execution.horizon_seconds,
                 "n_slices": cfg.execution.n_slices,
                 "cross_size": cfg.sweep.default_size_pct_adv,
                 "start_fraction": 0.25, "flow": cal.flow,
                 "cross": {t: {"stats": snap[t], "flow": cals[t].flow,
                               "kyle": cals[t].kyle_lambda,
                               "sigma_exo": cals[t].sigma_exo,
                               "ac_params": ACParams(
                                   X=1.0, T=cfg.execution.horizon_seconds,
                                   N=cfg.execution.n_slices,
                                   sigma=snap[t].sigma_per_second(
                                       cfg.market.seconds_per_day),
                                   eta=ic.eta * impact_scale(snap[t], stats),
                                   gamma=ic.gamma * impact_scale(snap[t], stats),
                                   epsilon=ic.epsilon)}
                           for t in snap.tickers}}
    seeds = list(range(30_000, 30_000 + (4 if quick else 60)))
    tasks = [(t, sd) for t in snap.tickers for sd in seeds]
    log(f"cross-section: {len(tasks)} runs across {len(snap.tickers)} names")
    rows: List[Dict[str, float]] = []
    t0 = time.time()
    for part in _parallel_map(_run_cross_section, tasks, cross_ctx, workers,
                             chunksize=2, log=log):
        rows.extend(part)
    cross = pd.DataFrame(rows)
    cross.to_csv(RESULTS / "cross_section.csv", index=False, float_format="%.6g")
    for t in snap.tickers:
        g = cross[(cross["ticker"] == t) & (cross["algo"] == "TWAP")]
        log(f"  {t:<5} TWAP {g['shortfall_adj_bps'].mean():6.2f} bp   "
            f"fill {g['fill_rate'].mean():5.1%}   "
            f"(vol {snap[t].sigma_annual:5.1%}, ADV {snap[t].adv_shares/1e6:5.1f}M)")
    log(f"  done in {time.time() - t0:.0f}s")
    return cross


def stage_tape(cfg: AppConfig, ctx, stats, log) -> Dict[str, object]:
    """One execution, recorded frame by frame, for the animation."""
    log("recording one execution for the book animation")
    lot = cfg.book.lot_size
    X = int(round(0.10 * stats.adv_shares / lot) * lot)
    p = ctx["ac_params"]
    p_cell = ACParams(X=float(X), T=cfg.execution.horizon_seconds,
                      N=cfg.execution.n_slices, sigma=p.sigma, eta=p.eta,
                      gamma=p.gamma, epsilon=p.epsilon)
    # Five candidate paths, and the one with the median shortfall is the one
    # that gets animated. Picking a single arbitrary seed would as likely as
    # not show a path where the price ran away and the algorithm looked either
    # heroic or terrible - neither of which is what the animation is for.
    cands = []
    for seed in (424242, 424243, 424244, 424245, 424246):
        algo = AlmgrenChriss(p_cell, cfg.execution.risk_aversion)
        r = run_parent(stats, cfg.book, ctx["flow"], algo, X, seed,
                       cfg.execution.horizon_seconds, cfg.execution.n_slices,
                       kyle_lambda=float(ctx["kyle"]),
                       sigma_exo=float(ctx["sigma_exo"]),
                       seconds_per_day=cfg.market.seconds_per_day,
                       record_every=cfg.sweep.tape_snapshot_every)
        cands.append(r)
    res = sorted(cands, key=lambda r: r.shortfall_bps)[len(cands) // 2]

    frames = [{"t": round(s.t, 1), "bid": round(s.best_bid, 2),
               "ask": round(s.best_ask, 2), "latent": round(s.latent, 3),
               "bq": s.bid_lots, "aq": s.ask_lots,
               "bpx": [round(x, 2) for x in s.bid_px],
               "apx": [round(x, 2) for x in s.ask_px],
               "done": s.agent_done}
              for s in res.snapshots]
    tape = {"ticker": stats.ticker, "X": X, "arrival": round(res.arrival_mid, 4),
            "horizon": cfg.execution.horizon_seconds,
            "shortfall_bps": round(res.shortfall_bps, 3),
            "frames": frames,
            "slices": [{"k": s.k, "t": s.t_start, "target": round(s.target_shares),
                        "filled": s.filled_shares,
                        "px": (round(s.avg_price, 4) if np.isfinite(s.avg_price) else None),
                        "mid": round(s.mid_end, 4),
                        "mkt": s.market_shares} for s in res.slices]}
    (RESULTS / "tape.json").write_text(json.dumps(tape), encoding="utf-8")
    log(f"  {len(frames)} frames, shortfall {res.shortfall_bps:.2f} bp")
    return tape


def stage_figures(summary: pd.DataFrame, ic, model_frontier, sim_frontier,
                  cross: pd.DataFrame, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log("writing figures")
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

    # 1. impact calibration
    rows = pd.DataFrame(ic.rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].errorbar(rows["X"], rows["perm_usd"], yerr=rows["perm_se"], fmt="o")
    xs = np.linspace(0, rows["X"].max(), 50)
    ax[0].plot(xs, ic.gamma * xs, "-", label=f"gamma = {ic.gamma:.3g}")
    ax[0].set_xlabel("parent size (shares)")
    ax[0].set_ylabel("permanent price move ($)")
    ax[0].set_title("permanent impact vs size")
    ax[0].legend()
    resid = rows["cost_per_share"] - ic.epsilon - 0.5 * ic.gamma * rows["X"]
    ax[1].loglog(rows["rate"], resid.clip(lower=1e-6), "o")
    rr = np.linspace(rows["rate"].min(), rows["rate"].max(), 50)
    ax[1].loglog(rr, ic.eta * rr, "-", label=f"linear, eta = {ic.eta:.3g}")
    ax[1].set_xlabel("trading rate (shares/s)")
    ax[1].set_ylabel("temporary cost ($/share)")
    ax[1].set_title(f"temporary impact, fitted exponent {ic.exponent:.2f}")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_impact_calibration.png")
    plt.close(fig)

    # 2. frontier: model vs simulated
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.plot(model_frontier["model_stdev_bps"], model_frontier["model_cost_bps"],
            "-", label="Almgren-Chriss frontier (model)")
    for _, r in sim_frontier.iterrows():
        ax.errorbar(r["sim_stdev_bps"], r["sim_cost_bps"],
                    yerr=r["sim_stderr_bps"], fmt="o")
        ax.annotate(f"{r['algo']}", (r["sim_stdev_bps"], r["sim_cost_bps"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("cost standard deviation (bp)")
    ax.set_ylabel("expected cost (bp)")
    ax.set_title("efficient frontier: model line, simulated points")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_frontier.png")
    plt.close(fig)

    # 3. cost vs size by algorithm
    base = summary[(summary["vol_mult"] == 1.0)]
    fig, ax = plt.subplots(figsize=(5.4, 4))
    for algo, g in base.groupby("algo"):
        gg = g.groupby("size_pct")["mean_bps"].mean()
        ax.plot(gg.index * 100, gg.values, "o-", label=algo)
    ax.set_xlabel("parent size (% of ADV)")
    ax.set_ylabel("mean implementation shortfall (bp)")
    ax.set_title("cost against size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_cost_vs_size.png")
    plt.close(fig)

    # 4. tail risk
    fig, ax = plt.subplots(figsize=(5.4, 4))
    mid = base[base["size_pct"] == base["size_pct"].median()]
    lbl = [f"{r['algo']}" + (f" {r['lambda']:.0e}" if np.isfinite(r["lambda"]) else "")
           for _, r in mid.iterrows()]
    ax.bar(range(len(mid)), mid["cvar95_bps"], label="CVaR 95%")
    ax.bar(range(len(mid)), mid["mean_bps"], label="mean")
    ax.set_xticks(range(len(mid)))
    ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("bp")
    ax.set_title("mean and tail cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_tail.png")
    plt.close(fig)

    # 5. cross-section
    if len(cross):
        piv = cross.pivot_table(index="ticker", columns="algo",
                                values="shortfall_adj_bps", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        piv.plot(kind="bar", ax=ax)
        ax.set_ylabel("mean shortfall (bp)")
        ax.set_title("the same order, eight names")
        fig.tight_layout()
        fig.savefig(RESULTS / "fig_cross_section.png")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny grid, for checking the pipeline runs")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = all cores minus two")
    ap.add_argument("--only", choices=["cross"], default=None,
                    help="re-run one stage on top of an existing results/ "
                         "directory, instead of the whole pipeline")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    log_path = RESULTS / "make_results.log"
    log_file = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    t_start = time.time()
    cfg = AppConfig.load()
    snap = load_snapshot(cfg.market)
    log(f"snapshot: {snap.source}, as of {snap.as_of}, "
        f"{len(snap.tickers)} names ({', '.join(snap.tickers)})")
    if not snap.is_real:
        log("  WARNING: synthetic snapshot - no market data was available")

    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)
    cals = stage_calibrate(cfg, snap, args.quick, log)
    ticker = cfg.market.default_ticker
    stats = snap[ticker]
    if args.only == "cross":
        # Re-run one stage on top of an existing results/ directory. The impact
        # parameters are read back rather than remeasured, so the numbers stay
        # exactly the ones the rest of results/ was built from.
        import dataclasses as _dc
        from src.flow.calibrate import ImpactCalibration
        prev = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        ic = ImpactCalibration(ticker=ticker, gamma=prev["impact"]["gamma"],
                               eta=prev["impact"]["eta"],
                               epsilon=prev["impact"]["epsilon"],
                               exponent=prev["impact"]["exponent"],
                               exponent_r2=prev["impact"]["exponent_r2"],
                               linear_r2=prev["impact"]["linear_r2"],
                               rows=prev["impact"]["rows"])
        cross = stage_cross_section(cfg, snap, cals, cals[ticker], ic, stats,
                                    workers, args.quick, log)
        prev["cross_section"] = (cross.groupby(["ticker", "algo"])
            .agg(mean_bps=("shortfall_adj_bps", "mean"),
                 stderr_bps=("shortfall_adj_bps",
                             lambda x: x.std(ddof=1) / math.sqrt(len(x))),
                 fill_rate=("fill_rate", "mean")).reset_index()
            .replace({np.nan: None}).to_dict(orient="records"))
        (RESULTS / "summary.json").write_text(json.dumps(prev), encoding="utf-8")
        log("cross-section stage rewritten into results/summary.json")
        log_file.close()
        return 0

    ic = stage_impact(cfg, snap, cals, ticker, args.quick, workers, log)

    cal = cals[ticker]
    ac_params = ACParams(X=1.0, T=cfg.execution.horizon_seconds,
                         N=cfg.execution.n_slices,
                         sigma=stats.sigma_per_second(cfg.market.seconds_per_day),
                         eta=ic.eta, gamma=ic.gamma, epsilon=ic.epsilon)
    ctx = {"cfg": cfg, "stats": stats, "flow": cal.flow, "kyle": cal.kyle_lambda,
           "sigma_exo": cal.sigma_exo, "ac_params": ac_params,
           "horizon": cfg.execution.horizon_seconds,
           "n_slices": cfg.execution.n_slices,
           "lambdas": list(cfg.sweep.lambdas), "start_fraction": 0.25}

    sizes = (0.001, 0.005) if args.quick else cfg.sweep.sizes_pct_adv
    vols = (1.0,) if args.quick else cfg.sweep.vol_multipliers
    n_paths = 6 if args.quick else cfg.sweep.paths
    seeds = list(range(10_000, 10_000 + n_paths))

    df = stage_tournament(cfg, ctx, sizes, vols, seeds, workers, log)
    df.to_csv(RESULTS / "tournament_paths.csv", index=False, float_format="%.6g")
    summary = summarise_tournament(df, cfg)
    summary.to_csv(RESULTS / "tournament_summary.csv", index=False,
                   float_format="%.6g")

    # Headline comparison at the default size, base volatility.
    default_size = cfg.sweep.default_size_pct_adv
    if default_size not in sizes:
        default_size = sizes[len(sizes) // 2]
    interval_frac = (cfg.execution.horizon_seconds / cfg.market.seconds_per_day)
    log(f"algorithm comparison at {default_size:.2%} of ADV "
        f"(~{default_size / interval_frac:.0%} of the volume expected in a "
        f"{cfg.execution.horizon_seconds/60:.0f}-minute window), base volatility:")
    head = summary[(summary["size_pct"] == default_size)
                   & (summary["vol_mult"] == 1.0)]
    for _, r in head.iterrows():
        tag = f"{r['algo']}" + (f" (lam={r['lambda']:.0e})"
                                if np.isfinite(r["lambda"]) else "")
        log(f"  {tag:<22} mean {r['mean_bps']:6.2f} +/- {r['stderr_bps']:.2f} bp"
            f"   sd {r['stdev_bps']:6.2f}   CVaR95 {r['cvar95_bps']:7.2f}"
            f"   fill {r['fill_rate']:.3f}")

    # Paired tests against TWAP, which is the benchmark everyone claims to beat.
    pairs = []
    ref = df[(df["algo"] == "TWAP") & (df["size_pct"] == default_size)
             & (df["vol_mult"] == 1.0)].sort_values("seed")
    for (algo, lam), g in df[(df["size_pct"] == default_size)
                             & (df["vol_mult"] == 1.0)].groupby(
                                 ["algo", "lambda"], dropna=False):
        g = g.sort_values("seed")
        t = paired_test(g["shortfall_adj_bps"].to_numpy(),
                        ref["shortfall_adj_bps"].to_numpy())
        pairs.append({"algo": algo, "lambda": lam, **t.to_dict()})
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(RESULTS / "paired_vs_twap.csv", index=False,
                    float_format="%.6g")
    log("paired difference against TWAP (negative = cheaper):")
    for _, r in pairs_df.iterrows():
        tag = f"{r['algo']}" + (f" (lam={r['lambda']:.0e})"
                                if np.isfinite(r["lambda"]) else "")
        log(f"  {tag:<22} {r['mean_diff']:+6.2f} bp   t = {r['t_stat']:+6.2f}"
            f"   wins {r['win_rate']:.0%}")

    model_frontier, sim_frontier = stage_frontier(cfg, ctx, summary, ic, stats,
                                                  default_size, log)

    cross = stage_cross_section(cfg, snap, cals, cal, ic, stats, workers,
                                args.quick, log)

    tape = stage_tape(cfg, ctx, stats, log)
    stage_figures(summary, ic, model_frontier, sim_frontier, cross, log)

    # Histograms for the page, on shared bins per (size, vol).
    hists = []
    for (size, vol), g in df.groupby(["size_pct", "vol_mult"]):
        lo, hi = np.percentile(g["shortfall_bps"], [0.5, 99.5])
        for (algo, lam), gg in g.groupby(["algo", "lambda"], dropna=False):
            h = histogram(gg["shortfall_bps"], bins=cfg.frontend.hist_bins,
                          lo=lo, hi=hi)
            hists.append({"size_pct": size, "vol_mult": vol, "algo": algo,
                          "lambda": (None if not np.isfinite(lam) else lam),
                          "edges": [round(e, 4) for e in h["edges"]],
                          "counts": h["counts"]})

    summary_json = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "as_of": snap.as_of, "source": snap.source,
        "ticker": ticker,
        "names": {t: {"price": snap[t].price,
                      "sigma_annual": snap[t].sigma_annual,
                      "adv_shares": snap[t].adv_shares,
                      "spread_bps_cs": snap[t].spread_bps_cs,
                      "spread_bps_ar": snap[t].spread_bps_ar,
                      "tick_size": snap[t].tick_size,
                      **cals[t].to_dict()} for t in snap.tickers},
        "impact": ic.to_dict(),
        "config": {"horizon": cfg.execution.horizon_seconds,
                   "n_slices": cfg.execution.n_slices,
                   "pov_rate": cfg.execution.pov_rate,
                   "lambdas": list(cfg.sweep.lambdas),
                   "sizes": list(sizes), "vols": list(vols),
                   "paths": len(seeds), "lot": cfg.book.lot_size,
                   "seconds_per_day": cfg.market.seconds_per_day},
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "paired": pairs_df.replace({np.nan: None}).to_dict(orient="records"),
        "frontier_model": model_frontier.replace({np.inf: None, np.nan: None})
        .to_dict(orient="records"),
        "frontier_sim": sim_frontier.replace({np.nan: None}).to_dict(orient="records"),
        "cross_section": cross.groupby(["ticker", "algo"])
        .agg(mean_bps=("shortfall_adj_bps", "mean"),
             stderr_bps=("shortfall_adj_bps", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
             fill_rate=("fill_rate", "mean")).reset_index()
        .replace({np.nan: None}).to_dict(orient="records"),
        "histograms": hists,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary_json),
                                          encoding="utf-8")
    log(f"total runtime {time.time() - t_start:.0f}s")
    log_file.close()
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
