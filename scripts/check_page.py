r"""Verify ``docs/`` is shippable, and that the JavaScript port agrees with ``src/``.

    python scripts/check_page.py

Two jobs.

**Static checks.** The page must open from ``file://`` with the network off:
required files present, no ``fetch()`` of a local path, no absolute-URL script
tag without a vendored fallback, total size inside the 8 MB budget.

**Parity.** ``docs/index.html`` contains a JavaScript port of the
Almgren-Chriss solver, the schedule builders and the intraday volume curve, so
the page can recompute the real thing on every slider move. A port is a second
implementation, and a second implementation is a liability unless it is pinned
to the first. This script extracts the port, runs it under Node against the
same inputs, and compares every output with what the Python library returns.

Needs ``node`` on PATH for the parity half. Without it the static checks still
run and the parity section reports as skipped, so this is safe to run anywhere.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.almgren_chriss import (ACParams, expected_cost,  # noqa: E402
                                          cost_variance, kappa, schedule_cost,
                                          solve, trajectory)
from src.execution.schedules import TWAP, VWAP  # noqa: E402
from src.utils.config import AppConfig, FlowConfig, u_shape  # noqa: E402

DOCS = ROOT / "docs"
INDEX = DOCS / "index.html"
DATA = DOCS / "data.js"

PORT_START = "const V = {"
PORT_END = "/* =========================================================================="

FAIL = []
def check(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def static_checks() -> None:
    print("static checks")
    check(INDEX.exists(), "docs/index.html exists")
    check(DATA.exists(), "docs/data.js exists")
    check((DOCS / "README_LOCAL.txt").exists(), "docs/README_LOCAL.txt exists")
    vendor = list((DOCS / "vendor").glob("*.js"))
    check(bool(vendor), f"vendored charting library present ({len(vendor)} file(s))")

    html = INDEX.read_text(encoding="utf-8")
    check("fetch(" not in html.replace("// fetch(", ""),
          "no fetch() anywhere in the page")
    check("XMLHttpRequest" not in html, "no XMLHttpRequest")
    # A remote <script> is only acceptable inside the document.write fallback,
    # which runs solely when the vendored copy failed to define Plotly. Any
    # other one would make the page depend on a network it will not have.
    remote = [line.strip() for line in html.splitlines()
              if re.search(r'<script[^>]*src="http', line)
              and "document.write" not in line]
    check(not remote, f"no unconditional remote script tags (found {remote})")
    check('src="vendor/' in html, "loads the vendored library first")
    check('src="data.js"' in html, "loads data.js by script tag")
    check("cdn.plot.ly" in html, "keeps a CDN fallback for a broken checkout")

    total = sum(p.stat().st_size for p in DOCS.rglob("*") if p.is_file())
    check(total < 8 * 1024 * 1024,
          f"docs/ is {total / 1024 / 1024:.2f} MB, budget 8 MB")

    if DATA.exists():
        txt = DATA.read_text(encoding="utf-8")
        check(txt.lstrip().startswith("/*") and "window.DATA" in txt,
              "data.js assigns window.DATA")
        check("NaN" not in txt and "Infinity" not in txt,
              "data.js contains no NaN/Infinity literals")


def extract_port(html: str) -> str:
    i = html.index(PORT_START)
    j = html.index(PORT_END, i)
    return html[i:j]


def parity_checks() -> None:
    print("parity: JavaScript port against src/")
    node = shutil.which("node")
    if node is None:
        print("  skip  node is not on PATH")
        return

    html = INDEX.read_text(encoding="utf-8")
    port = extract_port(html)
    # sumTo lives just below the port block and the port uses it.
    port += "\nfunction sumTo(arr,k){let s=0;for(let i=0;i<k;i++)s+=arr[i];return s;}\n"

    cases = []
    for X, T, N, sigma, eta, gamma, eps in [
            (250_000.0, 1800.0, 30, 0.065, 1.75e-4, 2.6e-7, 0.01),
            (50_000.0, 600.0, 10, 0.02, 1.0e-5, 1.0e-7, 0.005),
            (1_000_000.0, 3600.0, 60, 0.01, 1.0e-5, 1.0e-7, 0.005),
            (10_000.0, 300.0, 5, 0.004, 5.0e-5, 3.0e-7, 0.002)]:
        for lam in (0.0, 1e-8, 1e-7, 1e-6, 1e-5):
            cases.append({"X": X, "T": T, "N": N, "sigma": sigma, "eta": eta,
                          "gamma": gamma, "epsilon": eps, "lam": lam})

    cfg = FlowConfig()
    us = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    script = port + """
const cases = %s;
const us = %s;
const out = {ac: [], u: []};
for(const c of cases){
  const p = {X:c.X,T:c.T,N:c.N,sigma:c.sigma,eta:c.eta,gamma:c.gamma,epsilon:c.epsilon};
  const s = V.solve(p, c.lam);
  const twap = V.scheduleCost(p, V.planTWAP(p));
  out.ac.push({kappa:s.kappa, cost:s.cost, stdev:s.stdev,
               n0:s.n[0], nlast:s.n[p.N-1], xmid:s.x[Math.floor(p.N/2)],
               twapCost:twap.cost, twapSd:twap.stdev});
}
const uc = {u_a:%r, u_b:%r, u_p:%r};
for(const u of us) out.u.push(V.uShape(u, uc));
console.log(JSON.stringify(out));
""" % (json.dumps(cases), json.dumps(us), cfg.u_a, cfg.u_b, cfg.u_p)

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "port.js"
        f.write_text(script, encoding="utf-8")
        res = subprocess.run([node, str(f)], capture_output=True, text=True,
                             timeout=120)
    if res.returncode != 0:
        check(False, f"node failed: {res.stderr.strip()[:400]}")
        return
    js = json.loads(res.stdout)

    worst = 0.0
    for c, got in zip(cases, js["ac"]):
        p = ACParams(X=c["X"], T=c["T"], N=c["N"], sigma=c["sigma"],
                     eta=c["eta"], gamma=c["gamma"], epsilon=c["epsilon"])
        s = solve(p, c["lam"])
        twap = schedule_cost(p, np.full(p.N, p.X / p.N))
        pairs = [(s["kappa"], got["kappa"]), (s["expected_cost"], got["cost"]),
                 (s["stdev"], got["stdev"]), (s["n"][0], got["n0"]),
                 (s["n"][-1], got["nlast"]), (s["x"][p.N // 2], got["xmid"]),
                 (twap["expected_cost"], got["twapCost"]),
                 (twap["stdev"], got["twapSd"])]
        for a, b in pairs:
            denom = max(abs(a), 1e-12)
            worst = max(worst, abs(a - b) / denom)
    check(worst < 1e-9,
          f"Almgren-Chriss port matches src/ on {len(cases)} cases "
          f"(worst relative error {worst:.2e})")

    u_py = [float(u_shape(u, cfg)) for u in us]
    err = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(u_py, js["u"]))
    check(err < 1e-12, f"volume curve port matches src/ (worst {err:.2e})")


def data_checks() -> None:
    print("data.js contents")
    if not DATA.exists():
        check(False, "data.js is missing - run build_frontend.py")
        return
    txt = DATA.read_text(encoding="utf-8")
    payload = txt[txt.index("window.DATA =") + len("window.DATA ="):].rstrip().rstrip(";")
    D = json.loads(payload)
    for key in ("meta", "config", "names", "impact", "summary", "frontier_model",
                "frontier_sim", "cross_section", "histograms", "tape"):
        check(key in D, f"data.js has '{key}'")
    check(len(D["tape"]["frames"]) > 50,
          f"animation has {len(D['tape']['frames'])} frames")
    check(len(D["summary"]) >= 9, f"{len(D['summary'])} grid rows shipped")
    check(D["meta"]["as_of"] not in ("", None), "data is dated")
    algos = {r["algo"] for r in D["summary"]}
    check(algos == {"TWAP", "VWAP", "POV", "AC", "Adaptive"},
          f"all five algorithms present ({sorted(algos)})")


def main() -> int:
    static_checks()
    data_checks()
    parity_checks()
    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed:")
        for m in FAIL:
            print("  - " + m)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
