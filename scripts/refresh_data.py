"""Re-pull the daily bar snapshot that calibrates the simulator.

    python scripts/refresh_data.py

Writes ``data/market_snapshot.csv`` plus a ``.meta.txt`` recording the source
and the as-of date. This is the only script in the repo that touches the
network; everything else reads the committed CSV, which is what makes a research
run reproducible after the fact.

Without a network, or without ``yfinance`` installed, the existing cache is left
alone and the script says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.market import fetch_yfinance, load_cached, write_cache  # noqa: E402
from src.utils.config import AppConfig  # noqa: E402


def main() -> int:
    cfg = AppConfig.load().market
    snap = fetch_yfinance(cfg)
    if snap is None:
        existing = load_cached(cfg)
        where = "the existing cache is unchanged" if existing else "no cache exists"
        print(f"could not fetch from yfinance - {where}", file=sys.stderr)
        return 1
    path = write_cache(snap, cfg)
    print(f"wrote {path.relative_to(ROOT)}  "
          f"({len(snap.tickers)} names, as of {snap.as_of}, "
          f"{snap.n_days} daily bars)")
    for t in snap.tickers:
        s = snap[t]
        print(f"  {t:<5} ${s.price:>8.2f}  vol {s.sigma_annual:>6.1%}  "
              f"ADV {s.adv_shares/1e6:>7.2f}M sh  spread {s.spread_bps:>6.2f} bp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
