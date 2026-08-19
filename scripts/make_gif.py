"""Record ``results/interface.gif`` - the README's moving picture.

    python scripts/make_gif.py

Each frame is one headless-Chrome screenshot of ``docs/index.html`` with a small
script injected that sets controls and fires their events exactly as a visitor's
hand would. The page recomputes on those events, so the frames show the real
engine responding rather than a storyboard.

One Chrome process per frame is slower than driving a persistent session, but it
needs no ``selenium``/``playwright`` dependency and it re-proves on every frame
the thing that actually matters: **the page loads from ``file://``, cold, with
no server.** Pillow then assembles the PNGs.

Because of that last property this doubles as an end-to-end smoke test. If a
frame comes back blank the page is broken from ``file://``, and the script fails
rather than shipping a GIF of a blank rectangle.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"
OUT = ROOT / "results" / "interface.gif"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

WIDTH, HEIGHT = 1280, 1040
FRAME_MS = 1900
HOLD_LAST_MS = 3400

FRAMES = [
    ("the book, mid-execution", "tab('book'); setv('cFrame', 120);"),
    ("further in - the ask side thinning as the order works",
     "tab('book'); setv('cFrame', 300);"),
    ("the schedules: Almgren-Chriss against TWAP, VWAP and POV", "tab('sched');"),
    ("more urgent - the trajectory bends and the cost rises",
     "tab('sched'); setv('cLam', -4.6);"),
    ("a bigger parent order, and the cost decomposition shifts",
     "tab('sched'); setv('cSize', 0.02);"),
    ("the efficient frontier, with the simulated points on it", "tab('frontier');"),
    ("cost distributions from the order book simulation", "tab('costs');"),
    ("ranked on the tail rather than the mean",
     "tab('costs'); pick('cStat', 'cvar95_bps');"),
    ("what free daily data cannot tell you about the spread", "tab('calib');"),
    ("break it: a quarter of the window's volume in one order",
     "tab('stress'); setv('cSSize', 0.05);"),
    ("break it: one slice, and the cost model stops being convex",
     "tab('stress'); setv('cSN', 1);"),
]

HARNESS = """
<script>
window.addEventListener('load', function(){
  function fire(el, name){ el.dispatchEvent(new Event(name, {bubbles:true})); }
  function setv(id, v){ var el=document.getElementById(id); if(!el)return;
    el.value=v; fire(el,'input'); fire(el,'change'); }
  function pick(id, v){ setv(id, v); }
  function click(id){ var el=document.getElementById(id); if(el) el.click(); }
  function tab(name){ var b=document.querySelector('nav button[data-tab="'+name+'"]');
    if(b) b.click(); }
  setTimeout(function(){ __ACTION__ }, 1400);
});
</script>
"""


def find_browser():
    for p in BROWSERS:
        if Path(p).exists():
            return p
    return shutil.which("google-chrome") or shutil.which("chromium")


def main() -> int:
    browser = find_browser()
    if browser is None:
        print("No Chrome/Chromium/Edge found - cannot record the GIF.", file=sys.stderr)
        return 1
    if not INDEX.exists() or not (INDEX.parent / "data.js").exists():
        print("docs/index.html or docs/data.js missing - run scripts/build_frontend.py first",
              file=sys.stderr)
        return 1
    try:
        from PIL import Image
    except ImportError:
        print("pillow is not installed: pip install pillow", file=sys.stderr)
        return 1

    shots = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        docs = tmp / "docs"
        shutil.copytree(INDEX.parent, docs)
        html = INDEX.read_text(encoding="utf-8")

        for i, (caption, action) in enumerate(FRAMES):
            page = docs / f"frame{i}.html"
            page.write_text(
                html.replace("</body>", HARNESS.replace("__ACTION__", action) + "</body>"),
                encoding="utf-8")
            shot = tmp / f"frame{i}.png"
            subprocess.run(
                [browser, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                 f"--window-size={WIDTH},{HEIGHT}", "--virtual-time-budget=30000",
                 f"--screenshot={shot}", page.as_uri()],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
            if not shot.exists():
                print(f"frame {i} ({caption}) produced no screenshot", file=sys.stderr)
                return 1
            im = Image.open(shot).convert("RGB").copy()
            # A page that failed to boot renders as one flat colour. Catch it
            # here rather than shipping a GIF of a blank rectangle.
            if len(im.getcolors(maxcolors=1 << 20) or [(0, 0)]) < 12:
                print(f"frame {i} ({caption}) is blank - the page did not render",
                      file=sys.stderr)
                return 1
            print(f"  frame {i}: {caption}")
            shots.append(im)

    # Half size and a 96-colour palette: the page is flat blocks of colour and
    # thin chart lines, so this is visually lossless at a fraction of the bytes.
    frames = [im.resize((WIDTH // 2, HEIGHT // 2), Image.LANCZOS)
                .quantize(colors=96, dither=Image.Dither.NONE) for im in shots]
    durations = [FRAME_MS] * (len(frames) - 1) + [HOLD_LAST_MS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], loop=0,
                   duration=durations, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB, "
          f"{len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
