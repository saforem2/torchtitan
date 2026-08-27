#!/usr/bin/env python3
"""Browser view of the production dashboard (localhost only).

Third rendering mode alongside `prod_dash.py` (kitcat/SVG/board) and
`prod_dash_app.py` (Textual TUI). All three share ONE data layer -- this module
calls `prod_dash.fetch()` and serves the payload verbatim -- so the web view
cannot drift from the committed charts or the TUI. Nothing here knows how to
talk to W&B, PBS, or Lustre; that all stays in prod_dash's remote aggregator.

Why a browser at all: the TUI is better over SSH, but a browser gives pan/zoom,
hover-to-read-a-point, and a link you can hand to someone.

    python torchtitan/experiments/ezpz/utils/prod_dash_web.py
    # -> http://127.0.0.1:8712

BINDS TO 127.0.0.1 ONLY, and that is deliberate. This serves ALCF training
telemetry and has NO authentication of any kind. Exposing it beyond localhost
is a policy question, not a technical one -- so `--host` refuses anything but a
loopback address unless PD_WEB_ALLOW_PUBLIC=1 is set explicitly. To view it from
a laptop while it runs on a login node, forward the port instead:

    ssh -L 8712:127.0.0.1:8712 aurora

Dependencies: stdlib only on the python side. uPlot (~51 KB, MIT) is vendored
under `webassets/` and served locally -- no CDN, so the page works on an air-
gapped host and pins the version.
"""
from __future__ import annotations

import argparse
import http.server
import ipaddress
import json
import os
import socketserver
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import prod_dash as pd  # noqa: E402

ASSETS = os.path.join(_HERE, "webassets")

# Serve the last good payload while a refresh runs, so a browser hitting
# /api/backbone during a cold build gets stale-but-real data instead of a
# multi-minute hang. Mirrors the aggregator's own stale-while-revalidate rule.
_state = {"payload": None, "fetched_at": 0.0, "error": None, "revision": 0}
_lock = threading.Lock()
_refreshing = threading.Event()

# How long a cached payload is served before a background refresh is kicked.
# The expensive W&B work is already cached cluster-side (PD_BACKBONE_TTL); this
# only bounds how often we pay the ~25 s SSH round-trip.
WEB_TTL = float(os.environ.get("PD_WEB_TTL", "60"))


def _refresh(force=False):
    """Populate _state from prod_dash.fetch(). Safe to call concurrently."""
    if _refreshing.is_set() and not force:
        return
    _refreshing.set()
    try:
        payload = pd.fetch()
        with _lock:
            # Keep the previous payload on a failed fetch rather than blanking
            # the page -- an empty {"chains": {}} would render as "everything
            # stopped", which is a materially different (and false) claim.
            if payload.get("chains"):
                _state["payload"] = payload
                _state["fetched_at"] = time.time()
                _state["error"] = None
                # Lets the browser distinguish a genuinely new backbone from
                # the same cached payload returned by its 30 s status poll.
                # Without this, it destroyed and rebuilt every uPlot twice per
                # WEB_TTL even though no chart data had changed.
                _state["revision"] += 1
            else:
                _state["error"] = "aggregator returned no chains"
    except Exception as e:  # noqa: BLE001 - surface any failure to the page
        with _lock:
            _state["error"] = repr(e)
    finally:
        _refreshing.clear()


def _payload_json() -> bytes:
    with _lock:
        payload = _state["payload"]
        age = time.time() - _state["fetched_at"] if _state["fetched_at"] else None
        err = _state["error"]
        revision = _state["revision"]
    if payload is None:
        _refresh()
        with _lock:
            payload = _state["payload"] or {"chains": {}}
            age = 0.0
            err = _state["error"]
            revision = _state["revision"]
    elif age is not None and age > WEB_TTL and not _refreshing.is_set():
        threading.Thread(target=_refresh, daemon=True).start()
    out = dict(payload)
    out["web_age"] = round(age, 1) if age is not None else None
    out["web_error"] = err
    out["web_revision"] = revision
    out["live_window"] = pd.LIVE_WINDOW
    return json.dumps(out).encode()


def _status_json() -> bytes:
    """Return cheap polling metadata without serializing every chart point."""
    with _lock:
        payload = _state["payload"] or {}
        fetched_at = _state["fetched_at"]
        err = _state["error"]
        revision = _state["revision"]
    age = time.time() - fetched_at if fetched_at else None
    if payload and age is not None and age > WEB_TTL and not _refreshing.is_set():
        threading.Thread(target=_refresh, daemon=True).start()
    # These are the only payload-level fields load() needs to update its status
    # line between revisions. `chains` is intentionally excluded: it dominates
    # both JSON serialization and transfer size.
    out = {k: payload.get(k) for k in
           ("built_age", "stale", "stale_age_hours", "stale_reason")}
    out.update({
        "web_age": round(age, 1) if age is not None else None,
        "web_error": err,
        "web_revision": revision,
    })
    return json.dumps(out).encode()


class Handler(http.server.BaseHTTPRequestHandler):
    # Quiet: one log line per asset fetch is noise in the terminal running this.
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8",
                       {"Cache-Control": "no-store"})
        elif path == "/api/backbone":
            self._send(200, _payload_json(), "application/json",
                       {"Cache-Control": "no-store"})
        elif path == "/api/status":
            self._send(200, _status_json(), "application/json",
                       {"Cache-Control": "no-store"})
        elif path == "/api/refresh":
            threading.Thread(target=_refresh, kwargs={"force": True},
                             daemon=True).start()
            self._send(202, b'{"kicked":true}', "application/json")
        elif path == "/favicon.ico":
            # Empty 204 rather than a 404 on every single page load.
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path in ("/uPlot.iife.min.js", "/uPlot.min.css"):
            fp = os.path.join(ASSETS, os.path.basename(path))
            try:
                with open(fp, "rb") as f:
                    body = f.read()
            except OSError:
                self._send(404, b"asset missing", "text/plain")
                return
            ctype = ("text/javascript" if path.endswith(".js")
                     else "text/css")
            self._send(200, body, ctype, {"Cache-Control": "max-age=86400"})
        else:
            self._send(404, b"not found", "text/plain")


INDEX_HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>AuroraGPT production</title>
<link rel="stylesheet" href="/uPlot.min.css">
<style>
:root {
  --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --panel:#f7f7f7;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e6e6e6; --muted:#9aa0a6; --line:#2c3038; --panel:#1b1e24; }
}
* { box-sizing: border-box; }
body {
  margin:0; background:var(--bg); color:var(--fg);
  font: 13px/1.45 "Iosevka Term", ui-monospace, SFMono-Regular, Menlo, monospace;
}
header { padding:10px 14px; border-bottom:1px solid var(--line); display:flex;
         gap:14px; align-items:center; flex-wrap:wrap; }
h1 { font-size:14px; margin:0; font-weight:600; }
.tabs { display:flex; gap:4px; flex-wrap:wrap; }
.tab { padding:3px 10px; border:1px solid var(--line); border-radius:4px;
       cursor:pointer; background:var(--panel); color:var(--fg); font:inherit; }
.tab[aria-selected="true"] { background:#4c78a8; color:#fff; border-color:#4c78a8; }
.spacer { flex:1 }
.meta { color:var(--muted); font-size:12px; }
main { display:flex; gap:12px; padding:12px; align-items:flex-start;
       flex-wrap:wrap; }
/* min-width:0 on BOTH flex children is load-bearing: a flex item defaults to
   min-width:auto, i.e. it refuses to shrink below its content's intrinsic
   width. The board table is nowrap, so its 396px intrinsic width pushed the
   280px aside 116px past its own box and overflowed the document by 104px
   (measured). Letting the items shrink is what actually stops that. */
#chartwrap { flex:1 1 640px; min-width:340px; }
/* Every metric at once, in a responsive grid. auto-fit + minmax means the
   column count follows the window: 3-up on a wide screen, 1-up on a laptop,
   with no breakpoint list to maintain. Each cell owns a <div> that uPlot
   appends its canvas into. */
.grid { display:grid; gap:10px;
        grid-template-columns:repeat(auto-fit, minmax(360px, 1fr)); }
.cell { border:1px solid var(--line); border-radius:5px; padding:6px 6px 2px;
        background:var(--bg); }
.cell h4 { margin:0 0 4px 2px; font-size:12px; font-weight:600;
           color:var(--muted); display:flex; align-items:center; gap:6px; }
.cell h4 .chart-title { flex:1; }
.expand-chart { border:0; background:transparent; color:var(--muted);
                cursor:pointer; font:16px/1 monospace; padding:1px 4px;
                border-radius:3px; }
.expand-chart:hover { color:var(--fg); background:var(--panel); }
/* Maximized charts stay in this page rather than opening another browser
   window (which would need to duplicate payload/visibility state). Rebuilding
   the uPlot after applying this class gives it the actual viewport-sized
   canvas; CSS scaling a fixed canvas would make labels and hover positions
   blurry/wrong. */
body.chart-expanded { overflow:hidden; }
.cell.expanded { position:fixed; inset:12px; z-index:1000; padding:10px;
                 border-color:#4c78a8;
                 box-shadow:0 8px 40px rgba(0,0,0,.38); }
.cell.expanded h4 { font-size:14px; margin-bottom:8px; }
.cell .plot { width:100%; }
/* The focused metric reads first: full width, taller, above the grid. */
#focuswrap { margin-bottom:12px; }
h3.sec { margin:16px 0 6px 2px; font-size:12px; font-weight:600;
         color:var(--muted); }
aside { flex:0 1 380px; min-width:0; max-width:100%; }
.legend { border:1px solid var(--line); border-radius:5px; overflow:hidden; }
.legend div { display:flex; gap:8px; align-items:center; padding:5px 9px;
              cursor:pointer; border-bottom:1px solid var(--line); }
.legend div:last-child { border-bottom:0 }
.legend div.off { opacity:.38 }
.sw { width:11px; height:11px; border-radius:2px; flex:0 0 auto }
.nm { flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis }
.live { color:#2ea043; font-weight:600 }
/* The table scrolls INSIDE its own box rather than widening the page. The
   numeric columns stay nowrap (a wrapped "92772" is unreadable); only the
   chain-name column is allowed to ellipsize, since the legend directly above
   already spells every name out in full. */
.boardwrap { margin-top:12px; overflow-x:auto; max-width:100% }
table { border-collapse:collapse; width:100%; font-size:12px }
th,td { text-align:left; padding:3px 6px; border-bottom:1px solid var(--line);
        white-space:nowrap }
th { color:var(--muted); font-weight:600 }
/* first column = chain label: the only one that may be trimmed */
td:first-child, th:first-child {
  max-width:190px; overflow:hidden; text-overflow:ellipsis }
/* right-align the numbers so columns read as columns */
td:nth-child(n+2), th:nth-child(n+2) { text-align:right }
/* a live chain's row, matching the legend badge */
tr.r-live td:first-child { color:#2ea043; font-weight:600 }
#err { color:#e45756; padding:0 14px }
label.ctl { color:var(--muted); cursor:pointer; user-select:none }
</style>

<header>
  <h1>AuroraGPT production</h1>
  <div class="tabs" id="tabs"></div>
  <label class="ctl"><input type="checkbox" id="tokens"> tokens axis</label>
  <label class="ctl"><input type="checkbox" id="ylog"> log y</label>
  <label class="ctl"><input type="checkbox" id="clip"> clip outliers</label>
  <span class="spacer"></span>
  <span class="meta" id="meta">loading…</span>
  <button class="tab" id="refresh">refresh</button>
</header>
<div id="err"></div>
<main>
  <div id="chartwrap">
    <div id="focuswrap"><div id="chart"></div></div>
    <h3 class="sec" id="allsec">all metrics</h3>
    <div class="grid" id="allgrid"></div>
    <h3 class="sec" id="evalsec">eval benchmarks vs training step</h3>
    <div class="grid" id="evalgrid"></div>
  </div>
  <aside>
    <div class="legend" id="legend"></div>
    <div class="boardwrap"><table id="board"></table>
<h3 class="meta">eval scores (newest evaluated ckpt; acc_norm where applicable, gsm8k = exact_match)</h3>
<table id="evals"></table></div>
  </aside>
</main>

<script src="/uPlot.iife.min.js"></script>
<script>
// Palettes mirror prod_dash.COLORS_{LIGHT,DARK}. Chain -> color is by POSITION
// in the sorted chain list, matching prod_dash_app._color_for (which uses the
// chain-order index, not crc32 -- the SVG plotter is the one that uses crc32).
const LIGHT = ["#4c78a8","#f58518","#54a24b","#e45756","#72b7b2","#b279a2",
               "#ff9da6","#9d755d","#bab0ac","#edc948","#b07aa1","#86bcb6"];
const DARK  = ["#6ea8dc","#ffa94d","#7bc96f","#ff7b7b","#5fd0c8","#d29fd8",
               "#ffc0c8","#c79a6a","#d5cfca","#f6e05e","#d3a0ce","#a8ded6"];
const METRICS = [["loss","loss (global avg)"],["grad_norm","grad norm"],
                 ["tps","tokens / sec / GPU"],["tflops","TFLOPs"],["mfu","MFU (%)"],
                 ["lr","learning rate"]];
// LR values are ~2e-5, so the default axis formatter renders every tick as
// "0.00". Force exponential notation for this metric only.
const SCI = v => (v == null ? "" : v.toExponential(2));

let payload = null, metric = "loss", hidden = new Set(), chart = null;
let expandedPanel = null;
let loadInFlight = false;
let orderRevision = null, orderedKeys = [];
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;
const palette = () => dark() ? DARK : LIGHT;

// Liveness comes from the CURRENT job, never from `log_age`.
//
// `log_age` is the mtime of this chain's HISTORICAL .o logs, taken from the
// cached path index -- for a chain resuming after days idle it is days old
// even while the chain trains right now. Gating on it reported 0 live while
// four chains were running (measured). It is a "when did we last see this
// chain at all" figure, not a heartbeat.
//
// The real heartbeat is `live_tip.age`: live_layer computes it from the
// RUNNING job's own log and already marks a quiet seat "stale" (an umbrella
// keeps running after one trainer dies). So:
//   - queue_state "stale"/Q/H/E  -> not live, whatever else says
//   - queue_state R + fresh tip  -> live
//   - queue_state R + no tip yet -> live (venv prestage: started, nothing
//     written yet -- this is the window the umbrella fix addresses)
//   - no queue_state             -> fall back to log_age, which is all we have
//     for a chain the live layer never matched.
const STALE_STATES = new Set(["stale", "Q", "H", "E"]);
const isLive = c => {
  if (STALE_STATES.has(c.queue_state)) return false;
  const win = payload.live_window ?? 300;
  if (c.queue_state === "R") {
    const tip = c.live_tip;
    if (tip && tip.age !== null && tip.age !== undefined) return tip.age <= win;
    return true;                    // running, no tip observed yet
  }
  return c.log_age !== null && c.log_age !== undefined && c.log_age <= win;
};

// Sorted chain keys: canonical first, then by model/nodes -- same ordering rule
// as render_board so the legend, the table, and the colors all agree.
function chainOrder() {
  if (orderRevision === payload.web_revision) return orderedKeys;
  const ch = payload.chains || {};
  orderedKeys = Object.keys(ch).sort((a, b) => {
    const A = ch[a], B = ch[b];
    const ac = A.kind !== "canonical", bc = B.kind !== "canonical";
    if (ac !== bc) return ac - bc;
    const am = A.model || "z", bm = B.model || "z";
    if (am !== bm) return am < bm ? -1 : 1;
    const an = A.num_nodes || 0, bn = B.num_nodes || 0;
    if (an !== bn) return bn - an;
    return a < b ? -1 : 1;
  });
  orderRevision = payload.web_revision;
  return orderedKeys;
}
const colorFor = key => palette()[chainOrder().indexOf(key) % palette().length];

// Series for one chain on the current metric, with the live tip appended when
// the tip HAS that metric. The tip is parsed from the .o per-step line, which
// carries loss/grad_norm/tps/tflops/mfu but NOT lr -- so the lr curve simply
// ends at the last W&B-synced step rather than being extended with undefined.
function seriesFor(key, c, mk) {
  const met = mk || metric;
  let s = ((c.series || {})[met] || []).slice();
  const tip = c.live_tip;
  if (tip && tip[met] !== undefined && tip.step !== undefined &&
      (!s.length || tip.step > s[s.length - 1][0])) {
    s = s.concat([[tip.step, tip[met]]]);
  }
  if (!s.length) return null;
  const tps = (c.gbs || 0) * (c.seq_len || 0);
  if (document.getElementById("tokens").checked) {
    if (!tps) return null;          // no gbs -> can't place on a tokens axis
    // A stage-2 chain restarts its step counter at 1, but its weights already
    // carry the parent run's tokens (dolmino seeds from stage-1 step-46429).
    // "tokens seen" is CUMULATIVE, so it must start where the parent ended --
    // plotting from 0 claims it saw its first token alongside stage-1.
    // Steps stay per-chain, which is why only this branch offsets.
    const prior = (c.prior_tokens || 0) / 1e9;
    return s.map(p => [prior + p[0] * tps / 1e9, p[1]]);
  }
  return s;
}

// Eval scores come from c.evals.history (written by prod_dash._eval_scores),
// not c.series -- different source, same [[step, value]] shape. Kept separate
// because an eval point exists only where a checkpoint was actually evaluated,
// which is a far sparser grid than the per-step training metrics.
function evalSeriesFor(key, c, task) {
  const h = ((c.evals || {}).history || {})[task];
  if (!h || h.length < 2) return null;
  const tps = (c.gbs || 0) * (c.seq_len || 0);
  if (document.getElementById("tokens").checked) {
    if (!tps) return null;
    const prior = (c.prior_tokens || 0) / 1e9;
    return h.map(p => [prior + p[0] * tps / 1e9, p[1]]);
  }
  return h.slice();
}

// p1/p99 across VISIBLE chains. Throughput metrics spike hard on checkpoint
// steps (a save inflates that step's wall-clock, so tps/tflops/mfu collapse);
// those points are real, but they drag the y-axis and paint full-height lines.
// Clipping the AXIS keeps every point in the data and just bounds the view.
function clipRange(cols) {
  const all = [];
  for (const ys of cols) for (const y of ys) if (y != null && isFinite(y)) all.push(y);
  if (all.length < 20) return null;
  all.sort((a, b) => a - b);
  const lo = all[Math.floor(all.length * 0.01)];
  const hi = all[Math.floor(all.length * 0.99)];
  if (!(hi > lo)) return null;
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad];
}

// Eval metrics are bounded probabilities, but pinning every panel to [0, 1]
// hides most of the movement (MMLU lives in a few points around 0.25 and
// gsm8k is often very close to zero). Fit each panel to its visible data while
// retaining a minimum five-percentage-point window and the legal [0, 1]
// bounds. The minimum window keeps a flat/two-point series from producing a
// misleading microscope-scale axis.
function scoreRange(cols) {
  const all = [];
  for (const ys of cols) for (const y of ys)
    if (y != null && isFinite(y)) all.push(y);
  if (!all.length) return [0, 1];

  const dataLo = Math.min(...all), dataHi = Math.max(...all);
  const span = Math.max(0.05, (dataHi - dataLo) * 1.20);
  const mid = (dataLo + dataHi) / 2;
  let lo = mid - span / 2, hi = mid + span / 2;
  // Shift rather than merely clamp at a boundary, preserving the requested
  // viewing span for scores clustered at exactly zero or one.
  if (lo < 0) { hi -= lo; lo = 0; }
  if (hi > 1) { lo -= hi - 1; hi = 1; }
  return [Math.max(0, lo), Math.min(1, hi)];
}

// ONE chart builder for every panel. `getter(key, chain)` returns that chain's
// [[x, y], ...] for whichever metric this panel shows, so the focus chart, the
// all-metrics grid and the eval grid share identical axis, color, liveness and
// gap-spanning behavior instead of drifting apart.
function buildChart(host, label, getter, opts2) {
  const o = opts2 || {};
  const ch = payload.chains || {};
  const keys = chainOrder().filter(k => !hidden.has(k));
  const logY = o.noLog ? false : document.getElementById("ylog").checked;

  // uPlot wants one shared x array; chains have different step grids, so union
  // the x values and index each chain's y into it (null = no sample there,
  // which uPlot renders as a gap rather than interpolating).
  const prepared = [];
  for (const k of keys) {
    const ser = getter(k, ch[k]);
    if (ser && ser.length > 1) prepared.push([k, ser]);
  }
  host.innerHTML = "";
  if (!prepared.length) {
    host.innerHTML = '<p class="meta">no data</p>';
    return null;
  }
  const xs = [...new Set(prepared.flatMap(([, ser]) => ser.map(p => p[0])))]
             .sort((a, b) => a - b);
  const data = [xs];
  const series = [{}];
  for (const [k, ser] of prepared) {
    const m = new Map(ser);
    let ys = xs.map(x => (m.has(x) ? m.get(x) : null));
    if (logY) ys = ys.map(v => (v != null && v > 0 ? v : null));
    data.push(ys);
    const live = isLive(ch[k]);
    series.push({
      label: ch[k].label || k,
      stroke: colorFor(k),
      width: live ? (o.thin ? 2 : 3) : (o.thin ? 1.4 : 2),
      alpha: live ? 1 : 0.55,
      // spanGaps MUST be true. Chains sit on DIFFERENT step grids, so the
      // union x-axis is ~3.5k values of which any one chain occupies ~600 --
      // the other ~83% are nulls meaning "this chain has no sample HERE",
      // not "training gapped". With spanGaps:false every chain renders as
      // isolated points and (points.show:false) draws nothing at all.
      spanGaps: true,
      // Eval points are sparse (one per evaluated ckpt, often <40 total), so
      // showing the markers tells you where a real measurement sits rather
      // than implying a continuous curve.
      points: { show: !!o.points },
    });
  }

  const yr = (!o.noClip && document.getElementById("clip").checked)
             ? clipRange(data.slice(1)) : null;
  const fixedRange = typeof o.range === "function"
                     ? o.range(data.slice(1)) : o.range;
  const opts = {
    width: o.width, height: o.height,
    // x is a step/token COUNT, not a timestamp. Without time:false uPlot
    // formats the axis as dates ("12/31/69" for small step numbers).
    scales: { x: { time: false },
              y: { distr: logY ? 3 : 1,
                   ...(fixedRange ? { range: fixedRange }
                                  : (yr ? { range: yr } : {})) } },
    axes: [
      { label: o.thin ? "" : (document.getElementById("tokens").checked
               ? "tokens seen (billions)" : "training step (cumulative)"),
        stroke: dark() ? "#d5d5d5" : "#333",
        grid: { stroke: dark() ? "#262b33" : "#eee" },
        size: o.thin ? 28 : 50 },
      { label: o.thin ? "" : label, stroke: dark() ? "#d5d5d5" : "#333",
        grid: { stroke: dark() ? "#262b33" : "#eee" },
        // ~2e-5 values would all print as "0.00" under the default formatter.
        ...(o.sci ? { values: (u, ts) => ts.map(SCI), size: 62 } : {}) },
    ],
    legend: { show: false },
    series,
  };
  return new uPlot(opts, data, host);
}

let gridCharts = [];

// A cell's inner width after layout. Falls back to a sane default when the
// element is not laid out yet (display:none section, or a draw() that lands
// before first paint) -- uPlot throws on width 0.
function hostW(host) {
  return Math.max(240, Math.floor(host.clientWidth) || 360);
}

// Build a small-multiple cell with an in-page maximize/restore button. The
// caller appends the cell before measuring `host`, so fixed-position expanded
// cells and ordinary grid cells both report their real rendered width.
function chartCell(title, panelKey) {
  const cell = document.createElement("div");
  const expanded = expandedPanel === panelKey;
  cell.className = "cell" + (expanded ? " expanded" : "");

  const heading = document.createElement("h4");
  const text = document.createElement("span");
  text.className = "chart-title";
  text.textContent = title;
  const button = document.createElement("button");
  button.className = "expand-chart";
  button.type = "button";
  button.textContent = expanded ? "×" : "⛶";
  button.title = expanded ? "Restore chart (Esc)" : "Maximize chart";
  button.setAttribute("aria-label", button.title);
  button.onclick = () => {
    expandedPanel = expanded ? null : panelKey;
    document.body.classList.toggle("chart-expanded", !!expandedPanel);
    draw();
  };
  heading.append(text, button);

  const host = document.createElement("div");
  host.className = "plot";
  cell.append(heading, host);
  return [cell, host, expanded];
}

document.addEventListener("keydown", e => {
  if (e.key === "Escape" && expandedPanel) {
    expandedPanel = null;
    document.body.classList.remove("chart-expanded");
    draw();
  }
});

// uPlot renders to a fixed-pixel canvas: it does NOT reflow with its container.
// The CSS grid happily re-columns on resize, which left canvases at their old
// width overlapping their neighbours' axes. Redraw on container resize instead.
// Debounced through rAF + a timer so a click-drag resize coalesces into one
// rebuild rather than one per resize event (each rebuild destroys and
// re-creates every chart).
let resizeTimer = null;
let lastW = {};      // element id -> width at last redraw

// draw() rewrites #allgrid/#evalgrid innerHTML, which resizes them, which
// re-fires the observer. Without this guard that is an unbounded rebuild loop.
// Only a WIDTH change can invalidate a canvas, so height churn (cells being
// added, a section unhiding) is ignored.
function widthChanged() {
  let changed = false;
  for (const id of ["focuswrap", "allgrid", "evalgrid"]) {
    const el = document.getElementById(id);
    if (!el) continue;
    const w = Math.floor(el.clientWidth);
    if (w > 0 && lastW[id] !== w) { lastW[id] = w; changed = true; }
  }
  // Expanded chart height follows the viewport too. The ordinary grid ignores
  // height-only changes to avoid observer loops, but a maximized chart should
  // reflow when a laptop is rotated or browser chrome changes available space.
  if (expandedPanel) {
    const vh = Math.floor(window.innerHeight);
    if (lastW.viewportH !== vh) { lastW.viewportH = vh; changed = true; }
  }
  return changed;
}

function scheduleRedraw() {
  if (resizeTimer) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    resizeTimer = null;
    requestAnimationFrame(() => { if (payload && widthChanged()) draw(); });
  }, 120);
}

function initResizeObserver() {
  if (typeof ResizeObserver === "undefined") {   // very old browser
    window.addEventListener("resize", scheduleRedraw);
    return;
  }
  // Observe the containers, not window: this also catches the aside wrapping
  // to a new flex row, which changes chart width without changing window size.
  const ro = new ResizeObserver(scheduleRedraw);
  for (const id of ["focuswrap", "allgrid", "evalgrid"]) {
    const el = document.getElementById(id);
    if (el) ro.observe(el);
  }
}

function draw() {
  if (!payload) return;
  // Measure the element the canvas actually lives in, not its grandparent.
  // #chartwrap is a flex item that wraps; #focuswrap is the real content box.
  const w = document.getElementById("focuswrap").clientWidth;

  // ---- focus chart: the tab-selected metric, full width ----
  if (chart) { chart.destroy(); chart = null; }
  const label = (METRICS.find(m => m[0] === metric) || [, metric])[1];
  chart = buildChart(document.getElementById("chart"), label,
                     (k, c) => seriesFor(k, c),
                     { width: w,
                       height: Math.max(300, Math.round(window.innerHeight * 0.46)),
                       sci: metric === "lr" });

  // Destroy before rebuilding: uPlot keeps window resize + pointer listeners
  // per instance, so redrawing without destroy() leaks one listener set per
  // refresh (the board auto-refreshes, so that compounds).
  gridCharts.forEach(c => { try { c.destroy(); } catch (e) {} });
  gridCharts = [];

  // ---- all metrics, small multiples ----
  const grid = document.getElementById("allgrid");
  grid.innerHTML = "";
  // Append every cell BEFORE sizing any chart. Re-deriving the column count in
  // JS duplicates the CSS `minmax()` and drifted from it (JS said 380, CSS says
  // 360), so a canvas could be laid out wider than the cell holding it. Letting
  // the browser lay the grid out and then reading back each host's real
  // clientWidth keeps the two in sync by construction -- change the CSS and the
  // canvases follow, with no second copy of the math to keep aligned.
  const pending = [];
  for (const [mk, mlab] of METRICS) {
    if (mk === metric) continue;              // already the focus chart
    const [cell, host, expanded] = chartCell(mlab, `metric:${mk}`);
    grid.appendChild(cell);
    pending.push([host, mk, mlab, expanded]);
  }
  for (const [host, mk, mlab, expanded] of pending) {
    const c = buildChart(host, mlab, (k, ch) => seriesFor(k, ch, mk),
                         { width: hostW(host),
                           height: expanded ? window.innerHeight - 76 : 190,
                           thin: !expanded,
                           sci: mk === "lr" });
    if (c) gridCharts.push(c);
  }

  // ---- eval benchmarks, one panel per task ----
  const eg = document.getElementById("evalgrid");
  eg.innerHTML = "";
  const chs = payload.chains || {};
  const tasks = [];
  for (const k of chainOrder()) {
    const h = ((chs[k] || {}).evals || {}).history || {};
    for (const t of Object.keys(h)) if (!tasks.includes(t)) tasks.push(t);
  }
  const PREF = ["hellaswag", "arc_challenge", "arc_easy", "mmlu", "gsm8k",
                "winogrande", "piqa"];
  const ordered = PREF.filter(t => tasks.includes(t))
                      .concat(tasks.filter(t => !PREF.includes(t)));
  document.getElementById("evalsec").style.display = ordered.length ? "" : "none";
  const epending = [];
  for (const t of ordered) {
    const [cell, host, expanded] = chartCell(t, `eval:${t}`);
    eg.appendChild(cell);
    epending.push([host, t, expanded]);
  }
  for (const [host, t, expanded] of epending) {
    // Accuracy is a probability, so use a data-fitted axis clamped to [0,1].
    // scoreRange also enforces a five-point minimum span, preventing a sparse
    // or flat curve from filling the cell with a meaningless microscope zoom.
    // Never log-scaled or outlier-clipped: every eval point is real.
    const c = buildChart(host, t, (k, ch) => evalSeriesFor(k, ch, t),
                         { width: hostW(host),
                           height: expanded ? window.innerHeight - 76 : 190,
                           thin: !expanded,
                           points: true, noLog: true, noClip: true,
                           range: scoreRange });
    if (c) gridCharts.push(c);
  }
}

function drawLegend() {
  const ch = payload.chains || {};
  const el = document.getElementById("legend");
  el.innerHTML = "";
  for (const k of chainOrder()) {
    const c = ch[k], row = document.createElement("div");
    if (hidden.has(k)) row.className = "off";
    row.innerHTML =
      `<span class="sw" style="background:${colorFor(k)}"></span>` +
      `<span class="nm">${c.label || k}</span>` +
      (isLive(c) ? '<span class="live">live</span>' : "");
    row.onclick = () => { hidden.has(k) ? hidden.delete(k) : hidden.add(k);
                          drawLegend(); draw(); };
    el.appendChild(row);
  }
}

function drawBoard() {
  const ch = payload.chains || {};
  const rows = chainOrder().map(k => {
    const c = ch[k], tip = c.live_tip || {};
    const st = c.queue_state || (isLive(c) ? "run" : "idle");
    const f = (v, d = 0) => (v === null || v === undefined) ? "-" : (+v).toFixed(d);
    const nm = c.label || k;
    // `st` is dropped from the table: the legend directly above already shows
    // live/idle as a badge, and at 6 columns the aside clipped %tgt and tps.
    // title= keeps the full name readable when the column ellipsizes.
    return `<tr class="${isLive(c) ? "r-live" : ""}">` +
           `<td title="${nm}">${nm}</td>` +
           `<td>${c.latest_step ?? "-"}</td>` +
           `<td>${f(c.latest_loss, 4)}</td>` +
           `<td>${c.pct_target == null ? "-" : c.pct_target.toFixed(1) + "%"}</td>` +
           `<td>${f(tip.tps ?? c.wb_tps)}</td></tr>`;
  }).join("");
  document.getElementById("board").innerHTML =
    "<tr><th>chain</th><th>step</th><th>loss</th><th>%tgt</th><th>tps</th></tr>" + rows;

  // Eval scores: a SECOND table rather than extra columns. Tasks vary per
  // chain, so a merged table would be mostly empty cells, and the union of
  // task columns changes as backfills land.
  const evEl = document.getElementById("evals");
  if (!evEl) return;
  const PREF = ["hellaswag", "arc_challenge", "arc_easy", "mmlu", "gsm8k",
                "winogrande", "piqa"];
  const scored = chainOrder()
    .map(k => [k, ch[k]])
    .filter(([k, c]) => c && c.evals && c.evals.scores &&
                        Object.keys(c.evals.scores).length);
  if (!scored.length) {
    // Say WHY it is empty -- a blank panel reads as a bug.
    evEl.innerHTML = "<tr><td class='meta'>no evaluated checkpoints yet</td></tr>";
    return;
  }
  const seen = [];
  scored.forEach(([k, c]) => Object.keys(c.evals.scores).forEach(
      t => { if (!seen.includes(t)) seen.push(t); }));
  const cols = PREF.filter(t => seen.includes(t))
                   .concat(seen.filter(t => !PREF.includes(t)));
  evEl.innerHTML =
    "<tr><th>chain</th><th>step</th>" +
    cols.map(t => `<th>${t}</th>`).join("") + "</tr>" +
    scored.map(([k, c]) => {
      const nm = c.label || k;
      return `<tr><td title="${nm}">${nm}</td><td>${c.evals.step ?? "-"}</td>` +
        cols.map(t => `<td>${c.evals.scores[t] == null ? "-"
                          : c.evals.scores[t].toFixed(3)}</td>`).join("") +
        "</tr>";
    }).join("");
}

function drawTabs() {
  const el = document.getElementById("tabs");
  el.innerHTML = "";
  for (const [k, lab] of METRICS) {
    const b = document.createElement("button");
    b.className = "tab"; b.textContent = lab;
    b.setAttribute("aria-selected", k === metric);
    b.onclick = () => { metric = k; drawTabs(); draw(); };
    el.appendChild(b);
  }
}

async function load() {
  // A slow cold refresh can outlive the polling interval. Never stack another
  // fetch/render pipeline on top of one already in progress.
  if (loadInFlight) return;
  loadInFlight = true;
  let changed = !payload;
  try {
    // Once initialized, poll the tiny status document first. Download and
    // parse the full history only when the server says its revision changed.
    const statusURL = payload ? "/api/status" : "/api/backbone";
    const statusResp = await fetch(statusURL);
    if (!statusResp.ok) throw new Error(`${statusURL}: HTTP ${statusResp.status}`);
    const status = await statusResp.json();
    changed = !payload || status.web_revision !== payload.web_revision;
    if (payload && changed) {
      const dataResp = await fetch("/api/backbone");
      if (!dataResp.ok) throw new Error(`/api/backbone: HTTP ${dataResp.status}`);
      payload = await dataResp.json();
    } else if (payload) {
      Object.assign(payload, status);
    } else {
      payload = status;
    }
  } catch (e) {
    document.getElementById("err").textContent = "fetch failed: " + e;
    return;
  } finally {
    loadInFlight = false;
  }

  const n = Object.keys(payload.chains || {}).length;
  const live = Object.values(payload.chains || {}).filter(isLive).length;
  const bb = payload.built_age == null ? "-" : Math.round(payload.built_age / 60) + "m";
  document.getElementById("meta").textContent =
    `${live} live / ${n} chains · backbone ${bb} old · payload ${payload.web_age ?? "?"}s`;
  // A stale payload rendering as if it were live is the real hazard here: the
  // "live" count above is computed from timestamps in a cache that may be
  // hours old, so it will happily claim chains are running when the cluster is
  // unreachable. Say so unmissably rather than trusting the reader to notice
  // the backbone age.
  const errEl = document.getElementById("err");
  if (payload.stale) {
    const h = payload.stale_age_hours ?? "?";
    errEl.textContent =
      `⚠ OFFLINE — showing cached data from ${h}h ago (${payload.stale_reason ||
        "aggregator unreachable"}). The live/chain counts above are NOT current.`;
    errEl.style.color = "#d62728";
    errEl.style.fontWeight = "600";
  } else {
    errEl.textContent = payload.web_error || "";
    errEl.style.color = "";
    errEl.style.fontWeight = "";
  }
  // Keep lightweight age/error text current on every poll, but only rebuild
  // the fixed-size uPlot canvases and tables when the server installed a new
  // backbone. A typical unchanged poll now does no chart allocation at all.
  if (changed) { drawTabs(); drawLegend(); drawBoard(); draw(); }
}

for (const id of ["tokens", "ylog", "clip"])
  document.getElementById(id).onchange = draw;
document.getElementById("refresh").onclick = async () => {
  await fetch("/api/refresh"); setTimeout(load, 1500);
};
initResizeObserver();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  drawLegend(); draw();
});
load();
// Background tabs need neither JSON transfers nor canvas work. Refresh once
// immediately when the user returns, then resume the ordinary cadence.
setInterval(() => { if (!document.hidden) load(); }, 30000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) load();
});
</script>
"""


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _check_host(host: str) -> str:
    """Refuse a non-loopback bind unless explicitly overridden.

    This serves unauthenticated ALCF telemetry. Binding 0.0.0.0 on a login node
    exposes it to everyone on the network, which is a decision for a human to
    make deliberately, not a default to trip over."""
    if os.environ.get("PD_WEB_ALLOW_PUBLIC") == "1":
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    sys.stderr.write(
        "prod_dash_web: refusing to bind %r -- this server has NO auth and\n"
        "serves production telemetry. Use an SSH tunnel instead:\n"
        "    ssh -L 8712:127.0.0.1:8712 <host>\n"
        "Override with PD_WEB_ALLOW_PUBLIC=1 only if you mean it.\n" % host)
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PD_WEB_PORT", "8712")))
    ap.add_argument("--host", default=os.environ.get("PD_WEB_HOST", "127.0.0.1"))
    ap.add_argument("--no-prefetch", action="store_true",
                    help="skip the startup fetch (first page load pays for it)")
    args = ap.parse_args()
    host = _check_host(args.host)

    if not args.no_prefetch:
        print("prod_dash_web: priming payload (first fetch may take ~30 s) ...")
        _refresh()
        with _lock:
            n = len((_state["payload"] or {}).get("chains", {}))
            err = _state["error"]
        print("prod_dash_web: %d chain(s)%s" % (n, "  ERROR: %s" % err if err else ""))

    with _Server((host, args.port), Handler) as httpd:
        print("prod_dash_web: http://%s:%d  (Ctrl-C to stop)" % (host, args.port))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
