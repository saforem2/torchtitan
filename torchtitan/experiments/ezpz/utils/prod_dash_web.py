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
_state = {"payload": None, "fetched_at": 0.0, "error": None}
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
    if payload is None:
        _refresh()
        with _lock:
            payload = _state["payload"] or {"chains": {}}
            age = 0.0
            err = _state["error"]
    elif age is not None and age > WEB_TTL and not _refreshing.is_set():
        threading.Thread(target=_refresh, daemon=True).start()
    out = dict(payload)
    out["web_age"] = round(age, 1) if age is not None else None
    out["web_error"] = err
    out["live_window"] = pd.LIVE_WINDOW
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
  <div id="chartwrap"><div id="chart"></div></div>
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
  const ch = payload.chains || {};
  return Object.keys(ch).sort((a, b) => {
    const A = ch[a], B = ch[b];
    const ac = A.kind !== "canonical", bc = B.kind !== "canonical";
    if (ac !== bc) return ac - bc;
    const am = A.model || "z", bm = B.model || "z";
    if (am !== bm) return am < bm ? -1 : 1;
    const an = A.num_nodes || 0, bn = B.num_nodes || 0;
    if (an !== bn) return bn - an;
    return a < b ? -1 : 1;
  });
}
const colorFor = key => palette()[chainOrder().indexOf(key) % palette().length];

// Series for one chain on the current metric, with the live tip appended when
// the tip HAS that metric. The tip is parsed from the .o per-step line, which
// carries loss/grad_norm/tps/tflops/mfu but NOT lr -- so the lr curve simply
// ends at the last W&B-synced step rather than being extended with undefined.
function seriesFor(key, c) {
  let s = ((c.series || {})[metric] || []).slice();
  const tip = c.live_tip;
  if (tip && tip[metric] !== undefined && tip.step !== undefined &&
      (!s.length || tip.step > s[s.length - 1][0])) {
    s = s.concat([[tip.step, tip[metric]]]);
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

function draw() {
  if (!payload) return;
  const ch = payload.chains || {};
  const keys = chainOrder().filter(k => !hidden.has(k));
  const logY = document.getElementById("ylog").checked;

  // uPlot wants one shared x array; chains have different step grids, so union
  // the x values and index each chain's y into it (null = no sample there,
  // which uPlot renders as a gap rather than interpolating).
  const prepared = [];
  for (const k of keys) {
    const s = seriesFor(k, ch[k]);
    if (s && s.length > 1) prepared.push([k, s]);
  }
  const xs = [...new Set(prepared.flatMap(([, s]) => s.map(p => p[0])))]
             .sort((a, b) => a - b);
  const data = [xs];
  const series = [{}];
  for (const [k, s] of prepared) {
    const m = new Map(s);
    let ys = xs.map(x => (m.has(x) ? m.get(x) : null));
    if (logY) ys = ys.map(v => (v != null && v > 0 ? v : null));
    data.push(ys);
    const live = isLive(ch[k]);
    series.push({
      label: ch[k].label || k,
      stroke: colorFor(k),
      width: live ? 3 : 2,
      alpha: live ? 1 : 0.55,
      // spanGaps MUST be true. Chains sit on DIFFERENT step grids, so the
      // union x-axis is ~3.5k values of which any one chain occupies ~600 --
      // the other ~83% are nulls meaning "this chain has no sample HERE",
      // not "training gapped". With spanGaps:false every chain renders as
      // isolated points and (points.show:false) draws nothing at all.
      // Spanning connects each chain's own adjacent samples: the same curve
      // matplotlib draws from that chain's private xs array.
      spanGaps: true,
      points: { show: false },
    });
  }

  const yr = document.getElementById("clip").checked
             ? clipRange(data.slice(1)) : null;
  const label = (METRICS.find(m => m[0] === metric) || [, metric])[1];
  const w = document.getElementById("chartwrap").clientWidth;
  const opts = {
    width: w, height: Math.max(340, Math.round(window.innerHeight * 0.62)),
    // x is a step/token COUNT, not a timestamp. Without time:false uPlot
    // formats the axis as dates ("12/31/69" for small step numbers).
    scales: { x: { time: false },
              y: { distr: logY ? 3 : 1, ...(yr ? { range: yr } : {}) } },
    axes: [
      { label: document.getElementById("tokens").checked
               ? "tokens seen (billions)" : "training step (cumulative)",
        stroke: dark() ? "#d5d5d5" : "#333",
        grid: { stroke: dark() ? "#262b33" : "#eee" } },
      { label, stroke: dark() ? "#d5d5d5" : "#333",
        grid: { stroke: dark() ? "#262b33" : "#eee" },
        // ~2e-5 values would all print as "0.00" under the default formatter.
        ...(metric === "lr" ? { values: (u, ts) => ts.map(SCI), size: 70 } : {}) },
    ],
    legend: { show: false },
    series,
  };
  if (chart) { chart.destroy(); chart = null; }
  // Clear FIRST: uPlot appends its canvas here, so a leftover placeholder
  // (or a destroyed chart's node) would stack up across redraws.
  const host = document.getElementById("chart");
  host.innerHTML = "";
  if (data.length > 1) chart = new uPlot(opts, data, host);
  else host.innerHTML = '<p class="meta">no data for this metric</p>';
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
  try {
    const r = await fetch("/api/backbone");
    payload = await r.json();
  } catch (e) {
    document.getElementById("err").textContent = "fetch failed: " + e;
    return;
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
  drawTabs(); drawLegend(); drawBoard(); draw();
}

for (const id of ["tokens", "ylog", "clip"])
  document.getElementById(id).onchange = draw;
document.getElementById("refresh").onclick = async () => {
  await fetch("/api/refresh"); setTimeout(load, 1500);
};
addEventListener("resize", () => { clearTimeout(window._rt);
                                   window._rt = setTimeout(draw, 150); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  drawLegend(); draw();
});
load();
setInterval(load, 30000);
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
