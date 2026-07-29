#!/usr/bin/env python3
"""Textual TUI for the AuroraGPT production dashboard -- multi-metric view.

Companion to ``prod_dash.py`` (which owns the data layer: the remote W&B
aggregator, the cached backbone, and the text board). This module is the
interactive front-end only: it imports ``prod_dash`` for ``fetch()`` +
``render_board()`` and draws switchable per-metric charts
(loss / grad_norm / tps / tflops / mfu) with all production chains overlaid.

Opt-in via ``python prod_dash.py --app`` (which imports this module and calls
``run_app()``). Requires ``textual`` + ``textual-plotext``:

    uv pip install textual textual-plotext

Both are pure-python (NOT torch deps -- safe for the XPU build). If they are not
installed, ``prod_dash.py --app`` falls back to the text board.

Keys: q quit | r refresh (live tails) | ctrl+r force full W&B re-pull |
x step<->tokens x-axis | a show-all experiments |
b/c board/charts pane | t focus run-toggles | T hide/show the run panel |
z cycle focus through runs (bright + fit; wraps to reset) | Z/0 reset view |
p pan/zoom mode | +/- zoom x in/out | h/l pan left/right | j/k pan down/up |
L y-axis log/linear | s zen mode (chart only) | X set xlim | Y set ylim |
left/right (or the tab bar) switch metric | d dark/light |
space (in the run list) toggle a run.
"""
from __future__ import annotations

import os

import prod_dash as pd  # data layer (same utils/ dir -> on sys.path[0])

from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import (
    Footer, Header, Input, RichLog, Static, Tabs, Tab, TabbedContent, TabPane,
    SelectionList,
)
from textual.widgets.selection_list import Selection
from textual.worker import get_current_worker
from textual import work
from textual_plotext import PlotextPlot

# SSH-fallback metric set (short key, axis label) -- the 5 the remote aggregator
# ships in the payload `series` dict. Used only when the local wandb_cache is
# unavailable. When the cache IS available, the metric list is DISCOVERED
# dynamically from the cached DataFrames (all ~24 W&B metrics) -- see
# _refresh_data / _rebuild_metric_tabs.
METRICS = [
    ("loss", "loss (global avg)"),
    ("grad_norm", "grad norm"),
    ("tps", "tokens / sec / GPU"),
    ("tflops", "TFLOPs"),
    ("mfu", "MFU (%)"),
]
_METRIC_KEYS = [m[0] for m in METRICS]

# Map the SSH-fallback short names to the full W&B metric keys the local cache
# uses, so the SAME selected-metric string works whether data came from the
# cache (full keys) or the SSH series (short keys). When the cache is active the
# selector holds full W&B keys directly and this map is bypassed.
_SHORT_TO_WANDB = {
    "loss": "loss_metrics/global_avg_loss",
    "grad_norm": "grad_norm",
    "tps": "throughput(tps)",
    "tflops": "tflops",
    "mfu": "mfu(%)",
}


def _metric_label(key: str) -> str:
    """Human axis label for a metric key (full W&B key or short fallback)."""
    for short, lab in METRICS:
        if key == short:
            return lab
    # full W&B key: strip the group prefix for a compact tab label
    return key


# Textual Tab ids must be valid Python identifiers, but W&B metric keys contain
# '/', '(', ')', '%'. Encode to a safe id and keep a reverse map so a tab
# activation resolves back to the real metric key.
import re as _re

_TAB_ID_TO_METRIC: dict = {}


def _tab_id(metric_key: str) -> str:
    tid = "m_" + _re.sub(r"[^0-9A-Za-z]", "_", metric_key)
    _TAB_ID_TO_METRIC[tid] = metric_key
    return tid

# Curated high-contrast categorical palette (RGB), assigned by STABLE sorted
# chain index so consecutive chains get maximally-different hues -- avoids the
# crc32-hash clustering that made several chains read as the same purple.
# Two hue-PARALLEL palettes, indexed the same way, so a chain keeps its hue
# family when the Textual theme flips -- only the tone shifts for contrast.
# _PALETTE_DARK is the original bright set (tuned for the default dark theme);
# _PALETTE_LIGHT darkens/saturates each hue so it stays legible on a light
# theme background (textual-light etc.). _palette_for() picks per current theme.
_PALETTE_DARK = [
    (66, 135, 245),   # blue
    (245, 133, 24),   # orange
    (84, 196, 75),    # green
    (228, 87, 86),    # red
    (162, 122, 255),  # purple
    (0, 199, 190),    # teal
    (255, 105, 180),  # pink
    (214, 197, 45),   # gold
    (150, 100, 60),   # brown
    (120, 220, 150),  # mint
    (140, 160, 175),  # slate
    (255, 160, 90),   # apricot
]
_PALETTE_LIGHT = [
    (31, 96, 196),    # blue
    (196, 96, 8),     # orange
    (40, 140, 45),    # green
    (196, 40, 40),    # red
    (110, 70, 200),   # purple
    (0, 140, 132),    # teal
    (200, 50, 130),   # pink
    (150, 130, 20),   # gold
    (110, 72, 40),    # brown
    (50, 150, 90),    # mint
    (80, 100, 120),   # slate
    (190, 110, 40),   # apricot
]
# Back-compat alias for any external reference to the original name.
_PALETTE = _PALETTE_DARK


# Light<->dark counterparts within a theme FAMILY, so the `d` toggle flips
# polarity while preserving the user's chosen palette (gruvbox stays gruvbox,
# solarized stays solarized, etc.) instead of always jumping to textual-*.
# Bidirectional: each pair is registered both ways at import. Any theme not in a
# pair falls back to the textual-dark/textual-light default (see _theme_counterpart).
_THEME_PAIRS = [
    ("textual-dark", "textual-light"),
    ("gruvbox", "textual-light"),            # gruvbox has no light sibling in-tree
    ("solarized-dark", "solarized-light"),
    ("atom-one-dark", "atom-one-light"),
    ("rose-pine", "rose-pine-dawn"),
    ("rose-pine-moon", "rose-pine-dawn"),
    ("catppuccin-mocha", "catppuccin-latte"),
    ("catppuccin-macchiato", "catppuccin-latte"),
    ("catppuccin-frappe", "catppuccin-latte"),
    ("nord", "textual-light"),               # no in-tree light sibling
    ("dracula", "textual-light"),
    ("monokai", "textual-light"),
    ("tokyo-night", "textual-light"),
    ("flexoki", "textual-light"),
    ("ansi-light", "textual-dark"),
]
_THEME_COUNTERPART = {}
for _d, _l in _THEME_PAIRS:
    _THEME_COUNTERPART.setdefault(_d, _l)
    _THEME_COUNTERPART.setdefault(_l, _d)


def _dim(rgb, f=0.55, bg=(0, 0, 0)):
    """Fade an RGB color toward the background so idle chains recede.

    The old version multiplied toward BLACK (rgb*f), which correctly dims on a
    dark canvas but on a LIGHT canvas makes a color DARKER -- i.e. higher
    contrast against white, so idle lines stood out MORE, not less. Blending
    toward the actual background color dims correctly in either theme:
    result = rgb*f + bg*(1-f). Default bg=black preserves the old dark-theme
    look; the caller passes the theme background for light mode."""
    return tuple(int(c * f + b * (1.0 - f)) for c, b in zip(rgb, bg))


class ProdDashApp(App):
    """Live multi-metric production dashboard."""

    TITLE = "AuroraGPT production"
    # Top-level tabs ("Charts" / "Board") so the table gets its own pane and no
    # longer competes with the chart for vertical space. Inside Charts: a run
    # SelectionList (left, toggle chains on/off) beside the PlotextPlot (right);
    # #metrictabs is the per-metric selector above them.
    CSS = """
    #metrictabs { dock: top; }
    #runs { width: 34; border-right: solid $panel; }
    #runs:focus-within { border-right: solid $accent; }
    #runs.hidden { display: none; }
    #chart { width: 1fr; }
    #board { height: 1fr; overflow-y: auto; color: $text-muted; padding: 0 1; }
    #log { height: 6; display: none; dock: bottom; }
    #log.building { display: block; }
    #axinput { dock: bottom; display: none; }
    #axinput.active { display: block; }
    /* Zen mode: hide all chrome so only the chart remains. */
    App.zen #metrictabs { display: none; }
    App.zen #runs { display: none; }
    App.zen Header { display: none; }
    App.zen Footer { display: none; }
    App.zen #log { display: none; }
    """
    # priority=True so these app-level keys fire even when a focused child widget
    # (the SelectionList or the scrollable PlotextPlot) would otherwise consume
    # them. Capital X/Y are shift-bindings; keep them priority so they reach us.
    BINDINGS = [
        Binding("q", "quit", "quit", priority=True),
        Binding("r", "refresh", "refresh (live tails)", priority=True),
        Binding("ctrl+r", "force_refresh", "force full re-pull", priority=True),
        Binding("x", "toggle_xaxis", "step/tokens", priority=True),
        Binding("a", "toggle_all", "show-all", priority=True),
        Binding("b", "show_board", "board", priority=True),
        Binding("c", "show_charts", "charts", priority=True),
        Binding("t", "focus_runs", "focus runs", priority=True),
        Binding("T", "toggle_runs_panel", "hide/show runs", priority=True),
        Binding("d", "toggle_theme", "dark/light", priority=True),
        Binding("z", "focus_selected", "focus run", priority=True),
        Binding("Z", "reset_view", "reset view", priority=True),
        Binding("0", "reset_view", "reset view", priority=True),
        Binding("p", "pan_zoom_mode", "pan/zoom", priority=True),
        Binding("X", "set_xlim", "set xlim", priority=True),
        Binding("Y", "set_ylim", "set ylim", priority=True),
        # pan/zoom controls:
        Binding("plus", "zoom_in", "zoom in", priority=True),
        Binding("equals_sign", "zoom_in", show=False, priority=True),
        Binding("minus", "zoom_out", "zoom out", priority=True),
        Binding("h", "pan_left", "pan left", priority=True),
        Binding("l", "pan_right", "pan right", priority=True),
        Binding("j", "pan_down", "pan down", priority=True),
        Binding("k", "pan_up", "pan up", priority=True),
        Binding("L", "toggle_ylog", "y log/linear", priority=True),
        Binding("s", "toggle_zen", "zen mode", priority=True),
    ]

    def __init__(self):
        super().__init__()
        self.metric = "loss"
        self.xaxis = "tokens" if os.environ.get("PD_XAXIS") == "tokens" else "step"
        self.payload = {"chains": {}}
        self._fresh_kicked = False
        self._hidden = set()      # chain keys toggled OFF
        self._runs_built = False  # whether the SelectionList is populated yet
        self._runs_panel_hidden = False  # whole run-toggle panel collapsed
        # view state: explicit axis limits (None,None = autoscale on that axis).
        # xlim clips data in python (plotext xlim IndexErrors when a series has 0
        # points in-window); ylim uses plt.ylim (verified crash-safe). Focus-run
        # and pan/zoom just compute + set these limits.
        self._xlim = (None, None)
        self._ylim = (None, None)
        self._focus_key = None     # chain key the view is fitted to, or None
        self._focus_idx = -1       # z-cycle position into the visible-chain list
        self._ylog = False         # y-axis log scale (L toggles)
        self._zen = False          # zen mode: only the chart, no panels/chrome
        self._force_next = False   # ctrl+r: force a full cache re-pull next fetch

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="panes"):
            with TabPane("Charts", id="pane-charts"):
                # Initial fallback tabs (the 5 SSH metrics); _rebuild_metric_tabs
                # diffs against these once the cache discovers the full set. Use
                # _tab_id() so ids match the rebuild's encoding (else the diff
                # can't dedup them).
                yield Tabs(*[Tab(lab, id=_tab_id(k)) for k, lab in METRICS],
                           id="metrictabs")
                with Horizontal():
                    yield SelectionList(id="runs")
                    yield PlotextPlot(id="chart")
            with TabPane("Board", id="pane-board"):
                yield Static("(loading ...)", id="board")
        yield Input(id="axinput")
        yield RichLog(id="log", highlight=False, markup=False, wrap=True)
        yield Footer()

    def get_system_commands(self, screen):
        """Surface every app binding in the command palette (Ctrl-P).

        Textual only auto-populates the palette with SYSTEM commands, not a
        subclass's BINDINGS, so none of q/r/x/z/d/... showed up. Yield one
        command per binding (skipping hidden ones) with its action wired through
        run_action, alongside the built-in system commands."""
        yield from super().get_system_commands(screen)
        for b in self.BINDINGS:
            action = getattr(b, "action", None)
            desc = getattr(b, "description", "") or ""
            show = getattr(b, "show", True)
            key = getattr(b, "key", "")
            if not action or not show:
                continue
            title = "%s  (%s)" % (desc or action, key)
            yield SystemCommand(
                title, "prod_dash: %s" % (desc or action),
                (lambda a=action: self.run_action(a)),
            )

    def on_mount(self) -> None:
        # Transparent plotext canvas so the chart inherits the Textual widget
        # background (which follows the app theme) instead of the auto-theme
        # path, which rendered a BLACK canvas under a light theme. "textual-clear"
        # sets canvas/axes/ticks to "default" (terminal default = transparent +
        # theme foreground); our explicit per-line colors are unaffected.
        try:
            self.query_one("#chart", PlotextPlot).theme = "textual-clear"
        except Exception:
            pass
        self._refresh_data()
        # Auto-refresh on the same cadence as the kitcat live loop.
        self.set_interval(pd.INTERVAL, self._refresh_data)

    # ---- data (threaded so the ~min-long SSH build never blocks the UI) ----
    @work(exclusive=True, thread=True, group="fetch")
    def _refresh_data(self) -> None:
        worker = get_current_worker()
        self.call_from_thread(self._log_show, True)

        def cb(line):
            if not worker.is_cancelled:
                self.call_from_thread(self._log_line, line)

        data = pd.fetch(stderr_cb=cb)
        # Enrich with the LOCAL all-metric cache (W&B-direct, bypasses SSH). This
        # is best-effort: if wandb_cache / creds / network fail, chains simply
        # keep only the SSH `series` (5 metrics) and the selector stays on the
        # fallback set. On success each chain gains a `metrics` dict
        # {wandb_key: [[step, val], ...]} spanning all ~24 logged metrics.
        if not worker.is_cancelled:
            try:
                self._enrich_from_cache(data, cb)
            except Exception as e:
                cb("wandb_cache enrich skipped: %s: %s" % (type(e).__name__, e))
        if not worker.is_cancelled:
            self.call_from_thread(self._apply_payload, data)

    def _enrich_from_cache(self, data, cb):
        """Attach all-metric series to each payload chain from the local cache.

        Runs on the worker thread. Maps each payload chain (keyed by trajectory
        key) to its cached DataFrame via trajectories.py's run-id lists, then
        stores a `metrics` dict of [[step, value]] pairs per metric column. The
        union of columns seen becomes the dynamic selector set."""
        import wandb_cache as wc
        from torchtitan.experiments.ezpz.utils import trajectories as traj
        chains = (data or {}).get("chains", {})
        if not chains:
            return
        by_key = {t["key"]: t for t in traj.TRAJECTORIES}
        discovered = set()
        counters = {}  # shared -> one cache-hit summary instead of per-run spam
        total_runs = 0
        for key, c in chains.items():
            t = by_key.get(key)
            ids = (t or {}).get("wandb_run_ids") or []
            if not ids:
                continue
            total_runs += len(ids)
            df = wc.fetch_chain(ids, log=cb, counters=counters,
                                force=self._force_next)
            if df is None or df.empty or "_step" not in df.columns:
                continue
            steps = df["_step"].tolist()
            metrics = {}
            for col in wc.metric_columns(df):
                vals = df[col].tolist()
                pairs = [[int(s), float(v)] for s, v in zip(steps, vals)
                         if v == v and s == s]  # drop NaN steps/values
                if len(pairs) >= 2:
                    metrics[col] = pairs
                    discovered.add(col)
            if metrics:
                c["metrics"] = metrics
        if self._force_next:
            cb("cache: FORCE re-pull -- refetched all %d runs from W&B"
               % total_runs)
        elif counters.get("hit"):
            cb("cache: %d/%d runs served from disk (terminal); rest fetched"
               % (counters["hit"], total_runs))
        self._force_next = False  # one-shot: only the ctrl+r fetch forces
        if discovered:
            data["_metric_keys"] = sorted(discovered)

    def _log_show(self, building: bool) -> None:
        log = self.query_one("#log", RichLog)
        log.set_class(building, "building")

    def _log_line(self, line: str) -> None:
        self.query_one("#log", RichLog).write(line)

    def _chain_order(self):
        """Chains in stable display order (canonical first, then by model/nodes).
        The index into this list drives the color assignment so hues stay stable
        and maximally separated across refreshes."""
        def order(item):
            k, c = item
            return (c.get("kind") != "canonical", c.get("model") or "z",
                    -(c.get("num_nodes") or 0), k)
        return sorted(self.payload.get("chains", {}).items(), key=order)

    def _is_dark(self):
        """True if the active Textual theme is dark. current_theme.dark is the
        reliable signal in Textual 8.x (App.dark is deprecated)."""
        try:
            return bool(self.current_theme.dark)
        except Exception:
            return True  # app default is textual-dark

    def _bg_rgb(self):
        """The active theme's background (surface) color as an (r,g,b) tuple, so
        _dim can fade idle chains toward the ACTUAL canvas color (correct in both
        light and dark). Falls back to black/white by theme if it can't parse."""
        try:
            from textual.color import Color
            return Color.parse(self.theme_variables.get("surface")).rgb
        except Exception:
            return (0, 0, 0) if self._is_dark() else (255, 255, 255)

    def _palette_for_theme(self):
        """Pick the hue-parallel palette matching the active Textual theme.

        Falls back to the dark palette if the theme can't be read, matching the
        app's default (textual-dark)."""
        return _PALETTE_DARK if self._is_dark() else _PALETTE_LIGHT

    def _color_for(self, key):
        keys = [k for k, _ in self._chain_order()]
        idx = keys.index(key) if key in keys else 0
        palette = self._palette_for_theme()
        return palette[idx % len(palette)]

    def _rebuild_runs(self):
        """Populate the run-toggle SelectionList from the current chains. Doubles
        as the chart legend: each row carries a color swatch matching that chain's
        plotted line (the in-canvas plotext legend was dropped because it is
        pinned to the top-left corner and covered the early-step data). Preserves
        on/off state across refreshes."""
        from rich.text import Text
        sl = self.query_one("#runs", SelectionList)
        want = [(k, c) for k, c in self._chain_order()]
        sl.clear_options()
        for k, c in want:
            live = (c.get("queue_state") == "R"
                    or (c.get("log_age") is not None
                        and c["log_age"] <= pd.LIVE_WINDOW))
            r, g, b = self._color_for(k)
            swatch = "rgb(%d,%d,%d)" % (r, g, b)
            prompt = Text.assemble(
                ("* " if live else "  "),
                ("━━ ", swatch),  # heavy-line swatch in the line color
                c.get("label", k),
            )
            sl.add_option(Selection(prompt, k, k not in self._hidden))
        sl.border_title = "runs / legend (space=toggle)"
        self._runs_built = True

    def _rebuild_metric_tabs(self):
        """Sync the metric selector to the payload's discovered metric set (all
        ~24 cached W&B metrics) when the cache is active, else the 5 SSH fallback
        metrics. DIFFS the tabs (remove stale, add missing) rather than
        clear()+add: Textual's Tabs.clear() is async (removal happens next pump
        cycle), so a clear()+add in the same call races -- a later rebuild then
        re-adds an id whose old tab is still mounted -> DuplicateIds. Diffing is
        idempotent and race-safe. Preserves the current selection if it lives."""
        keys = self.payload.get("_metric_keys") or _METRIC_KEYS
        if keys == getattr(self, "_metric_tab_keys", None):
            return  # unchanged -> don't churn the Tabs widget
        self._metric_tab_keys = list(keys)
        try:
            tabs = self.query_one("#metrictabs", Tabs)
        except Exception:
            return
        want_ids = {_tab_id(k): k for k in keys}       # id -> metric key
        have_ids = {t.id for t in tabs.query(Tab)}
        for tid in have_ids - set(want_ids):           # remove stale
            try:
                tabs.remove_tab(tid)
            except Exception:
                pass
        for tid, k in want_ids.items():                # add missing
            if tid not in have_ids:
                tabs.add_tab(Tab(_metric_label(k), id=tid))
        # keep the current metric if still present, else default to first
        if self.metric not in keys:
            self.metric = keys[0] if keys else "loss"

    def _apply_payload(self, data: dict) -> None:
        self.payload = data or {"chains": {}}
        self._log_show(False)
        self.query_one("#log", RichLog).clear()
        self._rebuild_metric_tabs()
        self._rebuild_runs()
        self._redraw()
        self.query_one("#board", Static).update(pd.render_board(self.payload))
        # Self-heal a STALE pre-`series` cache: an old backbone (built before the
        # multi-metric change) has `curve` but no `series`, so the board renders
        # but every chart is blank. If no chain carries a series, force ONE fresh
        # rebuild so the next payload has the metric series.
        chains = self.payload.get("chains", {})
        has_series = any(c.get("series") for c in chains.values())
        if chains and not has_series and not self._fresh_kicked:
            self._fresh_kicked = True
            self._log_line("stale cache (no metric series) -- forcing a fresh "
                           "backbone rebuild ...")
            os.environ["PD_FRESH"] = "1"
            self._refresh_data()
        elif has_series:
            os.environ.pop("PD_FRESH", None)  # don't force-rebuild every cycle

    # ---- rendering ----
    def _series_xy(self, key, c):
        """Return the (xs, ys, is_live) a chain contributes to the current
        metric on the current x-axis, or None if it can't be drawn. Includes
        the live-tip point. No zoom clipping here (see _redraw)."""
        # Prefer the local all-metric cache (metrics keyed by full W&B name);
        # fall back to the SSH `series` (5 short-named metrics). self.metric holds
        # a full W&B key when the cache is active, else a short fallback name.
        cache_metrics = c.get("metrics") or {}
        series = cache_metrics.get(self.metric)
        if series is None:
            # try the short<->wandb mapping in either direction
            series = cache_metrics.get(_SHORT_TO_WANDB.get(self.metric, ""))
        if series is None:
            series = (c.get("series") or {}).get(self.metric) or []
        series = list(series)
        # Live-tip augmentation only applies to the 5 SSH short metrics the tip
        # carries (loss/tps/mfu/...). For cache-only metrics there is no tip.
        tip = c.get("live_tip") or {}
        tip_metric = self.metric
        if self.metric not in _METRIC_KEYS:
            # map a full W&B key back to a short tip key if one exists
            inv = {v: k for k, v in _SHORT_TO_WANDB.items()}
            tip_metric = inv.get(self.metric, "")
        if tip and tip_metric in _METRIC_KEYS:
            tv = tip.get(tip_metric if tip_metric != "loss" else "loss")
            ts = tip.get("step")
            if tv is not None and ts is not None and (
                    not series or ts > series[-1][0]):
                series = series + [[ts, tv]]
        if len(series) < 2:
            return None
        toks_per_step = (c.get("gbs") or 0) * (c.get("seq_len") or 0)
        if self.xaxis == "tokens":
            if not toks_per_step:
                return None  # can't place a no-gbs experiment on a tokens axis
            xs = [p[0] * toks_per_step / 1e9 for p in series]
        else:
            xs = [float(p[0]) for p in series]
        ys = [p[1] for p in series]
        is_live = (c.get("queue_state") == "R"
                   or (c.get("log_age") is not None
                       and c["log_age"] <= pd.LIVE_WINDOW))
        return xs, ys, is_live

    def _redraw(self) -> None:
        widget = self.query_one("#chart", PlotextPlot)
        plt = widget.plt
        plt.clear_data()
        plt.clear_figure()

        # draw idle first, live last (so live sits on top); hidden chains skipped
        ordered = self._chain_order()
        live_last = sorted(ordered, key=lambda kc: bool(
            kc[1].get("queue_state") == "R"
            or (kc[1].get("log_age") is not None
                and kc[1]["log_age"] <= pd.LIVE_WINDOW)))

        # Pass 1: gather each visible chain's (xs, ys).
        prepped = []
        for key, c in live_last:
            if key in self._hidden:
                continue
            got = self._series_xy(key, c)
            if got is None:
                continue
            xs, ys, is_live = got
            prepped.append((key, c, xs, ys, is_live))

        xlo, xhi = self._xlim
        ylo, yhi = self._ylim
        # Pass 2: plot. BOTH x- and y-limits are applied by CLIPPING in python,
        # not via plt.xlim/plt.ylim -- plotext's build_plot IndexErrors in its
        # legend loop whenever a plotted series has 0 points inside the limit
        # window (confirmed for both axes). Clipping + skipping empty series
        # sidesteps it entirely; the axes then autoscale to the surviving points,
        # which visually equals the requested window.
        xL = xlo if xlo is not None else float("-inf")
        xH = xhi if xhi is not None else float("inf")
        yL = ylo if ylo is not None else float("-inf")
        yH = yhi if yhi is not None else float("inf")
        clip = (xlo is not None or xhi is not None
                or ylo is not None or yhi is not None)
        bg = self._bg_rgb()
        drawn = 0
        for key, c, xs, ys, is_live in prepped:
            if clip:
                pts = [(x, y) for x, y in zip(xs, ys)
                       if xL <= x <= xH and yL <= y <= yH]
                if len(pts) < 2:
                    continue  # nothing (or a single point) in the window
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
            if self._ylog:
                # Do the log transform OURSELVES (plot log10(y) on a linear axis)
                # rather than plt.yscale("log"): plotext's log path runs log10 over
                # auto-generated y-ticks/limits, not just the data, and raises
                # math-domain on any <=0 it synthesizes. Transforming here (after
                # dropping y<=0) keeps plotext on a plain linear axis it can't
                # choke on; the y-tick labels read as log10 values (see y:log tag).
                import math
                lp = [(x, math.log10(y)) for x, y in zip(xs, ys) if y > 0]
                if len(lp) < 2:
                    continue
                xs = [p[0] for p in lp]
                ys = [p[1] for p in lp]
            color = self._color_for(key)
            # braille everywhere (2x4 sub-cells per char = 8x dot resolution).
            # Dimming rule: when a chain is FOCUSED (z-cycle), it alone stays
            # bright and every other chain is dimmed toward the background;
            # otherwise the usual idle-dims-vs-live-bright rule applies. Fading
            # toward the theme bg makes dimmed lines recede in light + dark.
            if self._focus_key is not None:
                if key != self._focus_key:
                    color = _dim(color, bg=bg)
            elif not is_live:
                color = _dim(color, bg=bg)
            # No plt label=: plotext's legend is hardcoded to the top-left corner
            # (no reposition/disable API) and covered the early-step points of
            # interest. The run SelectionList on the left IS the legend now -- it
            # carries a color swatch per chain (see _rebuild_runs).
            plt.plot(xs, ys, color=color, marker="braille")
            drawn += 1

        axis_label = _metric_label(self.metric)
        if self._ylog:
            axis_label = "log10(%s)" % axis_label
        tags = []
        if self._ylog:
            tags.append("y:log")
        if self._zen:
            tags.append("zen")
        if self._focus_key is not None:
            tags.append("focus: %s" % self._focus_key)
        if xlo is not None or xhi is not None:
            tags.append("x[%s,%s]" % (self._fmt(xlo), self._fmt(xhi)))
        if ylo is not None or yhi is not None:
            tags.append("y[%s,%s]" % (self._fmt(ylo), self._fmt(yhi)))
        if not drawn:
            tags.append("no data")
        suffix = ("  [" + " ".join(tags) + "]") if tags else ""
        plt.title("AuroraGPT production -- %s%s" % (axis_label, suffix))
        plt.xlabel("tokens seen (billions)" if self.xaxis == "tokens"
                   else "training step (cumulative)")
        plt.ylabel(axis_label)
        widget.refresh()

    @staticmethod
    def _fmt(v):
        return "auto" if v is None else ("%g" % v)

    # ---- actions ----
    def action_refresh(self) -> None:
        # Incremental: SSH board/live-tip + cache re-fetch of LIVE runs' new
        # steps only (terminal runs served from disk). Fast, the common case.
        self._refresh_data()

    def action_force_refresh(self) -> None:
        # Full re-pull: ignore the parquet cache and re-scan every run's history
        # from W&B. Use when a crashed run resumed under the same id, or you
        # suspect the cache is stale. Slower (~30-40s for all runs).
        self._force_next = True
        self._refresh_data()

    def action_toggle_xaxis(self) -> None:
        self.xaxis = "step" if self.xaxis == "tokens" else "tokens"
        self._redraw()

    def action_toggle_theme(self) -> None:
        """Flip the theme's polarity, then redraw so the chart palette follows
        the new background.

        Preference order for the target theme:
          1. The theme we flipped AWAY from last time, if it has the opposite
             polarity to the current one -- so ANY theme round-trips exactly,
             including families with no in-tree light sibling (gruvbox, nord: we
             remember them and come straight back instead of snapping to a
             textual-* default).
          2. Otherwise the _THEME_COUNTERPART sibling (solarized-dark<->
             solarized-light, catppuccin-*<->catppuccin-latte, etc.).
          3. Otherwise the textual-dark/textual-light default by polarity."""
        cur = self.theme
        cur_dark = self._is_dark()
        prev = getattr(self, "_prev_theme", None)

        def _polarity(name):
            t = self.available_themes.get(name)
            return None if t is None else bool(getattr(t, "dark", False))

        cand = None
        if prev is not None and prev in self.available_themes \
                and _polarity(prev) == (not cur_dark):
            cand = prev  # exact round-trip to where we last came from
        if cand is None:
            sib = _THEME_COUNTERPART.get(cur)
            if sib is not None and sib in self.available_themes:
                cand = sib
        if cand is None:
            cand = "textual-light" if cur_dark else "textual-dark"

        self._prev_theme = cur  # remember so the next toggle can return here
        self.theme = cand
        self._redraw()

    def action_toggle_all(self) -> None:
        pd.SHOW_ALL = not pd.SHOW_ALL
        os.environ["PD_SHOW_ALL"] = "1" if pd.SHOW_ALL else ""
        self._refresh_data()

    def action_show_board(self) -> None:
        self.query_one("#panes", TabbedContent).active = "pane-board"

    def action_show_charts(self) -> None:
        self.query_one("#panes", TabbedContent).active = "pane-charts"

    def action_focus_runs(self) -> None:
        self.query_one("#panes", TabbedContent).active = "pane-charts"
        if self._runs_panel_hidden:            # un-hide before focusing
            self.action_toggle_runs_panel()
        self.query_one("#runs", SelectionList).focus()

    def action_toggle_runs_panel(self) -> None:
        # Collapse/restore the whole run-toggle sidebar so the chart can use the
        # full width. Hidden chains stay hidden; this only affects the panel.
        self._runs_panel_hidden = not self._runs_panel_hidden
        runs = self.query_one("#runs", SelectionList)
        runs.set_class(self._runs_panel_hidden, "hidden")
        if self._runs_panel_hidden and runs.has_focus:
            self.query_one("#chart", PlotextPlot).focus()
        self._redraw()  # chart reflows to the reclaimed width

    # ---- view: focus one run, pan/zoom, explicit limits ----
    def _selected_key(self):
        """The chain key highlighted in the run list (for focus-run), or the
        first visible chain if the list has no highlight yet."""
        sl = self.query_one("#runs", SelectionList)
        try:
            opt = sl.get_option_at_index(sl.highlighted)
            if opt is not None:
                return opt.value
        except Exception:
            pass
        vis = [k for k, _ in self._chain_order() if k not in self._hidden]
        return vis[0] if vis else None

    def _chain_extent(self, key):
        """(xmin,xmax,ymin,ymax) of one chain on the current metric+x-axis."""
        c = self.payload.get("chains", {}).get(key)
        if not c:
            return None
        got = self._series_xy(key, c)
        if got is None:
            return None
        xs, ys, _ = got
        return min(xs), max(xs), min(ys), max(ys)

    def action_focus_selected(self) -> None:
        # Repeated `z` CYCLES focus through the visible chains: each press
        # advances to the next one, fits the view to its x+y extent, and marks it
        # as the focus (so _redraw draws it BRIGHT and dims the rest). After the
        # last chain it wraps to reset-view (all runs, autoscale).
        vis = [k for k, _ in self._chain_order() if k not in self._hidden]
        if not vis:
            return
        # start the cycle from the list-highlighted run on the very first press
        if self._focus_key is None:
            sel = self._selected_key()
            self._focus_idx = vis.index(sel) if sel in vis else 0
        else:
            self._focus_idx += 1
        if self._focus_idx >= len(vis):
            # wrapped past the end -> clear focus, show everything
            self._focus_idx = -1
            self.action_reset_view()
            return
        key = vis[self._focus_idx]
        ext = self._chain_extent(key)
        if ext is None:
            return
        xmn, xmx, ymn, ymx = ext
        xpad = (xmx - xmn) * 0.02 or 1.0
        ypad = (ymx - ymn) * 0.05 or 0.01
        self._xlim = (xmn - xpad, xmx + xpad)
        self._ylim = (ymn - ypad, ymx + ypad)
        self._focus_key = key
        self._redraw()

    def action_toggle_ylog(self) -> None:
        """Toggle the y-axis between linear and log scale. Log needs positive
        values; if the current metric has non-positive points we still set it
        (plotext drops the bad points) but flag it in the title tag."""
        self._ylog = not self._ylog
        self._redraw()

    def action_toggle_zen(self) -> None:
        """Zen mode: hide the run panel, metric tabs, header, footer, and log --
        just the chart. Toggle back to restore. Uses a CSS class on the app so
        the layout reflows to give the chart the whole frame."""
        self._zen = not self._zen
        self.set_class(self._zen, "zen")
        # in zen we also collapse the run panel so the chart spans full width
        try:
            runs = self.query_one("#runs", SelectionList)
            runs.set_class(self._zen or self._runs_panel_hidden, "hidden")
        except Exception:
            pass
        self._redraw()

    def action_reset_view(self) -> None:
        self._xlim = (None, None)
        self._ylim = (None, None)
        self._focus_key = None
        self._focus_idx = -1
        self._redraw()

    def action_pan_zoom_mode(self) -> None:
        # Seed an explicit x-window from the current data span so +/-/h/l have
        # something to act on, then the user drives it interactively.
        if self._xlim == (None, None):
            span = self._visible_x_span()
            if span:
                self._xlim = span
        self.query_one("#chart", PlotextPlot).focus()
        self._redraw()

    def _visible_x_span(self):
        xs_all = []
        for key, c in self._chain_order():
            if key in self._hidden:
                continue
            got = self._series_xy(key, c)
            if got:
                xs_all += got[0]
        return (min(xs_all), max(xs_all)) if xs_all else None

    def _cur_xwin(self):
        """Current [lo,hi] x-window, filling autoscale bounds from the data."""
        span = self._visible_x_span() or (0.0, 1.0)
        lo = self._xlim[0] if self._xlim[0] is not None else span[0]
        hi = self._xlim[1] if self._xlim[1] is not None else span[1]
        return lo, hi

    def _visible_y_span(self):
        ys_all = []
        for key, c in self._chain_order():
            if key in self._hidden:
                continue
            got = self._series_xy(key, c)
            if got:
                ys_all += got[1]
        return (min(ys_all), max(ys_all)) if ys_all else None

    def _cur_ywin(self):
        """Current [lo,hi] y-window, filling autoscale bounds from the data."""
        span = self._visible_y_span() or (0.0, 1.0)
        lo = self._ylim[0] if self._ylim[0] is not None else span[0]
        hi = self._ylim[1] if self._ylim[1] is not None else span[1]
        return lo, hi

    def action_zoom_in(self) -> None:
        lo, hi = self._cur_xwin()
        c = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 * 0.7   # shrink window to 70%
        self._xlim = (c - half, c + half)
        self._focus_key = None
        self._redraw()

    def action_zoom_out(self) -> None:
        lo, hi = self._cur_xwin()
        c = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 / 0.7   # grow window
        self._xlim = (c - half, c + half)
        self._focus_key = None
        self._redraw()

    def action_pan_left(self) -> None:
        lo, hi = self._cur_xwin()
        d = (hi - lo) * 0.25
        self._xlim = (lo - d, hi - d)
        self._focus_key = None
        self._redraw()

    def action_pan_right(self) -> None:
        lo, hi = self._cur_xwin()
        d = (hi - lo) * 0.25
        self._xlim = (lo + d, hi + d)
        self._focus_key = None
        self._redraw()

    def action_pan_up(self) -> None:
        lo, hi = self._cur_ywin()
        d = (hi - lo) * 0.25
        self._ylim = (lo + d, hi + d)
        self._focus_key = None
        self._redraw()

    def action_pan_down(self) -> None:
        lo, hi = self._cur_ywin()
        d = (hi - lo) * 0.25
        self._ylim = (lo - d, hi - d)
        self._focus_key = None
        self._redraw()

    def action_set_xlim(self) -> None:
        self._prompt_axis("x")

    def action_set_ylim(self) -> None:
        self._prompt_axis("y")

    def _prompt_axis(self, axis) -> None:
        self._axis_target = axis
        cur = self._xlim if axis == "x" else self._ylim
        inp = self.query_one("#axinput", Input)
        inp.value = ""
        lo = "" if cur[0] is None else "%g" % cur[0]
        hi = "" if cur[1] is None else "%g" % cur[1]
        inp.placeholder = ("%slim>  min max   (blank=auto, e.g. '%s %s'; "
                           "empty line = reset)" % (axis, lo or "auto", hi or "auto"))
        inp.add_class("active")
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "axinput":
            return
        raw = event.value.strip()
        inp = self.query_one("#axinput", Input)
        inp.remove_class("active")
        axis = getattr(self, "_axis_target", "x")
        if not raw:
            # empty -> reset that axis to autoscale
            if axis == "x":
                self._xlim = (None, None)
            else:
                self._ylim = (None, None)
        else:
            parts = raw.replace(",", " ").split()

            def parse(tok):
                if tok in ("", "-", "auto", "_"):
                    return None
                try:
                    return float(tok)
                except ValueError:
                    return None
            lo = parse(parts[0]) if len(parts) >= 1 else None
            hi = parse(parts[1]) if len(parts) >= 2 else None
            if axis == "x":
                self._xlim = (lo, hi)
            else:
                self._ylim = (lo, hi)
        self._focus_key = None
        self.query_one("#chart", PlotextPlot).focus()
        self._redraw()

    def on_key(self, event) -> None:
        # Escape cancels an open axis-limit prompt without applying it.
        inp = self.query_one("#axinput", Input)
        if event.key == "escape" and inp.has_class("active"):
            inp.remove_class("active")
            self.query_one("#chart", PlotextPlot).focus()
            event.stop()

    def on_selection_list_selected_changed(
            self, event: SelectionList.SelectedChanged) -> None:
        # the SelectionList holds the set of VISIBLE chain keys -> hidden is the
        # complement. Recompute + redraw on every toggle.
        selected = set(event.selection_list.selected)
        all_keys = {k for k, _ in self._chain_order()}
        self._hidden = all_keys - selected
        self._redraw()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        # Only the per-metric selector (#metrictabs) drives the chart; ignore
        # the top-level Charts/Board TabbedContent's own tab events.
        if getattr(event.tabs, "id", None) != "metrictabs":
            return
        if event.tab is None:
            return
        tid = event.tab.id
        # Cache-active tabs use encoded ids (m_<sanitized>); fallback tabs use
        # the short metric name directly as the id.
        if tid in _TAB_ID_TO_METRIC:
            self.metric = _TAB_ID_TO_METRIC[tid]
            self._redraw()
        elif tid in _METRIC_KEYS:
            self.metric = tid
            self._redraw()


def run_app() -> None:
    # honor --all (prod_dash.main set PD_SHOW_ALL / SHOW_ALL already)
    if os.environ.get("PD_SHOW_ALL") == "1":
        pd.SHOW_ALL = True
    ProdDashApp().run()


if __name__ == "__main__":
    run_app()
