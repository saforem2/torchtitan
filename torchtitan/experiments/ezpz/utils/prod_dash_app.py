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

Keys: q quit | r refresh | x step<->tokens x-axis | a show-all experiments |
b/c board/charts pane | t focus run-toggles | T hide/show the run panel |
z fit the highlighted run (x+y) | Z/0 reset view | p pan/zoom mode |
+/- zoom x in/out | h/l pan left/right | X set xlim | Y set ylim |
left/right (or the tab bar) switch metric | space (in the run list) toggle a run.
"""
from __future__ import annotations

import os

import prod_dash as pd  # data layer (same utils/ dir -> on sys.path[0])

from textual.app import App, ComposeResult
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

# (short key, axis label) in tab order. Matches prod_dash._METRIC_MAP shorts.
METRICS = [
    ("loss", "loss (global avg)"),
    ("grad_norm", "grad norm"),
    ("tps", "tokens / sec / GPU"),
    ("tflops", "TFLOPs"),
    ("mfu", "MFU (%)"),
]
_METRIC_KEYS = [m[0] for m in METRICS]

# Curated high-contrast categorical palette (RGB), assigned by STABLE sorted
# chain index so consecutive chains get maximally-different hues -- avoids the
# crc32-hash clustering that made several chains read as the same purple.
_PALETTE = [
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


def _dim(rgb, f=0.55):
    return tuple(int(c * f) for c in rgb)


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
    """
    # priority=True so these app-level keys fire even when a focused child widget
    # (the SelectionList or the scrollable PlotextPlot) would otherwise consume
    # them. Capital X/Y are shift-bindings; keep them priority so they reach us.
    BINDINGS = [
        Binding("q", "quit", "quit", priority=True),
        Binding("r", "refresh", "refresh", priority=True),
        Binding("x", "toggle_xaxis", "step/tokens", priority=True),
        Binding("a", "toggle_all", "show-all", priority=True),
        Binding("b", "show_board", "board", priority=True),
        Binding("c", "show_charts", "charts", priority=True),
        Binding("t", "focus_runs", "focus runs", priority=True),
        Binding("T", "toggle_runs_panel", "hide/show runs", priority=True),
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="panes"):
            with TabPane("Charts", id="pane-charts"):
                yield Tabs(*[Tab(label, id=key) for key, label in METRICS],
                           id="metrictabs")
                with Horizontal():
                    yield SelectionList(id="runs")
                    yield PlotextPlot(id="chart")
            with TabPane("Board", id="pane-board"):
                yield Static("(loading ...)", id="board")
        yield Input(id="axinput")
        yield RichLog(id="log", highlight=False, markup=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
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
        if not worker.is_cancelled:
            self.call_from_thread(self._apply_payload, data)

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

    def _color_for(self, key):
        keys = [k for k, _ in self._chain_order()]
        idx = keys.index(key) if key in keys else 0
        return _PALETTE[idx % len(_PALETTE)]

    def _rebuild_runs(self):
        """Populate the run-toggle SelectionList from the current chains (once);
        preserve on/off state across refreshes."""
        sl = self.query_one("#runs", SelectionList)
        want = [(k, c) for k, c in self._chain_order()]
        sl.clear_options()
        for k, c in want:
            live = (c.get("queue_state") == "R"
                    or (c.get("log_age") is not None
                        and c["log_age"] <= pd.LIVE_WINDOW))
            tag = ("* " if live else "  ") + c.get("label", k)
            sl.add_option(Selection(tag, k, k not in self._hidden))
        sl.border_title = "runs (space=toggle)"
        self._runs_built = True

    def _apply_payload(self, data: dict) -> None:
        self.payload = data or {"chains": {}}
        self._log_show(False)
        self.query_one("#log", RichLog).clear()
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
        series = (c.get("series") or {}).get(self.metric) or []
        tip = c.get("live_tip") or {}
        if tip and self.metric in _METRIC_KEYS:
            tv = tip.get(self.metric if self.metric != "loss" else "loss")
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
        drawn = 0
        for key, c, xs, ys, is_live in prepped:
            if clip:
                pts = [(x, y) for x, y in zip(xs, ys)
                       if xL <= x <= xH and yL <= y <= yH]
                if len(pts) < 2:
                    continue  # nothing (or a single point) in the window
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
            color = self._color_for(key)
            label = ("* " if is_live else "") + c.get("label", key) + (
                "" if is_live else " (idle)")
            # braille everywhere (2x4 sub-cells per char = 8x dot resolution);
            # idle chains dimmed to keep the live one salient.
            if not is_live:
                color = _dim(color)
            plt.plot(xs, ys, color=color, label=label, marker="braille")
            drawn += 1

        axis_label = dict(METRICS)[self.metric]
        tags = []
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
        self._refresh_data()

    def action_toggle_xaxis(self) -> None:
        self.xaxis = "step" if self.xaxis == "tokens" else "tokens"
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
        # Fit the axes to the highlighted run's full x AND y extent (a small
        # margin), so a single curve fills the frame; other runs stay drawn.
        key = self._selected_key()
        ext = self._chain_extent(key) if key else None
        if ext is None:
            return
        xmn, xmx, ymn, ymx = ext
        xpad = (xmx - xmn) * 0.02 or 1.0
        ypad = (ymx - ymn) * 0.05 or 0.01
        self._xlim = (xmn - xpad, xmx + xpad)
        self._ylim = (ymn - ypad, ymx + ypad)
        self._focus_key = key
        self._redraw()

    def action_reset_view(self) -> None:
        self._xlim = (None, None)
        self._ylim = (None, None)
        self._focus_key = None
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
        if event.tab is not None and event.tab.id in _METRIC_KEYS:
            self.metric = event.tab.id
            self._redraw()


def run_app() -> None:
    # honor --all (prod_dash.main set PD_SHOW_ALL / SHOW_ALL already)
    if os.environ.get("PD_SHOW_ALL") == "1":
        pd.SHOW_ALL = True
    ProdDashApp().run()


if __name__ == "__main__":
    run_app()
