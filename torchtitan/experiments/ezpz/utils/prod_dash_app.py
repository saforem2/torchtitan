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

Keys: q quit | r refresh | x toggle step<->tokens x-axis | a toggle show-all
experiments | left/right (or the tab bar) switch metric.
"""
from __future__ import annotations

import os
import zlib

import prod_dash as pd  # data layer (same utils/ dir -> on sys.path[0])

from textual.app import App, ComposeResult
from textual.widgets import (
    Footer, Header, RichLog, Static, Tabs, Tab, TabbedContent, TabPane,
)
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


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


class ProdDashApp(App):
    """Live multi-metric production dashboard."""

    TITLE = "AuroraGPT production"
    # Top-level tabs ("Charts" / "Board") so the table gets its own pane and no
    # longer competes with the chart for vertical space -- the chart fills its
    # whole pane. #metrictabs is the per-metric selector inside the Charts pane.
    CSS = """
    #chart { height: 1fr; }
    #board { height: 1fr; overflow-y: auto; color: $text-muted; padding: 0 1; }
    #log { height: 6; display: none; dock: bottom; }
    #log.building { display: block; }
    """
    BINDINGS = [
        ("q", "quit", "quit"),
        ("r", "refresh", "refresh"),
        ("x", "toggle_xaxis", "step/tokens"),
        ("a", "toggle_all", "show-all"),
        ("b", "show_board", "board"),
        ("c", "show_charts", "charts"),
    ]

    def __init__(self):
        super().__init__()
        self.metric = "loss"
        self.xaxis = "tokens" if os.environ.get("PD_XAXIS") == "tokens" else "step"
        self.payload = {"chains": {}}
        self._fresh_kicked = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="panes"):
            with TabPane("Charts", id="pane-charts"):
                yield Tabs(*[Tab(label, id=key) for key, label in METRICS],
                           id="metrictabs")
                yield PlotextPlot(id="chart")
            with TabPane("Board", id="pane-board"):
                yield Static("(loading ...)", id="board")
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

    def _apply_payload(self, data: dict) -> None:
        self.payload = data or {"chains": {}}
        self._log_show(False)
        self.query_one("#log", RichLog).clear()
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
    def _redraw(self) -> None:
        widget = self.query_one("#chart", PlotextPlot)
        plt = widget.plt
        plt.clear_data()
        plt.clear_figure()
        chains = self.payload.get("chains", {})

        def order(item):
            k, c = item
            return (c.get("kind") != "canonical", c.get("model") or "z",
                    -(c.get("num_nodes") or 0), k)

        drawn = 0
        # draw idle first, live last (so live sits on top)
        items = sorted(chains.items(), key=order)
        live_last = sorted(items, key=lambda kc: bool(
            kc[1].get("queue_state") == "R"
            or (kc[1].get("log_age") is not None
                and kc[1]["log_age"] <= pd.LIVE_WINDOW)))
        for key, c in live_last:
            series = (c.get("series") or {}).get(self.metric) or []
            tip = c.get("live_tip") or {}
            # append the freshest running-job point for this metric if newer
            if tip and self.metric in _METRIC_KEYS:
                tv = tip.get(self.metric if self.metric != "loss" else "loss")
                ts = tip.get("step")
                if tv is not None and ts is not None and (
                        not series or ts > series[-1][0]):
                    series = series + [[ts, tv]]
            if len(series) < 2:
                continue
            toks_per_step = (c.get("gbs") or 0) * (c.get("seq_len") or 0)
            if self.xaxis == "tokens":
                if not toks_per_step:
                    continue  # can't place a no-gbs experiment on a tokens axis
                xs = [p[0] * toks_per_step / 1e9 for p in series]
            else:
                xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            is_live = (c.get("queue_state") == "R"
                       or (c.get("log_age") is not None
                           and c["log_age"] <= pd.LIVE_WINDOW))
            color = _hex_to_rgb(pd.COLORS[zlib.crc32(key.encode()) % len(pd.COLORS)])
            label = ("* " if is_live else "") + c.get("label", key) + (
                "" if is_live else " (idle)")
            marker = "braille" if is_live else "dot"
            plt.plot(xs, ys, color=color, label=label, marker=marker)
            drawn += 1

        axis_label = dict(METRICS)[self.metric]
        plt.title("AuroraGPT production -- %s%s" % (
            axis_label, "" if drawn else "  (no data)"))
        plt.xlabel("tokens seen (billions)" if self.xaxis == "tokens"
                   else "training step (cumulative)")
        plt.ylabel(axis_label)
        widget.refresh()

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
