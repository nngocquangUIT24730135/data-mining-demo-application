from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from datamining_app.console.formatter import ConsoleFormatter
from datamining_app.core.models import AlgorithmResult
from datamining_app.i18n import i18n
from datamining_app.ui.components.console_widget import make_console_text, tag_for_line
from datamining_app.ui.windowing import maximize
from datamining_app.visualizers.cluster_visualizer import ClusterVisualizer
from datamining_app.visualizers.tree_visualizer import TreeVisualizer


class VizPopup(tk.Toplevel):
    def __init__(self, master, result: AlgorithmResult, kind: str, highlight=None) -> None:
        super().__init__(master)
        self.result = result
        self.kind = kind
        self.highlight = highlight
        title = "Cây quyết định — ID3" if kind == "tree" else "Phân cụm K-Means"
        self.title(title)
        self.minsize(960, 640)
        self.transient(master)
        maximize(self)
        self._iter_index = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(8, 8, 8, 0))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text=title, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(top, text=i18n.t("close"), command=self.destroy).pack(side="right")

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        console_host = ttk.Frame(pane)
        fig_host = ttk.Frame(pane)
        pane.add(console_host, weight=1)
        pane.add(fig_host, weight=2)

        ttk.Label(console_host, text="Console", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
        text_host = ttk.Frame(console_host)
        text_host.pack(fill="both", expand=True)
        self.console = make_console_text(text_host)
        self.console.configure(takefocus=0)
        text = ConsoleFormatter().render_result(result)
        for line in text.splitlines(True):
            self.console.insert("end", line, tag_for_line(line))

        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor="white")
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=fig_host)
        self.toolbar = NavigationToolbar2Tk(self.canvas, fig_host)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        history = (result.output or {}).get("history") or []
        if kind == "cluster" and history:
            bar = ttk.Frame(fig_host)
            bar.pack(fill="x", padx=4, pady=6)
            ttk.Label(bar, text=i18n.t("iteration")).pack(side="left")
            values = [str(h.get("iteration", i + 1)) for i, h in enumerate(history)]
            self.iter_var = tk.StringVar(value=values[-1])
            combo = ttk.Combobox(bar, textvariable=self.iter_var, values=values, state="readonly", width=8)
            combo.pack(side="left", padx=8)
            combo.bind("<<ComboboxSelected>>", lambda _e: self._on_combo())
            self._iter_index = len(history) - 1
        self._draw()

    def _on_combo(self) -> None:
        history = (self.result.output or {}).get("history") or []
        try:
            chosen = int(self.iter_var.get())
        except (TypeError, ValueError, AttributeError):
            chosen = len(history)
        for i, snap in enumerate(history):
            if int(snap.get("iteration", i + 1)) == chosen:
                self._iter_index = i
                break
        self._draw()

    def _draw(self) -> None:
        if self.kind == "tree":
            TreeVisualizer().draw(self.result, self.ax)
        else:
            ClusterVisualizer().draw(
                self.result,
                self.ax,
                highlight=self.highlight,
                iteration=self._iter_index,
                voronoi=True,
            )
        self.figure.tight_layout()
        self.canvas.draw_idle()
