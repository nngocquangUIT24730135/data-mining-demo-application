from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from datamining_app.core.models import AlgorithmResult
from datamining_app.i18n import i18n
from datamining_app.ui.windowing import enable_maximize, maximize
from datamining_app.visualizers.cluster_visualizer import ClusterVisualizer
from datamining_app.visualizers.tree_visualizer import TreeVisualizer


class VizPopup(tk.Toplevel):
    def __init__(self, master, result: AlgorithmResult, kind: str, highlight=None) -> None:
        super().__init__(master)
        self.result = result
        self.kind = kind
        self.highlight = highlight
        if kind == "tree":
            algo = result.algorithm_name or "Tree"
            title = f"Cây quyết định — {algo}"
        else:
            title = "Phân cụm K-Means"
        self.title(title)
        self.minsize(720, 520)
        self.transient(master)
        enable_maximize(self)
        maximize(self)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(8, 8, 8, 0))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text=title, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(top, text=i18n.t("close"), command=self.destroy).pack(side="right")

        fig_host = ttk.Frame(self)
        fig_host.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor="white")
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=fig_host)
        self.toolbar = NavigationToolbar2Tk(self.canvas, fig_host)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        if kind == "tree":
            TreeVisualizer().draw(self.result, self.ax)
        else:
            ClusterVisualizer().draw(
                self.result,
                self.ax,
                highlight=self.highlight,
                voronoi=True,
            )
        self.figure.tight_layout()
        self.canvas.draw_idle()
