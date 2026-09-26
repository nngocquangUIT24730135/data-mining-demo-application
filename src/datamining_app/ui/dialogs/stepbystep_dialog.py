from __future__ import annotations

from typing import Any

import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from datamining_app.console.formatter import ConsoleFormatter
from datamining_app.core.models import AlgorithmResult, StepLog
from datamining_app.i18n import i18n
from datamining_app.ui.components.console_widget import LineTagger, make_console_text
from datamining_app.ui.step_slides import console_slides
from datamining_app.ui.windowing import enable_maximize, maximize
from datamining_app.visualizers.cluster_visualizer import ClusterVisualizer
from datamining_app.visualizers.tree_visualizer import TreeVisualizer


class StepByStepDialog(tk.Toplevel):
    def __init__(self, master, result: AlgorithmResult, kind: str | None = None) -> None:
        super().__init__(master)
        self.result = result
        self.kind = kind or _kind_for(result)
        self.slides = console_slides(result)
        self.index = 0
        self.formatter = ConsoleFormatter()
        self.title(i18n.t("demo_steps"))
        self.minsize(960, 640)
        self.transient(master)
        enable_maximize(self)

        chrome = ttk.Frame(self, padding=(12, 10, 12, 8))
        chrome.pack(side="top", fill="x")
        self.header = ttk.Label(chrome, font=("Segoe UI", 12, "bold"))
        self.header.pack(side="top", anchor="w")
        self.progress = ttk.Label(chrome)
        self.progress.pack(side="top", anchor="w", pady=(2, 8))

        bar = ttk.Frame(chrome)
        bar.pack(side="top", fill="x")
        self.prev_btn = ttk.Button(bar, text=i18n.t("prev"), command=self._prev)
        self.prev_btn.pack(side="left", padx=4)
        self.next_btn = ttk.Button(bar, text=i18n.t("next"), command=self._next)
        self.next_btn.pack(side="left", padx=4)
        ttk.Button(bar, text=i18n.t("close"), command=self.destroy).pack(side="right", padx=4)

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        console_host = ttk.Frame(pane)
        fig_host = ttk.Frame(pane)
        pane.add(console_host, weight=1)
        pane.add(fig_host, weight=2)

        ttk.Label(console_host, text="Console — bước hiện tại", font=("Segoe UI", 9, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        text_host = ttk.Frame(console_host)
        text_host.pack(fill="both", expand=True)
        self.console = make_console_text(text_host)
        self.console.configure(takefocus=0)

        self.figure = Figure(figsize=(5.0, 4.0), dpi=100, facecolor="white")
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=fig_host)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(width=100, height=100)
        canvas_widget.pack(fill="both", expand=True)

        self.bind("<Left>", lambda _e: self._prev())
        self.bind("<Right>", lambda _e: self._next())
        self.bind("<Return>", lambda _e: self._next())
        self.bind("<space>", lambda _e: self._next())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        maximize(self)
        self.after(50, self._focus_nav)
        self._draw()

    def _focus_nav(self) -> None:
        try:
            self.next_btn.focus_set()
        except tk.TclError:
            pass
        self.lift()
        self.focus_force()

    def _prev(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._draw()

    def _next(self) -> None:
        if self.index < len(self.slides) - 1:
            self.index += 1
            self._draw()

    def _draw(self) -> None:
        if not self.slides:
            self.header.configure(text=i18n.t("no_result"))
            return
        step = self.slides[self.index]
        if (step.data or {}).get("omit_step_number"):
            self.header.configure(text=step.title)
        else:
            self.header.configure(text=f"Bước {step.step_number}: {step.title}")
        self.progress.configure(text=i18n.t("step_progress", n=self.index + 1, total=len(self.slides)))
        self.ax.clear()
        draw_step(self.result, step, self.ax)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._fill_console(step)
        self.prev_btn.configure(state="normal" if self.index else "disabled")
        self.next_btn.configure(state="normal" if self.index < len(self.slides) - 1 else "disabled")

    def _fill_console(self, step: StepLog) -> None:
        self.console.delete("1.0", "end")
        text = self.formatter.render_step(step)
        tagger = LineTagger()
        for line in text.splitlines(True):
            self.console.insert("end", line, tagger.tag(line))
        self.console.see("1.0")


def _kind_for(result: AlgorithmResult) -> str:
    name = (result.algorithm_name or "").lower()
    if "id3" in name or "cart" in name or "gini" in name:
        return "tree"
    if "k-means" in name or "kmeans" in name:
        return "cluster"
    return "console"


def draw_step(result: AlgorithmResult, step: StepLog, ax) -> None:
    name = result.algorithm_name or ""
    data = step.data or {}
    title = step.title or ""
    if name == "K-Means":
        _draw_kmeans(result, step, ax)
        return
    if name == "ID3" or "CART" in name or "Gini" in name:
        _draw_id3(result, step, ax)
        return
    if _draw_bars_from_step(data, ax, title):
        return
    if data.get("cells"):
        _draw_matrix_cells(data, ax)
        return
    if data.get("headers") and data.get("rows") is not None:
        _draw_mpl_table(data["headers"], data["rows"], ax, title)
        return
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=12, wrap=True)
    ax.text(0.5, 0.35, "Chi tiết ở khung console bên dưới", ha="center", va="center", fontsize=9, color="#666666")


def _draw_kmeans(result: AlgorithmResult, step: StepLog, ax) -> None:
    viz = ClusterVisualizer()
    data = step.data or {}
    title = step.title or ""
    phase = data.get("phase")
    iteration = data.get("iteration")
    hist_index = (int(iteration) - 1) if iteration else 0
    if phase == "init" or "Khởi tạo" in title:
        viz.draw(result, ax, iteration=0, show_init=True, raw=True)
    elif phase == "assign" or "Gán" in title:
        viz.draw(result, ax, iteration=hist_index, phase="assign")
    elif phase == "update" or "Cập nhật" in title:
        viz.draw(result, ax, iteration=hist_index, phase="update")
    elif phase == "converged" or "Hội tụ" in title:
        viz.draw(result, ax, voronoi=True)
    else:
        viz.draw(result, ax, iteration=hist_index if iteration else None, phase=phase)


def _draw_id3(result: AlgorithmResult, step: StepLog, ax) -> None:
    data = step.data or {}
    snapshot = data.get("tree")
    TreeVisualizer().draw(result, ax, tree=snapshot, highlight=data.get("path"))
    ax.set_title(step.title or "Cây quyết định ID3")


def _draw_bars_from_step(data: dict[str, Any], ax, title: str) -> bool:
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) and "itemset" in candidates[0]:
        labels = ["{" + ", ".join(map(str, row.get("itemset") or [])) + "}" for row in candidates]
        vals = [float(row.get("support") or 0) for row in candidates]
        colors = ["#5CB85C" if row.get("accepted") else "#D9534F" for row in candidates]
        ax.barh(labels, vals, color=colors)
        ax.set_xlabel("Support")
        ax.set_title(title)
        return True
    rules = data.get("rules")
    if isinstance(rules, list) and rules and isinstance(rules[0], dict) and "antecedent" in rules[0]:
        labels = []
        vals = []
        for rule in rules:
            lhs = "{" + ", ".join(map(str, rule.get("antecedent") or [])) + "}"
            rhs = "{" + ", ".join(map(str, rule.get("consequent") or [])) + "}"
            labels.append(f"{lhs} → {rhs}")
            vals.append(float(rule.get("confidence") or 0))
        ax.barh(labels, vals, color="#4A90D9")
        ax.set_xlabel("Confidence")
        ax.set_title(title)
        return True
    priors = data.get("priors")
    if isinstance(priors, dict) and priors:
        ax.bar(list(priors.keys()), list(priors.values()), color="#5BC0DE")
        ax.set_ylabel("P(C)")
        ax.set_title(title)
        return True
    return False


def _draw_matrix_cells(data: dict[str, Any], ax) -> None:
    cells = data.get("cells") or []
    headers = ["u", "v", "M(u,v)"]
    rows = [[cell.get("i"), cell.get("j"), "{" + ", ".join(cell.get("attrs") or []) + "}"] for cell in cells]
    _draw_mpl_table(headers, rows, ax, "Ma trận phân biệt")


def _draw_mpl_table(headers, rows, ax, title: str) -> None:
    ax.axis("off")
    ax.set_title(title)
    if not headers:
        return
    display_rows = [[str(c) for c in row] for row in (rows or [])]
    if not display_rows:
        ax.text(0.5, 0.5, "(rỗng)", ha="center", va="center")
        return
    table = ax.table(cellText=display_rows, colLabels=[str(h) for h in headers], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
