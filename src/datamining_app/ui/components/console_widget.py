from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from datamining_app.console.formatter import ConsoleFormatter
from datamining_app.console.tags import CONSOLE_BG, CONSOLE_FG, FONT_CONSOLE, TAGS
from datamining_app.core.models import AlgorithmResult, StepLog
from datamining_app.i18n import i18n


class ConsoleWidget(ttk.Frame):
    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.formatter = ConsoleFormatter()
        self.title_lbl = ttk.Label(self, text=i18n.t("results"), font=("Segoe UI", 10, "bold"))
        self.title_lbl.pack(anchor="w", pady=(0, 4))
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.text = make_console_text(body)
        i18n.subscribe(self.refresh_texts)

    def refresh_texts(self, _lang: str | None = None) -> None:
        self.title_lbl.configure(text=i18n.t("results"))

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")

    def append(self, text: str) -> None:
        self.text.configure(state="normal")
        for line in text.splitlines(True):
            self.text.insert("end", line, tag_for_line(line))
        self.text.see("end")

    def log_step(self, step: StepLog) -> None:
        self.append(self.formatter.render_step(step) + "\n")

    def show_result(self, result: AlgorithmResult, dataset_name: str = "", n_rows: int = 0) -> None:
        self.clear()
        self.append(self.formatter.render_result(result, dataset_name, n_rows))

    def log_prediction(self, label: str, explanation: str) -> None:
        self.append("\n" + self.formatter.render_prediction(label, explanation))

    def log_info(self, message: str, tag: str = "info") -> None:
        self.append(message + "\n")


def make_console_text(master) -> tk.Text:
    """Monospace console: no wrap, so box-drawing tables stay aligned."""
    master.grid_rowconfigure(0, weight=1)
    master.grid_columnconfigure(0, weight=1)
    text = tk.Text(
        master,
        wrap="none",
        font=FONT_CONSOLE,
        bg=CONSOLE_BG,
        fg=CONSOLE_FG,
        insertbackground=CONSOLE_FG,
        relief="sunken",
        borderwidth=1,
        state="normal",
        padx=6,
        pady=6,
    )
    yscroll = ttk.Scrollbar(master, orient="vertical", command=text.yview)
    xscroll = ttk.Scrollbar(master, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    text.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    for name, color in TAGS.items():
        text.tag_configure(name, foreground=color)
    text.tag_configure("step_header", foreground=TAGS["step_header"], font=("Consolas", 10, "bold"))
    text.tag_configure("algo_header", foreground=TAGS["algo_header"], font=("Consolas", 10, "bold"))
    text.tag_configure("result", foreground=TAGS["result"], font=("Consolas", 10, "bold"))
    return text


def tag_for_line(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith(("╔", "║", "╚")):
        return "algo_header"
    if stripped.startswith("──"):
        return "step_header"
    if "✓" in line:
        return "accept"
    if "✗" in line:
        return "reject"
    if stripped.startswith(("→", "⇒")):
        return "arrow"
    if stripped[:1] in "┌┐└┘├┤┬┴┼│─":
        if "Kết quả:" in line:
            return "result"
        return "table_border"
    return "info"


def _tag_for(line: str) -> str:
    return tag_for_line(line)
