from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk

from datamining_app.algorithms.pseudocode import PSEUDOCODE
from datamining_app.console.tags import CONSOLE_BG, CONSOLE_FG, ensure_console_font
from datamining_app.i18n import i18n
from datamining_app.ui.windowing import enable_maximize

KEYWORD_COLOR = "#38BDF8"
STEP_COLOR = "#FBBF24"
FORMULA_COLOR = "#FDBA74"

_DECLARATION = re.compile(r"^\s*(ALGORITHM|INPUT|OUTPUT)\b")
_STEP = re.compile(r"Bước\s+\d+[a-zA-Z]?(?:\s*\[[^\]]*\])?")
_FORMULA = re.compile(
    r"log₂|Σ|∑|√|(?:Entropy|Gain|Gini(?:_[A-Za-z]+)?|support|conf|ΔGini|Info_[A-Za-z]+)\s*(?:\([^)\n]*\))?"
)
_FORMULA_HEADING = re.compile(r"^\s*(Công thức|Hàm khoảng cách)\s*:")


def is_declaration(line: str) -> bool:
    return _DECLARATION.search(line) is not None


def pseudocode_spans(line: str) -> list[tuple[int, int, str]]:
    """Non-overlapping (start, end, tag) ranges for one pseudocode line."""
    found: list[tuple[int, int, str]] = []
    for match in _STEP.finditer(line):
        found.append((match.start(), match.end(), "step"))
    for match in _FORMULA.finditer(line):
        found.append((match.start(), match.end(), "formula"))
    found.sort(key=lambda item: (item[0], item[0] - item[1]))
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for start, end, tag in found:
        if start < cursor or end <= start:
            continue
        spans.append((start, end, tag))
        cursor = end
    return spans


class PseudocodeDialog(tk.Toplevel):
    def __init__(self, master, algorithm_key: str, title: str | None = None) -> None:
        super().__init__(master)
        self.title(title or i18n.t("pseudocode"))
        self.geometry("760x580")
        self.minsize(480, 360)
        self.transient(master)
        self.configure(bg=CONSOLE_BG)
        enable_maximize(self)
        family = ensure_console_font(self)
        mono = (family, 13)
        mono_bold = (family, 13, "bold")

        source = PSEUDOCODE.get(algorithm_key, "Chưa có mã giả.")
        host = tk.Frame(self, bg=CONSOLE_BG)
        host.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        host.grid_rowconfigure(0, weight=1)
        host.grid_columnconfigure(0, weight=1)

        box = tk.Text(
            host,
            wrap="none",
            font=mono,
            bg=CONSOLE_BG,
            fg=CONSOLE_FG,
            insertbackground=CONSOLE_FG,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
            state="normal",
        )
        yscroll = ttk.Scrollbar(host, orient="vertical", command=box.yview)
        xscroll = ttk.Scrollbar(host, orient="horizontal", command=box.xview)
        box.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        box.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        box.tag_configure("code", foreground=CONSOLE_FG)
        box.tag_configure("keyword", foreground=KEYWORD_COLOR, font=mono_bold)
        box.tag_configure("step", foreground=STEP_COLOR, font=mono_bold)
        box.tag_configure("formula", foreground=FORMULA_COLOR)
        fill_pseudocode(box, source)
        box.configure(state="disabled")

        footer = tk.Frame(self, bg=CONSOLE_BG)
        footer.pack(fill="x", padx=8, pady=(0, 10))
        tk.Button(
            footer,
            text=i18n.t("close"),
            command=self.destroy,
            bg="#21262D",
            fg=CONSOLE_FG,
            activebackground="#30363D",
            activeforeground="#FFFFFF",
            relief="flat",
            font=("Segoe UI", 10),
            padx=14,
            pady=4,
        ).pack(side="right")


def fill_pseudocode(box: tk.Text, source: str) -> None:
    in_formula = False
    for line in source.splitlines(True):
        if _FORMULA_HEADING.search(line):
            in_formula = True
            box.insert("end", line, "formula")
            continue
        if in_formula and not is_declaration(line) and not line.lstrip().startswith("Bước"):
            box.insert("end", line, "formula")
            continue
        in_formula = False
        if is_declaration(line):
            box.insert("end", line, "keyword")
            continue
        _insert_spans(box, line)


def _insert_spans(box: tk.Text, line: str) -> None:
    spans = pseudocode_spans(line)
    if not spans:
        box.insert("end", line, "code")
        return
    cursor = 0
    for start, end, tag in spans:
        if start > cursor:
            box.insert("end", line[cursor:start], "code")
        box.insert("end", line[start:end], tag)
        cursor = end
    if cursor < len(line):
        box.insert("end", line[cursor:], "code")
