from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from datamining_app.console.formatter import ConsoleFormatter
from datamining_app.console.tags import (
    CONSOLE_BG,
    CONSOLE_FG,
    DEFAULT_FONT_SIZE,
    TAGS,
    adjust_font_size,
    apply_font_to_text,
    console_font,
    ensure_console_font,
    get_font_size,
    register_console_text,
    reset_font_size,
    subscribe_font,
)
from datamining_app.core.models import AlgorithmResult, StepLog
from datamining_app.i18n import i18n
from datamining_app.ui.theme import FONT_SECTION

_FORMULA_CHARS = set("Σ√∝∈∀∃μⱼᵢₛₐ")


class ConsoleWidget(ttk.Frame):
    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.formatter = ConsoleFormatter()
        self._copy_after: str | None = None
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 4))
        self.title_lbl = ttk.Label(header, text=i18n.t("results"), font=FONT_SECTION)
        self.title_lbl.pack(side="left")
        actions = ttk.Frame(header)
        actions.pack(side="right")
        self.copy_btn = ttk.Button(
            actions, text=i18n.t("console_copy"), width=12, style="Tool.TButton", command=self.copy_all
        )
        self.copy_btn.pack(side="left", padx=(0, 4))
        self.clear_btn = ttk.Button(
            actions, text=i18n.t("console_clear"), width=8, style="Tool.TButton", command=self.clear
        )
        self.clear_btn.pack(side="left", padx=(0, 4))
        self._wrap = True
        self.wrap_btn = ttk.Button(
            actions, text=i18n.t("console_wrap"), width=12, style="ToolOn.TButton", command=self.toggle_wrap
        )
        self.wrap_btn.pack(side="left", padx=(0, 8))
        self.zoom_out_btn = ttk.Button(
            actions, text=i18n.t("console_zoom_out"), width=4, style="Tool.TButton", command=lambda: self._zoom(-1)
        )
        self.zoom_out_btn.pack(side="left")
        self.zoom_reset_btn = ttk.Button(
            actions, text=self._zoom_label(), width=5, style="Tool.TButton", command=self._zoom_reset
        )
        self.zoom_reset_btn.pack(side="left", padx=2)
        self.zoom_in_btn = ttk.Button(
            actions, text=i18n.t("console_zoom_in"), width=4, style="Tool.TButton", command=lambda: self._zoom(+1)
        )
        self.zoom_in_btn.pack(side="left")
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.text = make_console_text(body, wrap="word")
        i18n.subscribe(self.refresh_texts)
        subscribe_font(self._on_font_size)

    def refresh_texts(self, _lang: str | None = None) -> None:
        self.title_lbl.configure(text=i18n.t("results"))
        self.copy_btn.configure(text=i18n.t("console_copy"))
        self.clear_btn.configure(text=i18n.t("console_clear"))
        self._sync_wrap_button()
        self.zoom_out_btn.configure(text=i18n.t("console_zoom_out"))
        self.zoom_in_btn.configure(text=i18n.t("console_zoom_in"))
        self._on_font_size(get_font_size())

    def copy_all(self) -> None:
        content = self.text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(content)
        self.copy_btn.configure(text=i18n.t("console_copied"))
        if self._copy_after is not None:
            self.after_cancel(self._copy_after)
        self._copy_after = self.after(1200, self._restore_copy_label)

    def _restore_copy_label(self) -> None:
        self._copy_after = None
        if self.winfo_exists():
            self.copy_btn.configure(text=i18n.t("console_copy"))

    def toggle_wrap(self) -> None:
        self._wrap = not self._wrap
        apply_console_wrap(self.text, self._wrap)
        self._sync_wrap_button()

    def _sync_wrap_button(self) -> None:
        self.wrap_btn.configure(
            text=i18n.t("console_wrap"),
            style="ToolOn.TButton" if self._wrap else "Tool.TButton",
        )

    def set_busy(self, busy: bool) -> None:
        self.text.configure(cursor="watch" if busy else "xterm")

    def _zoom(self, delta: int) -> None:
        adjust_font_size(delta)

    def _zoom_reset(self) -> None:
        reset_font_size()

    def _zoom_label(self) -> str:
        return f"{round(100 * get_font_size() / DEFAULT_FONT_SIZE)}%"

    def _on_font_size(self, _size: int) -> None:
        if self.winfo_exists():
            self.zoom_reset_btn.configure(text=self._zoom_label())

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")

    def append(self, text: str) -> None:
        self.text.configure(state="normal")
        tagger = LineTagger()
        for line in text.splitlines(True):
            self.text.insert("end", line, tagger.tag(line))
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


def apply_console_wrap(text: tk.Text, enabled: bool) -> None:
    """Word-wrap prose. Turning it off keeps box-drawing tables on one line."""
    text.configure(wrap="word" if enabled else "none")
    hscroll = getattr(text, "_hscroll", None)
    if hscroll is None:
        return
    if enabled:
        hscroll.grid_remove()
    else:
        hscroll.grid()


def make_console_text(master, *, wrap: str = "none") -> tk.Text:
    """Monospace console. ``wrap='none'`` keeps box-drawing tables aligned."""
    ensure_console_font(master)
    master.grid_rowconfigure(0, weight=1)
    master.grid_columnconfigure(0, weight=1)
    text = tk.Text(
        master,
        wrap=wrap,
        font=console_font(),
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
    text._hscroll = xscroll
    if wrap != "none":
        xscroll.grid_remove()
    for name, color in TAGS.items():
        text.tag_configure(name, foreground=color)
    apply_font_to_text(text)
    register_console_text(text)
    return text


class LineTagger:
    """Stateful line tagging so the MÃ GIẢ box stays green vs data tables."""

    def __init__(self) -> None:
        self.in_pseudocode = False
        self.expect_table_header = False

    def tag(self, line: str) -> str:
        stripped = line.lstrip()
        if " MÃ GIẢ " in stripped and stripped.startswith("┌"):
            self.in_pseudocode = True
            self.expect_table_header = False
            return "pseudocode"
        if self.in_pseudocode:
            tag = "pseudocode"
            if stripped.startswith("└"):
                self.in_pseudocode = False
            return tag

        if stripped.startswith("┌"):
            self.expect_table_header = True
            return "table_border"
        if stripped.startswith("├"):
            self.expect_table_header = False
            return "table_border"
        if stripped.startswith("└"):
            self.expect_table_header = False
            return "table_border"
        if stripped.startswith("│"):
            if self.expect_table_header:
                return "table_header"
            return "table_row"

        return tag_for_line(line)


def tag_for_line(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith(("╔", "║", "╚")):
        return "algo_header"
    if stripped.startswith("──"):
        return "step_header"
    if stripped.startswith(("①", "②", "③", "④", "⑤")):
        return "concept"
    if stripped.startswith("[Mã giả"):
        return "pseudocode"
    if "Cảnh báo" in line or stripped.startswith("WARNING"):
        return "warning"
    if "✓" in line:
        return "accept"
    if "✗" in line:
        return "reject"
    if stripped.startswith("⟹"):
        return "result"
    if stripped.startswith(("→", "⇒")):
        return "arrow"
    if "=" in line and any(ch in line for ch in _FORMULA_CHARS):
        return "formula"
    if stripped[:1] in "┌┐└┘├┤┬┴┼─":
        return "table_border"
    if stripped.startswith("│"):
        if "Kết quả:" in line:
            return "result"
        return "table_row"
    return "info"


def _tag_for(line: str) -> str:
    return tag_for_line(line)
