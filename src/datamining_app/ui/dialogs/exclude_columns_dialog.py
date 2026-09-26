from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from datamining_app.data.preprocessor import ExcludeValidationError, validate_exclude
from datamining_app.i18n import i18n
from datamining_app.ui.windowing import enable_maximize


class ExcludeColumnsDialog(tk.Toplevel):
    def __init__(
        self,
        master,
        headers: list[str],
        selected: list[str],
        on_apply: Callable[[list[str]], None],
        algorithm: str = "",
        rows: list | None = None,
        extra: dict | None = None,
    ) -> None:
        super().__init__(master)
        self.headers = headers
        self.on_apply = on_apply
        self.algorithm = algorithm
        self.rows = rows or []
        self.extra = extra or {}
        self.title(i18n.t("exclude_columns"))
        self.transient(master)
        self.minsize(320, 360)
        self.geometry("360x420")
        enable_maximize(self)

        ttk.Label(self, text=i18n.t("exclude_hint"), wraplength=320).pack(anchor="w", padx=12, pady=(12, 6))
        host = ttk.Frame(self)
        host.pack(fill="both", expand=True, padx=12)
        self.vars: dict[str, tk.BooleanVar] = {}
        chosen = set(selected)
        for header in headers:
            var = tk.BooleanVar(value=header in chosen)
            ttk.Checkbutton(host, text=header, variable=var, command=self._on_toggle).pack(anchor="w", pady=2)
            self.vars[header] = var

        self.error_lbl = ttk.Label(self, foreground="#B00020", wraplength=320)
        self.error_lbl.pack(anchor="w", padx=12, pady=(4, 0))

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=12, pady=12)
        self.apply_btn = ttk.Button(btns, text=i18n.t("exclude_apply"), command=self._apply)
        self.apply_btn.pack(side="right")
        ttk.Button(btns, text=i18n.t("exclude_cancel"), command=self.destroy).pack(side="right", padx=(0, 8))
        self._on_toggle()

    def _selected(self) -> list[str]:
        return [h for h, var in self.vars.items() if var.get()]

    def _on_toggle(self) -> None:
        try:
            validate_exclude(self.headers, self.rows, self._selected(), self.algorithm, self.extra)
            self.error_lbl.configure(text="")
            self.apply_btn.configure(state="normal")
        except ExcludeValidationError as exc:
            self.error_lbl.configure(text=str(exc))
            self.apply_btn.configure(state="disabled")

    def _apply(self) -> None:
        selected = self._selected()
        try:
            validate_exclude(self.headers, self.rows, selected, self.algorithm, self.extra)
        except ExcludeValidationError as exc:
            messagebox.showerror(i18n.t("exclude_columns"), str(exc), parent=self)
            return
        self.on_apply(selected)
        self.destroy()
