from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from datamining_app.algorithms.dataset_utils import to_float
from datamining_app.core.models import Dataset
from datamining_app.i18n import i18n


class PredictPanel(ttk.LabelFrame):
    def __init__(self, master, on_predict: Callable[[dict[str, Any]], None], **kwargs) -> None:
        super().__init__(master, text=i18n.t("predict_title"), **kwargs)
        self.on_predict = on_predict
        self.vars: dict[str, tk.StringVar] = {}
        self.host = ttk.Frame(self)
        self.host.pack(fill="x", padx=8, pady=4)
        self.predict_btn = ttk.Button(self, text=i18n.t("predict"), command=self._submit)
        self.predict_btn.pack(anchor="e", padx=8, pady=(0, 8))
        i18n.subscribe(self.refresh_texts)

    def refresh_texts(self, _lang: str | None = None) -> None:
        self.configure(text=i18n.t("predict_title"))
        self.predict_btn.configure(text=i18n.t("predict"))

    def set_fields(
        self,
        names: list[str],
        dataset: Dataset | None = None,
        *,
        force_entry: bool = False,
    ) -> None:
        for child in self.host.winfo_children():
            child.destroy()
        self.vars = {}
        for name in names:
            row = ttk.Frame(self.host)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name, width=16).pack(side="left")
            values = []
            if dataset:
                values = sorted({str(v) for v in dataset.column_values(name) if str(v) != ""})
            var = tk.StringVar(value=values[0] if values else "")
            use_entry = force_entry or _all_numeric(values) or not (0 < len(values) <= 16)
            if use_entry:
                ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            else:
                ttk.Combobox(row, textvariable=var, values=values, state="readonly").pack(
                    side="left", fill="x", expand=True
                )
            self.vars[name] = var

    def _submit(self) -> None:
        self.on_predict({name: var.get().strip() for name, var in self.vars.items()})


def _all_numeric(values: list[str]) -> bool:
    if not values:
        return False
    return all(to_float(v) is not None for v in values)
