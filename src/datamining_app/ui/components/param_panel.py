from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from datamining_app.core.models import ParamDef
from datamining_app.i18n import i18n

SKIP_KEYS = {"exclude_cols"}


class ParamPanel(ttk.LabelFrame):
    def __init__(self, master, schema: dict[str, ParamDef], defaults: dict[str, Any] | None = None, **kwargs) -> None:
        super().__init__(master, text=i18n.t("params"), **kwargs)
        self.schema = schema
        self.defaults = defaults or {}
        self.widgets: dict[str, Any] = {}
        self.headers: list[str] = []
        self.host = ttk.Frame(self)
        self.host.pack(fill="x", padx=8, pady=6)
        i18n.subscribe(self.refresh_texts)
        self._build()

    def apply_dataset(self, headers: list[str], extra_defaults: dict[str, Any] | None = None) -> None:
        self.headers = headers
        if extra_defaults:
            self.defaults = {**self.defaults, **extra_defaults}
        self.widgets = {}
        self._build()

    def set_headers(self, headers: list[str]) -> None:
        self.headers = headers
        self._build()

    def refresh_texts(self, _lang: str | None = None) -> None:
        self.configure(text=i18n.t("params"))
        self._build()

    def _build(self) -> None:
        values = self.get_values() if self.widgets else dict(self.defaults)
        for child in self.host.winfo_children():
            child.destroy()
        self.widgets = {}
        for key, spec in self.schema.items():
            if key in SKIP_KEYS:
                continue
            label = spec.label_vi if i18n.language == "vi" else spec.label_en
            row = ttk.Frame(self.host)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label or key, wraplength=260).pack(anchor="w")
            current = values.get(key, spec.default)
            if spec.type in {"float", "int"}:
                self._build_number(row, key, spec, current)
            elif spec.type == "radio":
                self._build_radio(row, key, spec, current)
            elif spec.type == "str":
                var = tk.StringVar(value=str(current if current is not None else ""))
                ttk.Entry(row, textvariable=var).pack(fill="x")
                self.widgets[key] = ("choice", var)
            elif spec.type == "choice":
                options = self._spec_options(spec)
                if spec.options == "headers" and "decision" in key:
                    options = [""] + options
                if current in options:
                    value = str(current)
                elif current in (None, "") and "" in options and "decision_attr" in self.defaults and not self.defaults.get("decision_attr"):
                    value = ""
                else:
                    fallback = options[-1] if "decision" in key and options else (options[0] if options else "")
                    value = str(fallback)
                var = tk.StringVar(value=value)
                ttk.Combobox(row, textvariable=var, values=options, state="readonly").pack(fill="x")
                self.widgets[key] = ("choice", var)
            elif spec.type == "multiselect":
                options = self._spec_options(spec)
                selected = set(current or [])
                box = ttk.Frame(row)
                box.pack(fill="x")
                vars_map = {}
                for opt in options:
                    var = tk.BooleanVar(value=opt in selected)
                    ttk.Checkbutton(box, text=opt, variable=var).pack(side="left", padx=2)
                    vars_map[opt] = var
                self.widgets[key] = ("multiselect", vars_map)

    def _build_number(self, row, key: str, spec: ParamDef, current: Any) -> None:
        is_int = spec.type == "int"
        lo = spec.min if spec.min is not None else 0
        hi = spec.max if spec.max is not None else 100
        step = spec.step if spec.step is not None else (1 if is_int else 0.05)
        try:
            start = current if current is not None else spec.default
            start = int(start) if is_int else float(start)
        except (TypeError, ValueError):
            start = int(lo) if is_int else float(lo)
        var = tk.StringVar(value=str(start))
        spin = ttk.Spinbox(
            row,
            from_=lo,
            to=hi,
            increment=step,
            textvariable=var,
            width=12,
        )
        spin.pack(anchor="w")
        self.widgets[key] = ("num", var, is_int)

    def _build_radio(self, row, key: str, spec: ParamDef, current: Any) -> None:
        options = self._spec_options(spec)
        labels = {
            "euclidean": "Euclidean",
            "manhattan": "Manhattan",
            "random": "Random",
        }
        var = tk.StringVar(value=str(current if current in options else (options[0] if options else "")))
        box = ttk.Frame(row)
        box.pack(anchor="w")
        for opt in options:
            ttk.Radiobutton(box, text=labels.get(opt, opt), value=opt, variable=var).pack(side="left", padx=(0, 10))
        self.widgets[key] = ("choice", var)

    def _spec_options(self, spec: ParamDef) -> list[str]:
        if spec.options == "headers":
            return list(self.headers)
        if isinstance(spec.options, list):
            return [str(x) for x in spec.options]
        return []

    def get_values(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            k: v for k, v in self.defaults.items() if k not in SKIP_KEYS
        }
        for key, widget in self.widgets.items():
            kind = widget[0]
            if kind == "num":
                _, var, is_int = widget
                raw = var.get().strip().replace(",", ".")
                try:
                    result[key] = int(float(raw)) if is_int else float(raw)
                except ValueError:
                    result[key] = self.schema[key].default
            elif kind == "choice":
                result[key] = widget[1].get()
            elif kind == "multiselect":
                result[key] = [name for name, var in widget[1].items() if var.get()]
        return result
