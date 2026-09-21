from __future__ import annotations

from tkinter import filedialog, ttk
import tkinter as tk
from typing import Any, Callable

from datamining_app.core.models import Dataset, DatasetMeta
from datamining_app.data.loader import list_datasets, load_dataset
from datamining_app.data.preprocessor import Preprocessor, identifier_headers, validate_exclude
from datamining_app.i18n import i18n
from datamining_app.ui.dialogs.exclude_columns_dialog import ExcludeColumnsDialog


class DataPanel(ttk.LabelFrame):
    def __init__(
        self,
        master,
        on_dataset: Callable[[Dataset], None],
        default_table: str = "",
        algorithm_key: str = "",
        get_params: Callable[[], dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, text=i18n.t("data_panel"), **kwargs)
        self.on_dataset = on_dataset
        self.raw: Dataset | None = None
        self.processed: Dataset | None = None
        self.preprocessor = Preprocessor()
        self.exclude_cols: list[str] = []
        self.algorithm_key = algorithm_key
        self.get_params = get_params or (lambda: {})
        self._table_name = default_table
        self._datasets: list[DatasetMeta] = list_datasets(algorithm_key) if algorithm_key else []
        self._label_to_table = {d.display_name: d.table_name for d in self._datasets}
        self._table_to_label = {d.table_name: d.display_name for d in self._datasets}
        self._desc: dict[str, str] = {d.table_name: d.description for d in self._datasets}

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=6)
        self.csv_btn = ttk.Button(btns, text=i18n.t("load_csv"), command=self.load_csv)
        self.csv_btn.pack(side="left")
        self.sqlite_lbl = ttk.Label(btns, text=i18n.t("sqlite_table"))
        self.sqlite_lbl.pack(side="left", padx=(8, 4))
        labels = [d.display_name for d in self._datasets]
        default_label = self._table_to_label.get(default_table, labels[0] if labels else "")
        self.table_var = tk.StringVar(value=default_label)
        self.table_combo = ttk.Combobox(btns, textvariable=self.table_var, values=labels, state="readonly", width=28)
        self.table_combo.pack(side="left", fill="x", expand=True)
        self.table_combo.bind("<<ComboboxSelected>>", lambda _e: self.load_sqlite())

        self.desc_lbl = ttk.Label(self, text="", wraplength=280, foreground="#555555")
        self.desc_lbl.pack(anchor="w", padx=8)

        self.status = ttk.Label(self, text=i18n.t("no_data"), wraplength=280)
        self.status.pack(anchor="w", padx=8)

        exclude_row = ttk.Frame(self)
        exclude_row.pack(fill="x", padx=8, pady=(8, 0))
        self.exclude_btn = ttk.Button(exclude_row, text=i18n.t("exclude_columns"), command=self.open_exclude)
        self.exclude_btn.pack(side="left")
        self.exclude_summary = ttk.Label(exclude_row, text=i18n.t("exclude_none"), wraplength=180)
        self.exclude_summary.pack(side="left", padx=8)

        self.preview_lbl = ttk.Label(self, text=i18n.t("preview"))
        self.preview_lbl.pack(anchor="w", padx=8, pady=(8, 0))
        self.tree = ttk.Treeview(self, show="headings", height=7)
        self.tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        i18n.subscribe(self.refresh_texts)
        if default_table or labels:
            self.after(50, self.load_sqlite)

    def refresh_texts(self, _lang: str | None = None) -> None:
        self.configure(text=i18n.t("data_panel"))
        self.csv_btn.configure(text=i18n.t("load_csv"))
        self.sqlite_lbl.configure(text=i18n.t("sqlite_table"))
        self.exclude_btn.configure(text=i18n.t("exclude_columns"))
        self.preview_lbl.configure(text=i18n.t("preview"))
        self._status_text()
        self._exclude_summary_text()

    def current_meta(self) -> DatasetMeta | None:
        table = self._table_name
        for item in self._datasets:
            if item.table_name == table:
                return item
        return None

    def load_csv(self) -> None:
        path = filedialog.askopenfilename(title=i18n.t("choose_csv"), filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        self.raw = load_dataset(path)
        self._table_name = self.raw.name
        self.exclude_cols = identifier_headers(self.raw.headers)
        self.desc_lbl.configure(text="")
        self._apply()

    def load_sqlite(self) -> None:
        label = self.table_var.get()
        table = self._label_to_table.get(label, self._table_name)
        if not table:
            return
        self.raw = load_dataset("sqlite", table)
        self._table_name = table
        self.exclude_cols = identifier_headers(self.raw.headers)
        self.desc_lbl.configure(text=self._desc.get(table, ""))
        self._apply()

    def open_exclude(self) -> None:
        if not self.raw:
            return
        extra = dict(self.get_params())
        ExcludeColumnsDialog(
            self.winfo_toplevel(),
            headers=self.raw.headers,
            selected=self.exclude_cols,
            on_apply=self._set_exclude,
            algorithm=self.algorithm_key,
            rows=self.raw.rows,
            extra=extra,
        )

    def _set_exclude(self, cols: list[str]) -> None:
        self.exclude_cols = cols
        self._apply()

    def _apply(self) -> None:
        if not self.raw:
            return
        extra = dict(self.get_params())
        try:
            validate_exclude(self.raw.headers, self.raw.rows, self.exclude_cols, self.algorithm_key, extra)
        except Exception:
            self.exclude_cols = identifier_headers(self.raw.headers)
        self.processed = self.preprocessor.transform(
            self.raw, exclude_cols=self.exclude_cols, strip_whitespace=True
        )
        self._render_preview()
        self._status_text()
        self._exclude_summary_text()
        self.on_dataset(self.processed)

    def _exclude_summary_text(self) -> None:
        if self.exclude_cols:
            self.exclude_summary.configure(text=i18n.t("exclude_summary", cols=", ".join(self.exclude_cols)))
        else:
            self.exclude_summary.configure(text=i18n.t("exclude_none"))

    def _status_text(self) -> None:
        if self.processed:
            self.status.configure(
                text=i18n.t(
                    "dataset_loaded",
                    name=self.processed.name,
                    n=len(self.processed.rows),
                    m=len(self.processed.headers),
                )
            )
        else:
            self.status.configure(text=i18n.t("no_data"))

    def _render_preview(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if not self.processed:
            return
        headers = self.processed.headers
        self.tree.configure(columns=headers)
        for h in headers:
            self.tree.heading(h, text=h)
            self.tree.column(h, width=80, stretch=True)
        for row in self.processed.rows[:40]:
            self.tree.insert("", "end", values=[row.get(h, "") for h in headers])

    def current_dataset(self) -> Dataset | None:
        return self.processed
