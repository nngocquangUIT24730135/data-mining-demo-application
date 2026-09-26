from __future__ import annotations

import threading
from tkinter import messagebox, ttk
from typing import Any, Callable

from datamining_app.algorithms.base import BaseAlgorithm
from datamining_app.core.models import Dataset
from datamining_app.data.embedded_db.catalog import DEFAULT_TABLES
from datamining_app.i18n import i18n
from datamining_app.report_defaults import params_for
from datamining_app.ui.components.console_widget import ConsoleWidget
from datamining_app.ui.components.data_panel import DataPanel
from datamining_app.ui.components.param_panel import ParamPanel
from datamining_app.ui.components.predict_panel import PredictPanel
from datamining_app.ui.components.viz_popup import VizPopup
from datamining_app.ui.dialogs.pseudocode_dialog import PseudocodeDialog
from datamining_app.ui.dialogs.stepbystep_dialog import StepByStepDialog


class AlgorithmPage(ttk.Frame):
    def __init__(
        self,
        master,
        algorithm: BaseAlgorithm,
        page_key: str,
        on_busy: Callable[..., None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.algorithm = algorithm
        self.page_key = page_key
        self.on_busy = on_busy or (lambda *_: None)
        self.dataset: Dataset | None = None
        self.running = False
        self._highlight = None

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        left = ttk.Frame(pane)
        right = ttk.Frame(pane)
        pane.add(left, weight=1)
        pane.add(right, weight=3)

        self.data_panel = DataPanel(
            left,
            on_dataset=self._on_dataset,
            default_table=DEFAULT_TABLES.get(page_key, ""),
            algorithm_key=page_key,
            get_params=lambda: self.param_panel.get_values() if hasattr(self, "param_panel") else {},
        )
        self.data_panel.pack(fill="both", expand=True, padx=6, pady=6)
        defaults = params_for(page_key, DEFAULT_TABLES.get(page_key, ""))
        self.param_panel = ParamPanel(left, schema=algorithm.param_schema, defaults=defaults)
        self.param_panel.pack(fill="x", padx=6, pady=(0, 6))

        self.console = ConsoleWidget(right)
        self.console.pack(fill="both", expand=True, padx=6, pady=6)
        self.predict_panel = PredictPanel(right, on_predict=self._predict)
        i18n.subscribe(self.refresh_texts)

    def refresh_texts(self, _lang: str | None = None) -> None:
        return

    def _on_dataset(self, dataset: Dataset) -> None:
        self.dataset = dataset
        extra = params_for(self.page_key, dataset.name)
        meta = self.data_panel.current_meta()
        if meta and meta.default_config:
            extra = {**extra, **meta.default_config}
        self.param_panel.apply_dataset(dataset.headers, extra)

    def run_algorithm(self) -> None:
        if self.running:
            return
        if not self.dataset:
            self.console.log_info(i18n.t("need_dataset"))
            return
        params = self.param_panel.get_values()
        self.running = True
        self.on_busy(True)
        self.console.clear()
        self.console.log_info(f"Chạy {self.algorithm.name}...")
        dataset = self.dataset

        def worker() -> None:
            try:
                result = self.algorithm.run(dataset, params)
                self.after(0, lambda: self._finish(result, None))
            except Exception as exc:
                self.after(0, lambda: self._finish(None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _finish(self, result, error: Exception | None) -> None:
        self.running = False
        self.on_busy(False, error is None)
        if error is not None:
            messagebox.showerror(i18n.t("app_title"), i18n.t("error", msg=str(error)))
            self.console.log_info(i18n.t("error", msg=str(error)))
            return
        name = self.dataset.name if self.dataset else ""
        n_rows = len(self.dataset.rows) if self.dataset else 0
        self.console.show_result(result, name, n_rows)
        if self.algorithm.supports_predict:
            fields = _predict_fields(self.algorithm, self.dataset)
            force_entry = self.page_key == "kmeans"
            self.predict_panel.set_fields(fields, self.dataset, force_entry=force_entry)
            if not self.predict_panel.winfo_ismapped():
                self.predict_panel.pack(fill="x", padx=6, pady=(0, 6))

    def show_graph(self) -> None:
        result = self.algorithm._last_result
        if not result:
            self.console.log_info(i18n.t("no_result"))
            return
        kind = "tree" if self.page_key in {"id3", "cart_gini"} else "cluster"
        VizPopup(self.winfo_toplevel(), result, kind, highlight=self._highlight)

    def show_pseudocode(self) -> None:
        PseudocodeDialog(self.winfo_toplevel(), self.page_key)

    def show_demo(self) -> None:
        result = self.algorithm._last_result
        if not result:
            self.console.log_info(i18n.t("need_run_first"))
            return
        StepByStepDialog(self.winfo_toplevel(), result)

    def toggle_predict(self) -> None:
        if not self.algorithm.supports_predict:
            return
        if self.predict_panel.winfo_ismapped():
            self.predict_panel.pack_forget()
        else:
            self.predict_panel.pack(fill="x", padx=6, pady=(0, 6))

    def _predict(self, sample: dict[str, Any]) -> None:
        if not getattr(self.algorithm, "_trained", False):
            self.console.log_info(i18n.t("need_run_first"))
            return
        try:
            pred = self.algorithm.predict(sample)
        except Exception as exc:
            messagebox.showerror(i18n.t("app_title"), str(exc))
            return
        if self.algorithm._last_result is not None:
            self.algorithm._last_result.predictions.append(
                {"label": pred.label, "explanation": pred.explanation, "sample": pred.sample}
            )
        self.console.log_prediction(pred.label, pred.explanation)
        self._highlight = pred.details.get("vector")


def _predict_fields(algorithm: BaseAlgorithm, dataset: Dataset | None) -> list[str]:
    output = (algorithm._last_result.output if algorithm._last_result else {}) or {}
    if "features" in output:
        return list(output["features"])
    if "condition" in output:
        reducts = output.get("reducts") or []
        if reducts:
            return list(reducts[0])
        return list(output["condition"])
    if dataset:
        decision = output.get("decision")
        return [h for h in dataset.headers if h != decision]
    return []
