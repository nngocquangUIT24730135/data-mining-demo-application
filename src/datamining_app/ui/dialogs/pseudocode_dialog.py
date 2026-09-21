from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from datamining_app.algorithms.pseudocode import PSEUDOCODE
from datamining_app.i18n import i18n


class PseudocodeDialog(tk.Toplevel):
    def __init__(self, master, algorithm_key: str, title: str | None = None) -> None:
        super().__init__(master)
        self.title(title or i18n.t("pseudocode"))
        self.geometry("720x560")
        self.transient(master)
        text = PSEUDOCODE.get(algorithm_key, "Chưa có mã giả.")
        box = ScrolledText(self, wrap="none", font=("Consolas", 11), state="normal")
        box.pack(fill="both", expand=True, padx=8, pady=8)
        box.insert("1.0", text)
        box.configure(state="disabled")
        ttk.Button(self, text=i18n.t("close"), command=self.destroy).pack(pady=(0, 10))
