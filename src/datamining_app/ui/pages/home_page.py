from tkinter import ttk

from datamining_app.i18n import i18n
from datamining_app.ui.theme import FONT_BODY, FONT_TITLE


class HomePage(ttk.Frame):
    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.title_lbl = ttk.Label(self, text=i18n.t("home_title"), font=FONT_TITLE)
        self.title_lbl.pack(anchor="w", padx=24, pady=(24, 8))
        self.body = ttk.Label(self, text=i18n.t("home_body"), wraplength=720, justify="left", font=FONT_BODY)
        self.body.pack(anchor="w", padx=24)
        i18n.subscribe(self.refresh_texts)

    def refresh_texts(self, _lang: str | None = None) -> None:
        self.title_lbl.configure(text=i18n.t("home_title"))
        self.body.configure(text=i18n.t("home_body"))

    def run_algorithm(self) -> None:
        return
