from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from datamining_app.i18n import i18n
from datamining_app.ui.pages.apriori_page import AprioriPage
from datamining_app.ui.pages.binary_vector_page import BinaryVectorPage
from datamining_app.ui.pages.home_page import HomePage
from datamining_app.ui.pages.id3_page import ID3Page
from datamining_app.ui.pages.kmeans_page import KMeansPage
from datamining_app.ui.pages.naive_bayes_page import NaiveBayesPage
from datamining_app.ui.pages.rough_set_page import RoughSetPage
from datamining_app.ui.theme import ALGO_KEYS, HAS_DEMO, HAS_VIZ
from datamining_app.ui.windowing import maximize


class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:
            pass
        self.title(i18n.t("app_title"))
        self.minsize(960, 640)
        maximize(self)

        self.pages: dict[str, ttk.Frame] = {}
        self.current_key = "home"
        self._algo_var = tk.StringVar()
        self._key_by_label: dict[str, str] = {}

        self._build_menu()
        self._build_toolbar()

        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.factories = {
            "home": lambda: HomePage(self.container),
            "apriori": lambda: AprioriPage(self.container, on_busy=self._set_busy),
            "binary_vector": lambda: BinaryVectorPage(self.container, on_busy=self._set_busy),
            "rough_set": lambda: RoughSetPage(self.container, on_busy=self._set_busy),
            "id3": lambda: ID3Page(self.container, on_busy=self._set_busy),
            "naive_bayes": lambda: NaiveBayesPage(self.container, on_busy=self._set_busy),
            "kmeans": lambda: KMeansPage(self.container, on_busy=self._set_busy),
        }

        i18n.subscribe(self.refresh_texts)
        self.show_page("home")

    def _build_menu(self) -> None:
        self.menubar = tk.Menu(self)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label=i18n.t("menu_load_csv"), command=self._menu_load_csv)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=i18n.t("menu_export_txt"), command=self._menu_export_txt)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=i18n.t("menu_exit"), command=self.destroy)
        self.menubar.add_cascade(label=i18n.t("menu_file"), menu=self.file_menu)

        self.lang_menu = tk.Menu(self.menubar, tearoff=0)
        self.lang_menu.add_command(label=i18n.t("lang_vi"), command=lambda: i18n.set_language("vi"))
        self.lang_menu.add_command(label=i18n.t("lang_en"), command=lambda: i18n.set_language("en"))
        self.menubar.add_cascade(label=i18n.t("menu_lang"), menu=self.lang_menu)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.help_menu.add_command(label=i18n.t("help_guide"), command=self._help_guide)
        self.help_menu.add_command(label=i18n.t("help_about"), command=self._help_about)
        self.menubar.add_cascade(label=i18n.t("menu_help"), menu=self.help_menu)
        self.config(menu=self.menubar)

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        self.algo_lbl = ttk.Label(bar, text=i18n.t("choose_algo"))
        self.algo_lbl.pack(side="left")
        self.algo_combo = ttk.Combobox(bar, textvariable=self._algo_var, state="readonly", width=22)
        self.algo_combo.pack(side="left", padx=8)
        self.algo_combo.bind("<<ComboboxSelected>>", self._on_algo_change)
        self.run_btn = ttk.Button(bar, text=i18n.t("run"), command=self._run)
        self.run_btn.pack(side="left", padx=4)
        self.pseudo_btn = ttk.Button(bar, text=i18n.t("pseudocode"), command=self._pseudocode)
        self.pseudo_btn.pack(side="left", padx=4)
        self.demo_btn = ttk.Button(bar, text=i18n.t("demo_steps"), command=self._demo)
        self.demo_btn.pack(side="left", padx=4)
        self.viz_btn = ttk.Button(bar, text=i18n.t("show_graph"), command=self._graph)
        self.viz_btn.pack(side="left", padx=4)
        self.predict_btn = ttk.Button(bar, text=i18n.t("predict_toggle"), command=self._predict)
        self.predict_btn.pack(side="left", padx=4)
        self.txt_btn = ttk.Button(bar, text=i18n.t("export_txt"), command=self._menu_export_txt)
        self.txt_btn.pack(side="left", padx=4)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self._refresh_algo_combo()

    def _refresh_algo_combo(self) -> None:
        labels = []
        self._key_by_label = {}
        for key, label_key in ALGO_KEYS:
            label = i18n.t(label_key)
            labels.append(label)
            self._key_by_label[label] = key
        self.algo_combo.configure(values=labels)
        current_label = next((i18n.t(lk) for k, lk in ALGO_KEYS if k == self.current_key), labels[0])
        self._algo_var.set(current_label)

    def _on_algo_change(self, _event=None) -> None:
        key = self._key_by_label.get(self._algo_var.get(), "home")
        self.show_page(key)

    def show_page(self, key: str) -> None:
        if key not in self.pages:
            page = self.factories[key]()
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page
        self.pages[key].tkraise()
        self.current_key = key
        self._refresh_algo_combo()
        self._sync_toolbar()

    def _sync_toolbar(self) -> None:
        home = self.current_key == "home"
        page = self.current_page()
        algo = getattr(page, "algorithm", None)
        self.run_btn.configure(state="disabled" if home else "normal")
        self.pseudo_btn.configure(state="disabled" if home else "normal")
        self.txt_btn.configure(state="disabled" if home else "normal")
        self.demo_btn.configure(state="normal" if self.current_key in HAS_DEMO else "disabled")
        self.viz_btn.configure(state="normal" if self.current_key in HAS_VIZ else "disabled")
        can_predict = bool(algo and getattr(algo, "supports_predict", False))
        self.predict_btn.configure(state="normal" if can_predict else "disabled")

    def current_page(self):
        return self.pages.get(self.current_key)

    def _run(self) -> None:
        page = self.current_page()
        if page and hasattr(page, "run_algorithm"):
            page.run_algorithm()

    def _pseudocode(self) -> None:
        page = self.current_page()
        if page and hasattr(page, "show_pseudocode"):
            page.show_pseudocode()

    def _demo(self) -> None:
        page = self.current_page()
        if page and hasattr(page, "show_demo"):
            page.show_demo()

    def _graph(self) -> None:
        page = self.current_page()
        if page and hasattr(page, "show_graph"):
            page.show_graph()

    def _predict(self) -> None:
        page = self.current_page()
        if page and hasattr(page, "toggle_predict"):
            page.toggle_predict()

    def _menu_load_csv(self) -> None:
        page = self.current_page()
        if page and hasattr(page, "data_panel"):
            page.data_panel.load_csv()

    def _menu_export_txt(self) -> None:
        page = self.current_page()
        if page and hasattr(page, "export_txt"):
            page.export_txt()

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.run_btn.configure(text=i18n.t("running"), state="disabled")
            self.progress.pack(side="left", padx=8)
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()
            self.run_btn.configure(
                text=i18n.t("run"),
                state="disabled" if self.current_key == "home" else "normal",
            )

    def _help_guide(self) -> None:
        messagebox.showinfo(i18n.t("help_guide"), i18n.t("guide"))

    def _help_about(self) -> None:
        messagebox.showinfo(i18n.t("help_about"), i18n.t("about"))

    def refresh_texts(self, _lang: str | None = None) -> None:
        self.title(i18n.t("app_title"))
        self._build_menu()
        self.algo_lbl.configure(text=i18n.t("choose_algo"))
        self.run_btn.configure(text=i18n.t("run"))
        self.pseudo_btn.configure(text=i18n.t("pseudocode"))
        self.demo_btn.configure(text=i18n.t("demo_steps"))
        self.viz_btn.configure(text=i18n.t("show_graph"))
        self.predict_btn.configure(text=i18n.t("predict_toggle"))
        self.txt_btn.configure(text=i18n.t("export_txt"))
        self._refresh_algo_combo()
