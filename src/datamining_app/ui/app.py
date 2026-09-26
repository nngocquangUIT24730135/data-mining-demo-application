from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from datamining_app.console.tags import (
    DEFAULT_FONT_SIZE,
    FONT_SIZE_PRESETS,
    adjust_font_size,
    get_font_size,
    reset_font_size,
    set_font_size,
)
from datamining_app.i18n import i18n
from datamining_app.ui.pages.apriori_page import AprioriPage
from datamining_app.ui.pages.binary_vector_page import BinaryVectorPage
from datamining_app.ui.pages.cart_gini_page import CARTGiniPage
from datamining_app.ui.pages.home_page import HomePage
from datamining_app.ui.pages.id3_page import ID3Page
from datamining_app.ui.pages.kmeans_page import KMeansPage
from datamining_app.ui.pages.laplace_bayes_page import LaplaceBayesPage
from datamining_app.ui.pages.naive_bayes_page import NaiveBayesPage
from datamining_app.ui.pages.rough_set_page import RoughSetPage
from datamining_app.ui.theme import ALGO_KEYS, HAS_STEP_VIZ, HAS_VIZ, apply_ui_theme
from datamining_app.ui.windowing import enable_maximize, maximize


class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:
            pass
        apply_ui_theme()
        self.title(i18n.t("app_title"))
        self.minsize(960, 640)
        enable_maximize(self)
        maximize(self)

        self.pages: dict[str, ttk.Frame] = {}
        self.current_key = "home"
        self._algo_var = tk.StringVar()
        self._key_by_label: dict[str, str] = {}
        self._font_size_var = tk.IntVar(value=get_font_size())
        self._busy = False
        self._pulse_id: str | None = None

        self._build_menu()
        self._build_toolbar()
        self._bind_font_shortcuts()

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
            "cart_gini": lambda: CARTGiniPage(self.container, on_busy=self._set_busy),
            "naive_bayes": lambda: NaiveBayesPage(self.container, on_busy=self._set_busy),
            "naive_bayes_laplace": lambda: LaplaceBayesPage(self.container, on_busy=self._set_busy),
            "kmeans": lambda: KMeansPage(self.container, on_busy=self._set_busy),
        }

        i18n.subscribe(self.refresh_texts)
        self.show_page("home")

    def _build_menu(self) -> None:
        self.menubar = tk.Menu(self)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label=i18n.t("menu_load_csv"), command=self._menu_load_csv)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=i18n.t("menu_exit"), command=self.destroy)
        self.menubar.add_cascade(label=i18n.t("menu_file"), menu=self.file_menu)

        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.view_menu.add_command(
            label=i18n.t("menu_font_increase"),
            command=lambda: self._change_font(+1),
            accelerator="Ctrl++",
        )
        self.view_menu.add_command(
            label=i18n.t("menu_font_decrease"),
            command=lambda: self._change_font(-1),
            accelerator="Ctrl+-",
        )
        self.view_menu.add_command(
            label=i18n.t("menu_font_reset"),
            command=self._reset_font,
            accelerator="Ctrl+0",
        )
        self.view_menu.add_separator()
        self.font_size_menu = tk.Menu(self.view_menu, tearoff=0)
        for size in FONT_SIZE_PRESETS:
            self.font_size_menu.add_radiobutton(
                label=str(size),
                value=size,
                variable=self._font_size_var,
                command=lambda s=size: self._set_font(s),
            )
        self.view_menu.add_cascade(label=i18n.t("menu_font_size"), menu=self.font_size_menu)
        self.menubar.add_cascade(label=i18n.t("menu_view"), menu=self.view_menu)

        self.lang_menu = tk.Menu(self.menubar, tearoff=0)
        self.lang_menu.add_command(label=i18n.t("lang_vi"), command=lambda: i18n.set_language("vi"))
        self.lang_menu.add_command(label=i18n.t("lang_en"), command=lambda: i18n.set_language("en"))
        self.menubar.add_cascade(label=i18n.t("menu_lang"), menu=self.lang_menu)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.help_menu.add_command(label=i18n.t("help_guide"), command=self._help_guide)
        self.help_menu.add_command(label=i18n.t("help_about"), command=self._help_about)
        self.menubar.add_cascade(label=i18n.t("menu_help"), menu=self.help_menu)
        self.config(menu=self.menubar)
        self._font_size_var.set(get_font_size())

    def _bind_font_shortcuts(self) -> None:
        for seq in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
            self.bind_all(seq, lambda _e: self._change_font(+1))
        for seq in ("<Control-minus>", "<Control-KP_Subtract>"):
            self.bind_all(seq, lambda _e: self._change_font(-1))
        self.bind_all("<Control-0>", lambda _e: self._reset_font())
        self.bind_all("<Control-KP_0>", lambda _e: self._reset_font())

    def _change_font(self, delta: int) -> None:
        adjust_font_size(delta)
        self._font_size_var.set(get_font_size())

    def _set_font(self, size: int) -> None:
        set_font_size(size)
        self._font_size_var.set(get_font_size())

    def _reset_font(self) -> None:
        reset_font_size()
        self._font_size_var.set(DEFAULT_FONT_SIZE)

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        self.algo_lbl = ttk.Label(bar, text=i18n.t("choose_algo"))
        self.algo_lbl.pack(side="left")
        self.algo_combo = ttk.Combobox(bar, textvariable=self._algo_var, state="readonly", width=22)
        self.algo_combo.pack(side="left", padx=8)
        self.algo_combo.bind("<<ComboboxSelected>>", self._on_algo_change)
        self.run_btn = ttk.Button(bar, text=i18n.t("run"), command=self._run, style="Primary.TButton")
        self.run_btn.pack(side="left", padx=(8, 4))
        self.pseudo_btn = ttk.Button(
            bar, text=i18n.t("pseudocode"), command=self._pseudocode, style="Secondary.TButton"
        )
        self.pseudo_btn.pack(side="left", padx=4)
        self.demo_btn = ttk.Button(
            bar, text=i18n.t("demo_steps"), command=self._demo, style="Secondary.TButton"
        )
        self.demo_btn.pack(side="left", padx=4)
        self.viz_btn = ttk.Button(
            bar, text=i18n.t("result_graph"), command=self._graph, style="Secondary.TButton"
        )
        self.viz_btn.pack(side="left", padx=4)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140, style="Run.Horizontal.TProgressbar")
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
        if self._busy:
            self.run_btn.configure(state="disabled", style="Running.TButton", text=i18n.t("running"))
        else:
            self.run_btn.configure(
                state="disabled" if home else "normal",
                style="Primary.TButton",
                text=i18n.t("run"),
            )
        self.pseudo_btn.configure(state="disabled" if home else "normal")
        self.demo_btn.configure(state="normal" if self.current_key in HAS_STEP_VIZ else "disabled")
        self.viz_btn.configure(state="normal" if self.current_key in HAS_VIZ else "disabled")

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

    def _menu_load_csv(self) -> None:
        page = self.current_page()
        if page and hasattr(page, "data_panel"):
            page.data_panel.load_csv()

    def _set_busy(self, busy: bool, success: bool = False) -> None:
        self._cancel_pulse()
        self._busy = busy
        self.configure(cursor="watch" if busy else "")
        page = self.current_page()
        console = getattr(page, "console", None)
        if console is not None and hasattr(console, "set_busy"):
            console.set_busy(busy)
        if busy:
            self.run_btn.configure(text=i18n.t("running"), state="disabled", style="Running.TButton")
            self.progress.pack(side="left", padx=8)
            self.progress.start(12)
            return
        self.progress.stop()
        self.progress.pack_forget()
        self._sync_toolbar()
        if success and self.current_key != "home":
            self._pulse_done(0)

    def _help_guide(self) -> None:
        messagebox.showinfo(i18n.t("help_guide"), i18n.t("guide"))

    def _help_about(self) -> None:
        messagebox.showinfo(i18n.t("help_about"), i18n.t("about"))

    def _cancel_pulse(self) -> None:
        if self._pulse_id is not None:
            self.after_cancel(self._pulse_id)
            self._pulse_id = None

    def _pulse_done(self, step: int) -> None:
        if self._busy or self.current_key == "home":
            return
        styles = ("Done.TButton", "Primary.TButton", "Done.TButton", "Primary.TButton")
        self.run_btn.configure(style=styles[step], state="normal", text=i18n.t("run"))
        if step + 1 < len(styles):
            self._pulse_id = self.after(160, lambda: self._pulse_done(step + 1))
        else:
            self._pulse_id = None

    def refresh_texts(self, _lang: str | None = None) -> None:
        self.title(i18n.t("app_title"))
        self._build_menu()
        self.algo_lbl.configure(text=i18n.t("choose_algo"))
        self.pseudo_btn.configure(text=i18n.t("pseudocode"))
        self.demo_btn.configure(text=i18n.t("demo_steps"))
        self.viz_btn.configure(text=i18n.t("result_graph"))
        self._sync_toolbar()
        self._refresh_algo_combo()
