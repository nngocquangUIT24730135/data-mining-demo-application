from tkinter import ttk

ALGO_KEYS = [
    ("home", "home"),
    ("apriori", "apriori"),
    ("binary_vector", "binary_vector"),
    ("rough_set", "rough_set"),
    ("id3", "id3"),
    ("cart_gini", "cart_gini"),
    ("naive_bayes", "naive_bayes"),
    ("naive_bayes_laplace", "naive_bayes_laplace"),
    ("kmeans", "kmeans"),
]

HAS_VIZ = {"id3", "cart_gini", "kmeans"}
HAS_STEP_VIZ = {"id3", "cart_gini", "kmeans"}

FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_SECTION = ("Segoe UI", 10, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_SUBTEXT = ("Segoe UI", 9)

COLOR_TEXT = "#1F2328"
COLOR_SUBTEXT = "#57606A"
COLOR_PRIMARY = "#0969DA"
COLOR_PRIMARY_HOVER = "#0854C2"
COLOR_PRIMARY_DISABLED = "#8CB4E8"
COLOR_RUNNING = "#D97706"
COLOR_DONE = "#2DA44E"
COLOR_SECONDARY_BG = "#F6F8FA"
COLOR_SECONDARY_FG = "#24292F"
COLOR_BORDER = "#D0D7DE"
COLOR_ZEBRA_ODD = "#FFFFFF"
COLOR_ZEBRA_EVEN = "#F8FAFC"
COLOR_SELECT = "#DDF4FF"


def apply_ui_theme() -> None:
    """Clam styles for the primary action, secondary tools, and data grid."""
    style = ttk.Style()
    style.configure(".", font=FONT_BODY)
    style.configure("TLabel", font=FONT_BODY, foreground=COLOR_TEXT)
    style.configure("TLabelframe.Label", font=FONT_SECTION, foreground=COLOR_TEXT)
    style.configure("TButton", font=FONT_BODY, padding=(8, 4))

    style.configure(
        "Primary.TButton",
        font=FONT_SECTION,
        background=COLOR_PRIMARY,
        foreground="#FFFFFF",
        bordercolor=COLOR_PRIMARY,
        lightcolor=COLOR_PRIMARY,
        darkcolor=COLOR_PRIMARY_HOVER,
        padding=(16, 6),
        focuscolor=COLOR_PRIMARY,
    )
    style.map(
        "Primary.TButton",
        background=[
            ("disabled", COLOR_PRIMARY_DISABLED),
            ("pressed", COLOR_PRIMARY_HOVER),
            ("active", COLOR_PRIMARY_HOVER),
        ],
        foreground=[("disabled", "#FFFFFF"), ("!disabled", "#FFFFFF")],
        bordercolor=[
            ("disabled", COLOR_PRIMARY_DISABLED),
            ("active", COLOR_PRIMARY_HOVER),
        ],
        darkcolor=[("disabled", COLOR_PRIMARY_DISABLED), ("active", COLOR_PRIMARY_HOVER)],
        lightcolor=[("disabled", COLOR_PRIMARY_DISABLED), ("active", COLOR_PRIMARY_HOVER)],
    )

    style.configure(
        "Running.TButton",
        font=FONT_SECTION,
        background=COLOR_RUNNING,
        foreground="#FFFFFF",
        bordercolor=COLOR_RUNNING,
        lightcolor=COLOR_RUNNING,
        darkcolor="#B45309",
        padding=(16, 6),
        focuscolor=COLOR_RUNNING,
    )
    style.map(
        "Running.TButton",
        background=[("disabled", COLOR_RUNNING), ("active", COLOR_RUNNING)],
        foreground=[("disabled", "#FFFFFF")],
        bordercolor=[("disabled", COLOR_RUNNING)],
        darkcolor=[("disabled", "#B45309")],
        lightcolor=[("disabled", COLOR_RUNNING)],
    )

    style.configure(
        "Done.TButton",
        font=FONT_SECTION,
        background=COLOR_DONE,
        foreground="#FFFFFF",
        bordercolor=COLOR_DONE,
        lightcolor=COLOR_DONE,
        darkcolor="#1A7F37",
        padding=(16, 6),
    )
    style.map(
        "Done.TButton",
        background=[("active", COLOR_DONE), ("!disabled", COLOR_DONE)],
        foreground=[("!disabled", "#FFFFFF")],
    )

    style.configure(
        "Secondary.TButton",
        font=FONT_BODY,
        background=COLOR_SECONDARY_BG,
        foreground=COLOR_SECONDARY_FG,
        bordercolor=COLOR_BORDER,
        lightcolor="#FFFFFF",
        darkcolor=COLOR_BORDER,
        padding=(10, 5),
    )
    style.map(
        "Secondary.TButton",
        background=[
            ("disabled", COLOR_SECONDARY_BG),
            ("pressed", "#EAEFF2"),
            ("active", "#FFFFFF"),
        ],
        foreground=[("disabled", "#8C959F"), ("!disabled", COLOR_SECONDARY_FG)],
        bordercolor=[("disabled", COLOR_BORDER), ("active", COLOR_PRIMARY)],
    )

    style.configure(
        "Tool.TButton",
        font=FONT_SUBTEXT,
        background=COLOR_SECONDARY_BG,
        foreground=COLOR_SECONDARY_FG,
        bordercolor=COLOR_BORDER,
        lightcolor="#FFFFFF",
        darkcolor=COLOR_BORDER,
        padding=(6, 2),
    )
    style.map(
        "Tool.TButton",
        background=[
            ("disabled", COLOR_SECONDARY_BG),
            ("pressed", "#EAEFF2"),
            ("active", "#FFFFFF"),
        ],
        foreground=[("disabled", "#8C959F"), ("!disabled", COLOR_SECONDARY_FG)],
    )
    style.configure(
        "ToolOn.TButton",
        font=FONT_SUBTEXT,
        background=COLOR_SELECT,
        foreground=COLOR_TEXT,
        bordercolor=COLOR_PRIMARY,
        lightcolor=COLOR_SELECT,
        darkcolor=COLOR_PRIMARY,
        padding=(6, 2),
    )
    style.map(
        "ToolOn.TButton",
        background=[("pressed", "#B6E3FF"), ("active", "#C8E9FF")],
        foreground=[("!disabled", COLOR_TEXT)],
        bordercolor=[("active", COLOR_PRIMARY)],
    )

    style.configure(
        "Data.Treeview",
        font=FONT_BODY,
        background=COLOR_ZEBRA_ODD,
        fieldbackground=COLOR_ZEBRA_ODD,
        foreground=COLOR_TEXT,
        rowheight=22,
        bordercolor=COLOR_BORDER,
    )
    style.map(
        "Data.Treeview",
        background=[("selected", COLOR_SELECT)],
        foreground=[("selected", COLOR_TEXT)],
    )
    style.configure(
        "Data.Treeview.Heading",
        font=FONT_SECTION,
        background=COLOR_SECONDARY_BG,
        foreground=COLOR_TEXT,
        relief="flat",
    )
    style.map("Data.Treeview.Heading", background=[("active", "#EAEFF2")])

    style.configure(
        "Run.Horizontal.TProgressbar",
        troughcolor=COLOR_SECONDARY_BG,
        background=COLOR_PRIMARY,
        thickness=8,
        bordercolor=COLOR_BORDER,
        lightcolor=COLOR_PRIMARY,
        darkcolor=COLOR_PRIMARY,
    )


def apply_data_tree_tags(tree: ttk.Treeview) -> None:
    tree.tag_configure("odd", background=COLOR_ZEBRA_ODD)
    tree.tag_configure("even", background=COLOR_ZEBRA_EVEN)
