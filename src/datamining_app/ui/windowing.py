from __future__ import annotations

import tkinter as tk


def maximize(window: tk.Misc) -> None:
    """Show a Tk window maximized (full screen with window chrome)."""

    def _apply() -> None:
        try:
            window.state("zoomed")
            return
        except tk.TclError:
            pass
        try:
            window.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass
        w = window.winfo_screenwidth()
        h = window.winfo_screenheight()
        window.geometry(f"{w}x{h}+0+0")

    window.after_idle(_apply)
