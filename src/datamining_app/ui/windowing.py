"""Helpers for Tk window chrome (maximize / resize) across platforms."""
from __future__ import annotations

import sys
import tkinter as tk

# Win32 styles — needed because Tk `transient()` often clears WS_MAXIMIZEBOX.
_GWL_STYLE = -16
_WS_MINIMIZEBOX = 0x00020000
_WS_MAXIMIZEBOX = 0x00010000
_WS_THICKFRAME = 0x00040000
_WS_SIZEBOX = _WS_THICKFRAME


def enable_maximize(window: tk.Misc) -> None:
    """Allow the user to maximize: enable resize + the title-bar maximize button.

    On Windows, transient Toplevels frequently lose WS_MAXIMIZEBOX; restore it
    via Win32 after the window is mapped.
    """
    try:
        window.resizable(True, True)
    except tk.TclError:
        pass

    if sys.platform == "win32":
        window.after_idle(lambda: _win32_enable_maximize_box(window))


def maximize(window: tk.Misc) -> None:
    """Show a Tk window maximized (full screen with window chrome)."""
    enable_maximize(window)

    def _apply() -> None:
        try:
            window.state("zoomed")
        except tk.TclError:
            try:
                window.attributes("-zoomed", True)
            except tk.TclError:
                w = window.winfo_screenwidth()
                h = window.winfo_screenheight()
                window.geometry(f"{w}x{h}+0+0")
        # Zooming can reset Win32 styles; restore maximize affordance.
        enable_maximize(window)

    window.after_idle(_apply)


def _win32_enable_maximize_box(window: tk.Misc) -> None:
    try:
        import ctypes
    except ImportError:
        return
    try:
        window.update_idletasks()
        hwnd = int(window.winfo_id())
        # Tk often nests the client under a parent frame; style the outer HWND.
        parent = ctypes.windll.user32.GetParent(hwnd)
        if parent:
            hwnd = parent
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, _GWL_STYLE)
        style |= _WS_MAXIMIZEBOX | _WS_MINIMIZEBOX | _WS_THICKFRAME | _WS_SIZEBOX
        user32.SetWindowLongW(hwnd, _GWL_STYLE, style)
        # Force a non-client redraw so the maximize button appears.
        user32.SetWindowPos(
            hwnd,
            0,
            0,
            0,
            0,
            0,
            0x0001 | 0x0002 | 0x0004 | 0x0020,  # NOSIZE|NOMOVE|NOZORDER|FRAMECHANGED
        )
    except Exception:
        return
