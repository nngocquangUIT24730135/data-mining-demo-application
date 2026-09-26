from __future__ import annotations

import weakref
from typing import Callable

TAGS = {
    # Headings and structure — GitHub Dark, WCAG AA+ on CONSOLE_BG.
    "algo_header": "#F0C040",
    "step_header": "#38BDF8",
    "arrow": "#60A5FA",
    # Teaching content
    "concept": "#E879F9",
    "pseudocode": "#86EFAC",
    "formula": "#FDBA74",
    # Tables: border lifted from #484F58 (2.4:1) to #6E7681 (4.2:1).
    "table_border": "#6E7681",
    "table_header": "#79C0FF",
    "table_row": "#F0F6FC",
    # Status
    "result": "#FBBF24",
    "accept": "#4ADE80",
    "reject": "#F87171",
    "warning": "#FB923C",
    "info": "#C9D1D9",
}

CONSOLE_BG = "#0D1117"
CONSOLE_FG = "#E6EDF3"

MONO_CANDIDATES = ("Consolas", "Cascadia Code", "JetBrains Mono", "Courier New")
FONT_FAMILY = MONO_CANDIDATES[0]
DEFAULT_FONT_SIZE = 12
MIN_FONT_SIZE = 9
MAX_FONT_SIZE = 24
FONT_SIZE_PRESETS = (10, 12, 14, 16, 18, 20)

_font_size = DEFAULT_FONT_SIZE
_font_resolved = False
_subscribers: list[Callable[[int], None]] = []
_console_widgets: list[weakref.ref] = []

# Backward-compatible tuples (updated whenever size changes).
FONT_CONSOLE: tuple = (FONT_FAMILY, DEFAULT_FONT_SIZE)
FONT_CONSOLE_BOLD: tuple = (FONT_FAMILY, DEFAULT_FONT_SIZE, "bold")


def get_font_size() -> int:
    return _font_size


def ensure_console_font(widget=None) -> str:
    """Pick the first installed monospace face from MONO_CANDIDATES."""
    global FONT_FAMILY, FONT_CONSOLE, FONT_CONSOLE_BOLD, _font_resolved
    if _font_resolved:
        return FONT_FAMILY
    try:
        import tkinter.font as tkfont

        families = set(tkfont.families(widget))
    except Exception:
        return FONT_FAMILY
    chosen = next((name for name in MONO_CANDIDATES if name in families), FONT_FAMILY)
    FONT_FAMILY = chosen
    FONT_CONSOLE = console_font()
    FONT_CONSOLE_BOLD = console_font_bold()
    _font_resolved = True
    return FONT_FAMILY


def console_font() -> tuple:
    return (FONT_FAMILY, _font_size)


def console_font_bold() -> tuple:
    return (FONT_FAMILY, _font_size, "bold")


def set_font_size(size: int) -> None:
    global _font_size, FONT_CONSOLE, FONT_CONSOLE_BOLD
    size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, int(size)))
    if size == _font_size:
        return
    _font_size = size
    FONT_CONSOLE = console_font()
    FONT_CONSOLE_BOLD = console_font_bold()
    _apply_to_registered()
    for callback in list(_subscribers):
        callback(_font_size)


def adjust_font_size(delta: int) -> None:
    set_font_size(_font_size + int(delta))


def reset_font_size() -> None:
    set_font_size(DEFAULT_FONT_SIZE)


def subscribe_font(callback: Callable[[int], None]) -> None:
    if callback not in _subscribers:
        _subscribers.append(callback)


def unsubscribe_font(callback: Callable[[int], None]) -> None:
    if callback in _subscribers:
        _subscribers.remove(callback)


def register_console_text(widget) -> None:
    _prune_console_refs()
    _console_widgets.append(weakref.ref(widget))
    apply_font_to_text(widget)


def apply_font_to_text(text) -> None:
    """Update base font and bold tags on a tk.Text console widget."""
    try:
        if not text.winfo_exists():
            return
    except Exception:
        return
    ensure_console_font(text)
    normal = console_font()
    bold = console_font_bold()
    text.configure(font=normal)
    for name in TAGS:
        text.tag_configure(name, foreground=TAGS[name])
    text.tag_configure("step_header", foreground=TAGS["step_header"], font=bold)
    text.tag_configure("algo_header", foreground=TAGS["algo_header"], font=bold)
    text.tag_configure("result", foreground=TAGS["result"], font=bold)
    text.tag_configure("warning", foreground=TAGS["warning"], font=bold)
    text.tag_configure("table_header", foreground=TAGS["table_header"], font=bold)
    text.tag_configure("table_row", foreground=TAGS["table_row"])
    text.tag_configure("table_border", foreground=TAGS["table_border"])
    text.tag_configure("concept", foreground=TAGS["concept"])
    text.tag_configure("pseudocode", foreground=TAGS["pseudocode"])
    text.tag_configure("formula", foreground=TAGS["formula"])


def _prune_console_refs() -> None:
    alive = []
    for ref in _console_widgets:
        if ref() is not None:
            alive.append(ref)
    _console_widgets[:] = alive


def _apply_to_registered() -> None:
    _prune_console_refs()
    for ref in list(_console_widgets):
        widget = ref()
        if widget is not None:
            apply_font_to_text(widget)
