import tkinter as tk

from datamining_app.console.tags import MONO_CANDIDATES, TAGS
from datamining_app.ui.components.console_widget import ConsoleWidget
from datamining_app.ui.dialogs.pseudocode_dialog import is_declaration, pseudocode_spans


def test_console_tags_match_contrast_palette():
    assert TAGS["table_border"] == "#6E7681"
    assert TAGS["table_header"] == "#79C0FF"
    assert TAGS["table_row"] == "#F0F6FC"
    assert TAGS["step_header"] == "#38BDF8"
    assert TAGS["concept"] == "#E879F9"
    assert TAGS["pseudocode"] == "#86EFAC"
    assert TAGS["formula"] == "#FDBA74"
    assert TAGS["result"] == "#FBBF24"
    assert TAGS["accept"] == "#4ADE80"
    assert TAGS["reject"] == "#F87171"
    assert TAGS["warning"] == "#FB923C"
    assert MONO_CANDIDATES[0] == "Consolas"
    assert "Courier New" in MONO_CANDIDATES


def test_pseudocode_declaration_and_spans():
    assert is_declaration("ALGORITHM ID3(S, Attributes, Target)")
    assert is_declaration("  INPUT  : S — tập mẫu")
    assert not is_declaration("  Bước 1: tính Entropy(S)")

    line = "  Bước 1: Tạo nút; tính Entropy(S) = -Σ pᵢ log₂(pᵢ)"
    tags = {tag for _start, _end, tag in pseudocode_spans(line)}
    assert "step" in tags
    assert "formula" in tags

    sub = "     Bước 2a [Gán cụm] VỚI mỗi xᵢ"
    assert pseudocode_spans(sub)[0][2] == "step"


def test_console_wrap_toggle():
    root = tk.Tk()
    root.withdraw()
    try:
        widget = ConsoleWidget(root)
        assert widget.text.cget("wrap") == "word"
        assert widget.text._hscroll.grid_info() == {}
        widget.toggle_wrap()
        assert widget.text.cget("wrap") == "none"
        assert widget.text._hscroll.grid_info().get("row") == 1
        widget.toggle_wrap()
        assert widget.text.cget("wrap") == "word"
        assert widget.wrap_btn.cget("style") == "ToolOn.TButton"
    finally:
        root.destroy()
