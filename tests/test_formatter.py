from datamining_app.console.formatter import ConsoleFormatter, display_width
from datamining_app.core.models import StepLog


def test_table_alignment_and_width():
    fmt = ConsoleFormatter(width=80)
    table = fmt.render_table(
        ["Tập mặt hàng", "Count", "SP", "Kết luận"],
        [
            ["{Bread}", 4, "80.00%", "✓ Chấp nhận"],
            ["{Eggs}", 1, "20.00%", "✗ Loại bỏ"],
        ],
        ["left", "right", "right", "left"],
    )
    lines = table.splitlines()
    assert lines[0].startswith("┌")
    assert lines[-1].startswith("└")
    widths = {display_width(line) for line in lines}
    assert len(widths) == 1
    assert "✓ Chấp nhận" in table
    assert "{Bread}" in table


def test_step_header_and_candidate_table():
    fmt = ConsoleFormatter(width=72)
    step = StepLog(
        step_number=1,
        title="Khởi tạo tập ứng cử viên C₁",
        description="⇒ L₁ = {Bread, Jam}",
        data={
            "candidates": [
                {"itemset": ["Bread"], "count": 4, "support": 0.8, "accepted": True},
                {"itemset": ["Eggs"], "count": 1, "support": 0.2, "accepted": False},
            ]
        },
        kind="table",
    )
    text = fmt.render_step(step)
    assert "Bước 1" in text
    assert "✓ Chấp nhận" in text
    assert "✗ Loại bỏ" in text
    assert "⇒ L₁ = {Bread, Jam}" in text
    assert "Tập ứng viên |" not in text


def test_banner_skips_exclude_cols():
    fmt = ConsoleFormatter()
    text = fmt.banner("Apriori", "apriori_daily_basket_5tx", 5, {"minsup": 0.5, "minconf": 0.75, "exclude_cols": ["tid"]})
    assert "THUẬT TOÁN APRIORI" in text
    assert "50.00%" in text
    assert "exclude_cols" not in text


def test_rules_table_not_duplicated_as_text_list():
    fmt = ConsoleFormatter()
    step = StepLog(
        step_number=7,
        title="Sinh luật kết hợp",
        description="",
        data={
            "rules": [
                {
                    "antecedent": ["Cornflakes"],
                    "consequent": ["Jam"],
                    "support": 0.6,
                    "confidence": 1.0,
                }
            ],
            "conclusion": "  → 1 luật đạt minconf = 75.00%.",
        },
        kind="rules",
    )
    text = fmt.render_step(step)
    assert "{Cornflakes} → {Jam}" in text
    assert "100.00%" in text
    assert text.count("{Cornflakes}") == 1
    assert "Các luật đạt" not in text
    assert all(display_width(line) == display_width(text.splitlines()[2]) for line in text.splitlines() if line.startswith(("┌", "├", "└", "│")))


def test_vector_and_column_uses_wedge():
    fmt = ConsoleFormatter()
    step = StepLog(
        step_number=3,
        title="Cấp k = 2",
        description="",
        data={
            "candidates": [
                {
                    "itemset": ["Bread", "Jam"],
                    "left": ["Bread"],
                    "right": ["Jam"],
                    "vector": [1, 1, 0],
                    "count": 2,
                    "support": 0.4,
                    "accepted": False,
                }
            ],
            "conclusion": "  → F_2 = ∅",
        },
        kind="table",
    )
    text = fmt.render_step(step)
    assert "∧" in text
    assert "Phép AND" in text
    lines = [line for line in text.splitlines() if line.startswith(("┌", "├", "└", "│"))]
    assert len({display_width(line) for line in lines}) == 1
