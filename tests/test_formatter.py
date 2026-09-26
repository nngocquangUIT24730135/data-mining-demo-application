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


def test_multiline_header_table():
    fmt = ConsoleFormatter()
    table = fmt.render_table(
        [
            "Thuộc tính xem xét\nphân nhánh",
            "Entropy tập dữ liệu\ntrước khi chia I(S)",
            "Entropy trung bình\nsau khi chia E(A, S)",
            "Mức tăng thông tin\nInformation Gain",
        ],
        [
            ["Outlook", "0.94003", "0.69354", "0.24649 ★"],
            ["Temperature", "0.94003", "0.91116", "0.02887"],
        ],
        ["left", "right", "right", "right"],
    )
    lines = table.splitlines()
    assert lines[0].startswith("┌")
    assert "I(S)" in table
    assert "E(A, S)" in table
    assert "Information Gain" in table
    assert "phân nhánh" in table
    # Header occupies two lines between top border and mid separator.
    assert lines[1].startswith("│")
    assert lines[2].startswith("│")
    assert lines[3].startswith("├")
    widths = {display_width(line) for line in lines}
    assert len(widths) == 1


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


def test_render_result_includes_pseudocode_block():
    from datamining_app.algorithms.kmeans import KMeansAlgorithm
    from datamining_app.core.models import Dataset
    from datamining_app.ui.components.console_widget import LineTagger

    ds = Dataset(
        name="tiny",
        headers=["x", "y"],
        rows=[{"x": i, "y": i} for i in range(6)],
        source="test",
    )
    result = KMeansAlgorithm().run(ds, {"k": 2, "max_iter": 5, "distance": "euclidean"})
    text = ConsoleFormatter().render_result(result, "tiny", 6)
    assert " MÃ GIẢ " in text
    assert "ALGORITHM K-MEANS" in text
    # Pseudocode box appears before first step
    assert text.index(" MÃ GIẢ ") < text.index("Khởi tạo")
    tagger = LineTagger()
    tags = [tagger.tag(line) for line in text.splitlines(True) if " MÃ GIẢ " in line or line.lstrip().startswith("│  ALGORITHM")]
    assert "pseudocode" in tags


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
    assert "1.0000000000" in text
    assert text.count("{Cornflakes}") == 1
    assert "Các luật đạt" not in text
    assert all(display_width(line) == display_width(text.splitlines()[2]) for line in text.splitlines() if line.startswith(("┌", "├", "└", "│")))


def test_line_tagger_distinguishes_table_header_and_rows():
    from datamining_app.ui.components.console_widget import LineTagger

    lines = [
        "┌────┬──────┐\n",
        "│ ID │ Play │\n",
        "├────┼──────┤\n",
        "│  1 │ Yes  │\n",
        "│  2 │ No   │\n",
        "└────┴──────┘\n",
    ]
    tagger = LineTagger()
    tags = [tagger.tag(line) for line in lines]
    assert tags == [
        "table_border",
        "table_header",
        "table_border",
        "table_row",
        "table_row",
        "table_border",
    ]


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
