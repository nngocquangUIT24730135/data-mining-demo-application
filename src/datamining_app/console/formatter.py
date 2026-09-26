from __future__ import annotations

import unicodedata
from typing import Any, Iterable, Sequence

from datamining_app.core.models import AlgorithmResult, StepLog
from datamining_app.fmt import SCORE_DECIMALS, fmt_pct, fmt_support

BOX = {
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "h": "─",
    "v": "│",
    "tm": "┬",
    "bm": "┴",
    "lm": "├",
    "rm": "┤",
    "x": "┼",
}


class ConsoleFormatter:
    def __init__(self, width: int = 80) -> None:
        self.width = width

    def banner(self, name: str, dataset_name: str, n_rows: int, params: dict[str, Any]) -> str:
        lines = [
            f"THUẬT TOÁN {name.upper()}",
            f"Dữ liệu : {dataset_name}  ({n_rows} dòng)",
        ]
        skip = {"exclude_cols", "condition_attrs"}
        for key, value in params.items():
            if key in skip:
                continue
            lines.append(f"{key} = {_fmt_param(key, value)}")
        inner_w = max(self.width - 4, max(display_width(s) for s in lines) + 2)
        top = "╔" + "═" * inner_w + "╗"
        bot = "╚" + "═" * inner_w + "╝"
        body = ["║  " + pad_display(s, inner_w - 2, "left") + "║" for s in lines]
        return "\n".join([top, *body, bot])

    def step_header(self, step: StepLog) -> str:
        # When title already encodes the algorithm step label (e.g. "Vòng lặp 1 — Bước 2a"),
        # skip the auto "Bước N:" prefix to avoid double numbering.
        if (step.data or {}).get("omit_step_number"):
            title = f" {step.title} "
        else:
            title = f" Bước {step.step_number}: {step.title} "
        fill = max(self.width - 2 - display_width(title), 4)
        return "──" + title + "─" * fill

    def render_step(self, step: StepLog) -> str:
        parts = [self.step_header(step), ""]
        data = step.data or {}
        description = (step.description or "").rstrip()
        table = self._table_from_step(step)
        show_rules = _should_render_rules(step)
        matrix_text = ""
        if step.kind == "matrix" or data.get("cells"):
            matrix_text = self.render_matrix(data)
        structured = bool(table) or show_rules or bool(matrix_text)
        conclusion_only = bool(description) and _conclusion_lines(description) == description
        if description and not _is_pipe_dump(description) and not (structured and conclusion_only):
            if not structured or not _is_duplicate_of_table(description, data):
                parts.append(description)
        if table:
            parts.append(table)
        if show_rules:
            parts.append(self.render_rules(data["rules"]))
        if matrix_text:
            parts.append(matrix_text)
        conclusion = data.get("conclusion")
        if conclusion:
            parts.append(str(conclusion).rstrip())
        elif structured:
            extra = _conclusion_lines(description)
            if extra and extra not in "\n".join(parts):
                parts.append(extra)
        return "\n".join(p for p in parts if p is not None).rstrip() + "\n"

    def render_table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
        alignments: Sequence[str] | None = None,
    ) -> str:
        if not headers:
            return ""
        str_rows = [[_cell(v) for v in row] for row in rows]

        # Multi-line headers: split on "\n", width = longest line per column.
        header_lines_per_col = [str(h).split("\n") for h in headers]
        max_header_lines = max(len(col_lines) for col_lines in header_lines_per_col)
        widths = [
            max(display_width(line) for line in col_lines)
            for col_lines in header_lines_per_col
        ]
        for row in str_rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], display_width(cell))
        widths = [w + COLUMN_PAD for w in widths]

        aligns = list(alignments or [])
        while len(aligns) < len(headers):
            aligns.append("left")

        h = BOX["h"]
        top = BOX["tl"] + BOX["tm"].join(h * (w + 2) for w in widths) + BOX["tr"]
        mid = BOX["lm"] + BOX["x"].join(h * (w + 2) for w in widths) + BOX["rm"]
        bot = BOX["bl"] + BOX["bm"].join(h * (w + 2) for w in widths) + BOX["br"]

        header_lines = []
        for line_idx in range(max_header_lines):
            line_parts = []
            for col_idx in range(len(headers)):
                col_lines = header_lines_per_col[col_idx]
                text = col_lines[line_idx] if line_idx < len(col_lines) else ""
                align = "center" if max_header_lines > 1 else aligns[col_idx]
                line_parts.append(pad_display(text, widths[col_idx], align, inner=True))
            header_lines.append(BOX["v"] + BOX["v"].join(line_parts) + BOX["v"])
        header = "\n".join(header_lines)

        body = [
            BOX["v"]
            + BOX["v"].join(
                pad_display(row[i] if i < len(row) else "", widths[i], aligns[i], inner=True)
                for i in range(len(headers))
            )
            + BOX["v"]
            for row in str_rows
        ]
        return "\n".join([top, header, mid, *body, bot])

    def render_rules(self, rules: Iterable[dict[str, Any]]) -> str:
        rule_list = list(rules)
        if not rule_list:
            return "  Không có luật hợp lệ."
        rows = []
        for i, rule in enumerate(rule_list, start=1):
            lhs = _itemset(rule.get("antecedent") or rule.get("lhs") or [])
            rhs = _itemset(rule.get("consequent") or rule.get("rhs") or [])
            sp = rule.get("support", 0)
            conf = rule.get("confidence", 0)
            rows.append([str(i), f"{lhs} → {rhs}", fmt_support(float(sp)), fmt_support(float(conf))])
        return self.render_table(
            ["#", "Luật", "Support", "Confidence"],
            rows,
            ["right", "left", "right", "right"],
        )

    def render_matrix(self, data: dict[str, Any]) -> str:
        cells = data.get("cells") or []
        universe = data.get("universe") or []
        if not cells:
            return ""
        rows = []
        for cell in cells:
            attrs = ", ".join(cell.get("attrs") or [])
            rows.append([str(cell.get("i")), str(cell.get("j")), f"{{{attrs}}}"])
        table = self.render_table(["u", "v", "M(u,v)"], rows, ["left", "left", "left"])
        extra = f"  |U| = {len(universe)}" if universe else ""
        return table + (("\n" + extra) if extra else "")

    def render_prediction(self, label: str, explanation: str) -> str:
        box_inner = f"  Kết quả: {label}  "
        width = max(display_width(box_inner), 28)
        top = "┌" + "─" * width + "┐"
        mid = "│" + pad_display(box_inner, width, "left") + "│"
        bot = "└" + "─" * width + "┘"
        header = self.step_header(
            StepLog(step_number=0, title="Dự đoán mẫu mới", description="", kind="prediction")
        ).replace(" Bước 0: ", " ")
        return "\n".join([header, "", explanation, "", top, mid, bot, ""])

    def render_pseudocode(self, algo_key: str) -> str:
        from datamining_app.algorithms.pseudocode import PSEUDOCODE

        text = PSEUDOCODE.get(algo_key, "")
        if not text:
            return ""
        lines = text.strip().splitlines()
        content_w = max(display_width(line) for line in lines)
        inner_w = max(self.width - 2, content_w + 4)
        label = " MÃ GIẢ "
        label_w = display_width(label)
        dash_total = max(inner_w - label_w, 2)
        left = dash_total // 2
        right = dash_total - left
        top = "┌" + "─" * left + label + "─" * right + "┐"
        bot = "└" + "─" * inner_w + "┘"
        body = ["│" + pad_display("  " + line, inner_w, "left") + "│" for line in lines]
        return "\n".join([top, *body, bot])

    def render_result(self, result: AlgorithmResult, dataset_name: str = "", n_rows: int = 0) -> str:
        chunks = [
            self.banner(result.algorithm_name, dataset_name or "", n_rows, result.parameters),
            "",
        ]
        pseudo = self.render_pseudocode(_pseudo_key(result.algorithm_name))
        if pseudo:
            chunks.extend([pseudo, ""])
        for step in result.steps:
            chunks.append(self.render_step(step))
            chunks.append("")
        if result.summary:
            chunks.append(self._summary_box(result.summary))
        if result.predictions:
            chunks.append("")
            for pred in result.predictions:
                chunks.append(self.render_prediction(str(pred.get("label")), str(pred.get("explanation", ""))))
        return "\n".join(chunks).rstrip() + "\n"

    def _summary_box(self, summary: str) -> str:
        lines = [line.strip() for line in summary.replace(". ", ".\n").splitlines() if line.strip()]
        if not lines:
            lines = [summary]
        body = ["  → " + line for line in lines]
        return "\n".join(body)

    def _table_from_step(self, step: StepLog) -> str:
        data = step.data or {}
        if data.get("headers") and data.get("rows") is not None:
            return self.render_table(data["headers"], data["rows"], data.get("alignments"))
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) and "itemset" in candidates[0]:
            has_vector = any("vector" in row for row in candidates)
            rows = []
            for row in candidates:
                verdict = "✓ Chấp nhận" if row.get("accepted") else "✗ Loại bỏ"
                item = _itemset(row.get("itemset") or [])
                sp = fmt_support(float(row.get("support", 0)))
                if has_vector:
                    vec = "(" + ", ".join(str(b) for b in (row.get("vector") or [])) + ")"
                    if row.get("left") and row.get("right"):
                        op = f"{_itemset(row['left'])} ∧ {_itemset(row['right'])}"
                    else:
                        op = "vector 1-item"
                    rows.append([item, op, vec, row.get("count", ""), sp, verdict])
                else:
                    rows.append([item, row.get("count", ""), sp, verdict])
            if has_vector:
                return self.render_table(
                    ["Tập mặt hàng", "Phép AND", "Vector", "Count", "SP", "Kết luận"],
                    rows,
                    ["left", "left", "left", "right", "right", "left"],
                )
            return self.render_table(
                ["Tập mặt hàng", "Count", "SP", "Kết luận"],
                rows,
                ["left", "right", "right", "left"],
            )
        return ""


COLUMN_PAD = 5  # extra spaces in every cell so box borders stay aligned in Consolas
WIDE_GLYPHS = set("✓✗★●◆■□▲▼○△▷∧√")


def display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        if char in WIDE_GLYPHS:
            width += 2
            continue
        east = unicodedata.east_asian_width(char)
        width += 2 if east in {"W", "F"} else 1
    return width


def pad_display(text: str, width: int, align: str, inner: bool = False) -> str:
    text = _cell(text)
    gap = max(width - display_width(text), 0)
    if align == "right":
        padded = (" " * gap) + text
    elif align == "center":
        left = gap // 2
        padded = (" " * left) + text + (" " * (gap - left))
    else:
        padded = text + (" " * gap)
    if inner:
        return " " + padded + " "
    return padded


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _itemset(items: Iterable[str]) -> str:
    seq = list(items)
    return "{" + ", ".join(str(x) for x in seq) + "}"


def _should_render_rules(step: StepLog) -> bool:
    data = step.data or {}
    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        return False
    first = rules[0]
    return isinstance(first, dict) and ("antecedent" in first or "lhs" in first)


def _is_pipe_dump(description: str) -> bool:
    return "|" in description and any(token in description for token in ("Count", "SP", "Vector", "Tập ứng viên"))


def _is_duplicate_of_table(description: str, data: dict[str, Any]) -> bool:
    if data.get("rules") and ("minconf" in description or "⇒" in description or "→" in description):
        if description.strip().startswith(("Các luật", "Không có luật")):
            return True
    return False


def _conclusion_lines(description: str) -> str:
    if not description:
        return ""
    kept = [
        line
        for line in description.splitlines()
        if line.strip().startswith(("⇒", "→", "=>"))
        or line.strip().startswith(("L₁", "L_", "F₁", "F_"))
    ]
    return "\n".join(kept)


def _fmt_param(key: str, value: Any) -> str:
    if key in {"minsup", "minconf"} and isinstance(value, (int, float)):
        return fmt_pct(float(value))
    if isinstance(value, float):
        return f"{value:.{SCORE_DECIMALS}f}".rstrip("0").rstrip(".")
    return str(value)


def _pseudo_key(algorithm_name: str) -> str:
    """Map AlgorithmResult.algorithm_name → PSEUDOCODE dict key."""
    # NFD + strip combining marks so Vietnamese (ổ, ể, …) → ASCII base letters.
    decomposed = unicodedata.normalize("NFD", algorithm_name.lower())
    ascii_ish = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    raw = (
        ascii_ish.replace("đ", "d")
        .replace(" ", "_")
        .replace("-", "_")
    )
    aliases = {
        "k_means": "kmeans",
        "kmeans": "kmeans",
        "apriori": "apriori",
        "binary_vector": "binary_vector",
        "rough_set": "rough_set",
        "id3": "id3",
        "cart_gini": "cart_gini",
        "cart_(gini_index)": "cart_gini",
        "naive_bayes": "naive_bayes",
        "naive_bayes_(co_dien)": "naive_bayes",
        "naive_bayes_(classic)": "naive_bayes",
        "naive_bayes_(laplace)": "naive_bayes_laplace",
        "naive_bayes_laplace": "naive_bayes_laplace",
    }
    return aliases.get(raw, raw)
