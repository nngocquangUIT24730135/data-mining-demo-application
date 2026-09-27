"""CART console steps. Gini formulas stay in ``algorithms.cart_gini``."""
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable
from typing import Any

from datamining_app.algorithms._logger import StepLogger
from datamining_app.fmt import fmt_score, fmt_score_diff

class _CARTSteps:
    """Ghi bước CART vào StepLogger. gini_fn được truyền từ thuật toán."""
    def __init__(
        self,
        log: StepLogger,
        n_rows: int,
        class_counts: dict[str, int],
        gini_fn: Callable[[list[dict[str, str]], str], float],
    ) -> None:
        self._log = log
        self._gini = gini_fn
        self._split_explained = False
        self._growing: dict[str, Any] = {
            "is_leaf": False,
            "attribute": "S",
            "pending": True,
            "children": {},
            "samples": n_rows,
            "class_counts": class_counts,
            "path": "Gốc",
        }

    def initial_gini_step(
        self,
        rows: list[dict[str, str]],
        decision: str,
        gini_s: float,
        counts: Counter,
    ) -> None:
        class_rows = [[label, count] for label, count in counts.items()]
        n = len(rows)
        terms = " + ".join(f"({c}/{n})²" for c in counts.values())
        description = (
            "① Ý tưởng CART:\n"
            "   Chọn thuộc tính giảm Gini Impurity nhiều nhất ở mỗi nút.\n"
            "\n"
            "② Gini Index G(S):\n"
            "   G(S) = 1 - Σ pᵢ² = xác suất phân loại sai khi đoán ngẫu nhiên.\n"
            "   → = 0: tập thuần / ≈ 0.5: hỗn loạn tối đa (2 lớp đều nhau)\n"
            "\n"
            "③ Tính G(S) gốc:\n"
            f"   G(S) = 1 - ({terms}) = {fmt_score(gini_s)}"
        )
        self._log.add(
            "Tính Gini G(S) — nút Gốc",
            description,
            {
                "gini": gini_s,
                "class_counts": dict(counts),
                "rows": class_rows,
                "headers": ["Lớp", "Số mẫu"],
                "alignments": ["left", "right"],
                "decision": decision,
                "sample_rows": rows,
                "path": "Gốc",
                "tree": copy.deepcopy(self._growing),
                "conclusion": f"  → G(S) = {fmt_score(gini_s)}",
            },
            "STEP_HEADER",
        )

    def sub_dataset_step(
        self,
        path: str,
        rows: list[dict[str, str]],
        features: list[str],
        decision: str,
    ) -> None:
        counts = Counter(r[decision] for r in rows)
        n = len(rows)
        g_s = self._gini(rows, decision)
        branch = _branch_from_path(path)
        if branch:
            attr, value = branch
            title = f"Dataset for {attr} = '{value}' ({n} mẫu)"
            branch_label = f"{attr} = '{value}'"
        else:
            title = f"Tập dữ liệu con — {path} ({n} mẫu)"
            branch_label = path

        table_headers = ["ID", *features, f"{decision} (Nhãn)"]
        table_rows = [
            [r.get("_id", ""), *[r.get(f, "") for f in features], r.get(decision, "")]
            for r in rows
        ]
        alignments = ["right"] + ["left"] * len(features) + ["left"]

        description = (
            f"Tập dữ liệu con — Dataset for {branch_label}\n"
            f"  Đường dẫn: {path}\n"
            f"  Kích thước: {n} mẫu | Phân phối nhãn: {_label_dist(counts)}\n"
            f"  Thuộc tính còn lại: {features}"
        )
        if len(counts) == 1:
            label = next(iter(counts))
            conclusion = (
                f"  ① Điều kiện dừng — Tập thuần (Bước 2 mã giả):\n"
                f'    Tất cả mẫu có {decision} = "{label}" → G(S) = {fmt_score(g_s)}\n'
                f"    → Tạo nút lá ngay: {decision} = {label}."
            )
        else:
            conclusion = (
                f"  → Tập chưa thuần khiết (Gini G(S) = {fmt_score(g_s)})\n"
                f"  → [Mã giả Bước 4] Đánh giá các thuộc tính còn lại trên tập {n} mẫu này:"
            )

        self._log.add_table(
            title,
            table_headers,
            table_rows,
            alignments,
            description=description,
            level="STEP_HEADER",
            path=path,
            sub_dataset=True,
            conclusion=conclusion,
            tree=copy.deepcopy(self._growing),
        )

    def leaf_step(
        self,
        path: str,
        majority: str,
        counts: Counter,
        node: dict[str, Any],
        stop_reason: str = "pure",
        max_depth: int = 6,
    ) -> None:
        self._place_node(path, node)
        if stop_reason == "pure":
            title = f"Lá — {path} (tập thuần)"
            description = (
                "① Điều kiện dừng — Tập thuần (Bước 2 mã giả):\n"
                f"  Tất cả {sum(counts.values())} mẫu cùng lớp \"{majority}\" → G(S) = 0\n"
                f"  → Lá = {majority}"
            )
        elif stop_reason == "no_attrs":
            title = f"Lá — {path} (hết thuộc tính)"
            description = (
                "① Điều kiện dừng — Hết thuộc tính (Bước 3 mã giả):\n"
                "  Không còn thuộc tính để phân nhánh.\n"
                f"  → Lá = {majority} ({self._fmt_counts(counts)})"
            )
        else:
            title = f"Lá — {path} (đạt max_depth={max_depth})"
            description = (
                f"① Điều kiện dừng — Đạt max_depth={max_depth}:\n"
                "  Giới hạn độ sâu để tránh overfitting.\n"
                f"  → Lá = {majority} ({self._fmt_counts(counts)})"
            )
        self._log.add(
            title,
            description,
            {
                **node,
                "headers": ["Lớp", "Số mẫu"],
                "rows": [[label, count] for label, count in counts.items()],
                "alignments": ["left", "right"],
                "tree": copy.deepcopy(self._growing),
                "conclusion": f"  → Lá = {majority}",
            },
            "SUCCESS",
        )

    def split_step(
        self,
        path: str,
        rows: list[dict[str, str]],
        decision: str,
        gini_s: float,
        scores: list[tuple[str, float, float]],
        best: str,
        split_node: dict[str, Any],
    ) -> None:
        best_gini_a = next(gini_a for feat, _, gini_a in scores if feat == best)
        values = sorted({r[best] for r in rows})
        self._place_node(path, split_node)
        calc = self._format_scores(rows, decision, gini_s, scores, best)
        if not self._split_explained:
            self._split_explained = True
            description = (
                "[Mã giả Bước 4] VỚI mỗi A ∈ Attributes: tính ΔGini(A) = G(S) - G(A,S)\n"
                "[Mã giả Bước 5] A* ← argmax ΔGini(A)\n"
                "\n"
                "① ΔGini(A) = mức giảm Gini khi chia theo A.\n"
                "  → ΔGini càng lớn → A phân loại càng hiệu quả.\n"
                "\n"
                "② G(A,S) — Gini trung bình sau khi chia:\n"
                "  G(A,S) = Σᵥ (|Sᵥ|/|S|) × G(Sᵥ)\n"
                f"\n{calc}"
            )
        else:
            description = (
                f"[Mã giả Bước 4] Đánh giá ΔGini — nút {path}\n"
                f"[Mã giả Bước 5] Chọn A* có ΔGini lớn nhất\n"
                f"\n{calc}"
            )
        self._log.add(
            f"Chọn thuộc tính theo ΔGini — {path}",
            description,
            {
                "gains": [{"feature": f, "gain": d, "info": g} for f, d, g in scores],
                "chosen": best,
                "gini": gini_s,
                "path": path,
                "split_values": values,
                "headers": [
                    "Thuộc tính xem xét\nphân nhánh",
                    "Chỉ số Gini\ntrước khi chia G(S)",
                    "Chỉ số Gini trung bình\nsau khi chia G(A, S)",
                    "Mức giảm chỉ số Gini\nGini Gain (ΔG)",
                ],
                "rows": [
                    [
                        feat,
                        fmt_score(gini_s),
                        fmt_score(gini_a),
                        fmt_score_diff(gini_s, gini_a) + (" ★" if feat == best else ""),
                    ]
                    for feat, delta, gini_a in scores
                ],
                "alignments": ["left", "right", "right", "right"],
                "tree": copy.deepcopy(self._growing),
                "conclusion": (
                    f'  → [Mã giả Bước 5] Chọn A* = "{best}" '
                    f"(ΔGini = {fmt_score_diff(gini_s, best_gini_a)} ★)"
                ),
            },
            "SUCCESS",
        )

    def _place_node(self, path: str, node: dict[str, Any]) -> None:
        if path == "Gốc" or not path:
            self._growing.clear()
            self._growing.update(copy.deepcopy(node))
            return
        segments = path.split(" / ")[1:]
        current = self._growing
        for i, segment in enumerate(segments):
            attr, _, value = segment.partition("=")
            children = current.setdefault("children", {})
            if i == len(segments) - 1:
                children[value] = copy.deepcopy(node)
                return
            nxt = children.get(value)
            if nxt is None:
                nxt = {"is_leaf": False, "attribute": attr, "children": {}, "pending": True}
                children[value] = nxt
            current = nxt

    @staticmethod
    def _fmt_counts(counts: Counter) -> str:
        return ", ".join(f"{k}: {v}" for k, v in counts.items())

    def _format_scores(
        self,
        rows: list[dict[str, str]],
        decision: str,
        gini_s: float,
        scores: list[tuple[str, float, float]],
        best: str,
    ) -> str:
        n = len(rows)
        blocks: list[str] = []
        for feat, delta, gini_a in scores:
            blocks.append(
                _format_feature_gini_block(
                    rows, decision, gini_s, feat, delta, gini_a, n, starred=(feat == best),
                    gini_fn=self._gini,
                )
            )
        return "\n".join(blocks)


def _label_dist(counts: Counter) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + "}"


def _branch_from_path(path: str) -> tuple[str, str] | None:
    """Parse 'Gốc / Outlook=Sunny / Humidity=High' → ('Humidity', 'High')."""
    if " / " not in path:
        return None
    last = path.rsplit(" / ", 1)[-1]
    attr, sep, value = last.partition("=")
    if not sep:
        return None
    return attr.strip(), value.strip().strip("'\"")


def _format_gini_trace(
    subset: list[dict[str, str]],
    decision: str,
    parent_n: int,
    feature: str,
    value: str,
    gini_fn: Callable[[list[dict[str, str]], str], float],
) -> tuple[str, float]:
    """G(v): fraction formula → float64 result (no intermediate decimals)."""
    m = len(subset)
    counts = Counter(r[decision] for r in subset)
    g = gini_fn(subset, decision)
    lines = [
        f"  [ Nhánh: {feature} = {value} ] ({m}/{parent_n} mẫu)"
        f" → Phân phối nhãn: {_label_dist(counts)}"
    ]
    if m == 0:
        lines.append(f"    G({value}) = {fmt_score(0.0)}  (nhánh rỗng)")
        return "\n".join(lines), 0.0

    formula_parts = [f"({counts[label]}/{m})²" for label in sorted(counts)]
    lines.append(f"    G({value}) = 1 - {' - '.join(formula_parts)} = {fmt_score(g)}")
    return "\n".join(lines), g


def _format_feature_gini_block(
    rows: list[dict[str, str]],
    decision: str,
    gini_s: float,
    feat: str,
    delta: float,
    gini_a: float,
    n: int,
    *,
    gini_fn: Callable[[list[dict[str, str]], str], float],
    starred: bool = False,
) -> str:
    bar = "═" * max(8, 61 - len(feat))
    lines = [f"  ══ Thuộc tính: {feat} {bar}", ""]
    branch_vals: list[tuple[str, int, float]] = []
    for value in sorted({r[feat] for r in rows}):
        subset = [r for r in rows if r[feat] == value]
        block, g_v = _format_gini_trace(subset, decision, n, feat, value, gini_fn)
        lines.append(block)
        lines.append("")
        branch_vals.append((value, len(subset), g_v))

    # Fraction weights + G(v) symbols only — no rounded-decimal substitution.
    weighted_sym = " + ".join(f"({m}/{n}) × G({v})" for v, m, _ in branch_vals)
    g_as = fmt_score(gini_a)
    delta_txt = fmt_score_diff(gini_s, gini_a)
    star = "  ★" if starred else ""

    lines.append(f"  Chỉ số Gini trung bình sau khi chia theo {feat}:")
    lines.append(f"    G({feat}, S) = {weighted_sym} = {g_as}")
    lines.append("")
    lines.append("  Mức giảm chỉ số Gini thu được:")
    lines.append(f"    ΔGini({feat}) = G(S) - G({feat}, S) = {delta_txt}{star}")
    lines.append("")
    return "\n".join(lines)



__all__ = ["_CARTSteps"]
