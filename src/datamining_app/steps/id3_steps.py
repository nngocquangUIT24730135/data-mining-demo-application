"""ID3 console steps. Entropy and information gain stay in ``algorithms.id3``."""
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable
from typing import Any

from datamining_app.algorithms._logger import StepLogger
from datamining_app.fmt import fmt_score, fmt_score_diff

class _ID3Steps:
    """Ghi bước ID3 vào StepLogger. entropy_fn được truyền từ thuật toán, không tính lại công thức ở đây ngoài phần diễn giải."""
    def __init__(
        self,
        log: StepLogger,
        n_rows: int,
        class_counts: dict[str, int],
        entropy_fn: Callable[[list[dict[str, str]], str], float],
    ) -> None:
        self._log = log
        self._entropy = entropy_fn
        self._gain_explained = False
        self._growing: dict[str, Any] = {
            "is_leaf": False,
            "attribute": "S",
            "pending": True,
            "children": {},
            "samples": n_rows,
            "class_counts": class_counts,
            "path": "Gốc",
        }

    def initial_entropy_step(
        self,
        rows: list[dict[str, str]],
        decision: str,
        info_d: float,
        counts: Counter,
    ) -> None:
        class_rows = [[label, count] for label, count in counts.items()]
        n = len(rows)
        dist = ", ".join(f"{v} {k}" for k, v in counts.items())
        terms = " - ".join(f"({c}/{n})×log₂({c}/{n})" for c in counts.values())
        description = (
            "① Ý tưởng cốt lõi của ID3:\n"
            "  Xây cây quyết định bằng cách liên tục chọn thuộc tính\n"
            '  "phân loại tốt nhất" (Information Gain cao nhất) làm\n'
            "  nút phân nhánh, cho đến khi tất cả lá thuần hoặc hết tài nguyên.\n"
            "\n"
            "② Entropy I(S) — đo độ hỗn loạn (tạp chất) của tập dữ liệu:\n"
            "  I(S) = -Σ pᵢ × log₂(pᵢ)\n"
            "  → = 0 bit: tập thuần (chỉ 1 lớp) — lý tưởng nhất\n"
            "  → = 1 bit: tập 50/50 (2 lớp đều nhau) — hỗn loạn nhất\n"
            "  → Càng cao → càng cần phân loại thêm.\n"
            "\n"
            f"  |S| = {n} mẫu → {dist}\n"
            f"  I(S) = -{terms} = {fmt_score(info_d)} bit"
        )
        self._log.add(
            "Tính Entropy I(S) — nút Gốc",
            description,
            {
                "entropy": info_d,
                "class_counts": dict(counts),
                "rows": class_rows,
                "headers": ["Lớp", "Số mẫu"],
                "alignments": ["left", "right"],
                "decision": decision,
                "sample_rows": rows,
                "path": "Gốc",
                "tree": copy.deepcopy(self._growing),
                "conclusion": f"  → I(S) = {fmt_score(info_d)} bit",
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
        info_s = self._entropy(rows, decision)
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
                f'    Tất cả mẫu có {decision} = "{label}" → I(S) = {fmt_score(info_s)}\n'
                f"    → Tạo nút lá ngay: {decision} = {label}."
            )
        else:
            conclusion = (
                f"  → Tập chưa thuần khiết (Entropy I(S) = {fmt_score(info_s)} bit)\n"
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
                f"  Tất cả {sum(counts.values())} mẫu đều cùng lớp \"{majority}\" → I(S) = 0\n"
                "  → Gán lá ngay, không cần tính thêm.\n"
                f"  → Lá = {majority}"
            )
        elif stop_reason == "no_attrs":
            title = f"Lá — {path} (hết thuộc tính)"
            description = (
                "① Điều kiện dừng — Hết thuộc tính (Bước 3 mã giả):\n"
                "  Không còn thuộc tính để phân nhánh tiếp.\n"
                "  → Gán lớp đa số trong tập hiện tại.\n"
                f"  → Lá = {majority} ({self._fmt_counts(counts)})"
            )
        else:
            title = f"Lá — {path} (đạt max_depth={max_depth})"
            description = (
                "① Điều kiện dừng — Giới hạn độ sâu (ràng buộc kỹ thuật):\n"
                f"  Độ sâu đã đạt max_depth={max_depth} → dừng để tránh overfitting.\n"
                "  → Gán lớp đa số.\n"
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

    def gain_step(
        self,
        path: str,
        rows: list[dict[str, str]],
        decision: str,
        info_d: float,
        gains: list[tuple[str, float, float]],
        best: str,
        split_node: dict[str, Any],
    ) -> None:
        best_split = next(split for feat, _, split in gains if feat == best)
        values = sorted({r[best] for r in rows})
        self._place_node(path, split_node)
        calc = self._format_gains(rows, decision, info_d, gains, best)
        if not self._gain_explained:
            self._gain_explained = True
            description = (
                "[Mã giả Bước 4] VỚI mỗi A ∈ Attributes: tính Gain(A,S) = I(S) - E(A,S)\n"
                "[Mã giả Bước 5] A* ← argmax Gain(A,S)\n"
                "\n"
                "① Information Gain là gì?\n"
                "  Gain(A) = mức giảm Entropy khi biết thêm thuộc tính A.\n"
                "  → Gain càng cao → A phân loại càng hiệu quả.\n"
                "\n"
                "② E(A,S) — Entropy trung bình sau khi chia theo A:\n"
                "  E(A,S) = Σᵥ (|Sᵥ|/|S|) × I(Sᵥ)\n"
                "  → Trung bình Entropy của các tập con sau khi chia theo A.\n"
                f"\n{calc}"
            )
        else:
            description = (
                f"[Mã giả Bước 4] Đánh giá Gain — nút {path}\n"
                f"[Mã giả Bước 5] Chọn A* có Gain lớn nhất\n"
                f"\n{calc}"
            )
        self._log.add(
            f"Chọn thuộc tính phân nhánh — {path}",
            description,
            {
                "gains": [{"feature": f, "gain": g, "info": s} for f, g, s in gains],
                "chosen": best,
                "entropy": info_d,
                "path": path,
                "split_values": values,
                "headers": [
                    "Thuộc tính xem xét\nphân nhánh",
                    "Entropy tập dữ liệu\ntrước khi chia I(S)",
                    "Entropy trung bình\nsau khi chia E(A, S)",
                    "Mức tăng thông tin\nInformation Gain",
                ],
                "rows": [
                    [
                        feat,
                        fmt_score(info_d),
                        fmt_score(split),
                        fmt_score_diff(info_d, split) + (" ★" if feat == best else ""),
                    ]
                    for feat, gain, split in gains
                ],
                "alignments": ["left", "right", "right", "right"],
                "tree": copy.deepcopy(self._growing),
                "conclusion": (
                    f'  → [Mã giả Bước 5] Chọn A* = "{best}" làm nút '
                    f"(Gain = {fmt_score_diff(info_d, best_split)} ★)"
                ),
            },
            "SUCCESS",
        )

    def _place_node(self, path: str, node: dict[str, Any]) -> None:
        """Insert/replace a node on the growing visualization tree (DFS snapshots)."""
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

    def _format_gains(
        self,
        rows: list[dict[str, str]],
        decision: str,
        info_d: float,
        gains: list[tuple[str, float, float]],
        best: str,
    ) -> str:
        n = len(rows)
        blocks: list[str] = []
        for feat, gain, split in gains:
            blocks.append(
                _format_feature_entropy_block(
                    rows, decision, info_d, feat, gain, split, n, starred=(feat == best),
                    entropy_fn=self._entropy,
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


def _format_entropy_trace(
    subset: list[dict[str, str]],
    decision: str,
    parent_n: int,
    feature: str,
    value: str,
    entropy_fn: Callable[[list[dict[str, str]], str], float],
) -> tuple[str, float]:
    """I(v): fraction formula → float64 result (no intermediate decimals)."""
    m = len(subset)
    counts = Counter(r[decision] for r in subset)
    e = entropy_fn(subset, decision)
    lines = [
        f"  [ Nhánh: {feature} = {value} ] ({m}/{parent_n} mẫu)"
        f" → Phân phối nhãn: {_label_dist(counts)}"
    ]
    if m == 0:
        lines.append(f"    I({value}) = {fmt_score(0.0)}  (nhánh rỗng)")
        return "\n".join(lines), 0.0

    formula_parts = [f"({counts[label]}/{m}) × log₂({counts[label]}/{m})" for label in sorted(counts)]
    lines.append(f"    I({value}) = -{' - '.join(formula_parts)} = {fmt_score(e)}")
    return "\n".join(lines), e


def _format_feature_entropy_block(
    rows: list[dict[str, str]],
    decision: str,
    info_d: float,
    feat: str,
    gain: float,
    split: float,
    n: int,
    *,
    entropy_fn: Callable[[list[dict[str, str]], str], float],
    starred: bool = False,
) -> str:
    bar = "═" * max(8, 61 - len(feat))
    lines = [f"  ══ Thuộc tính: {feat} {bar}", ""]
    branch_vals: list[tuple[str, int, float]] = []
    for value in sorted({r[feat] for r in rows}):
        subset = [r for r in rows if r[feat] == value]
        block, e_v = _format_entropy_trace(subset, decision, n, feat, value, entropy_fn)
        lines.append(block)
        lines.append("")
        branch_vals.append((value, len(subset), e_v))

    # Fraction weights + I(v) symbols only — no rounded-decimal substitution.
    weighted_sym = " + ".join(f"({m}/{n}) × I({v})" for v, m, _ in branch_vals)
    e_as = fmt_score(split)
    gain_txt = fmt_score_diff(info_d, split)
    star = "  ★" if starred else ""

    lines.append(f"  Entropy trung bình sau khi chia theo {feat}:")
    lines.append(f"    E({feat}, S) = {weighted_sym} = {e_as}")
    lines.append("")
    lines.append("  Mức tăng thông tin thu được:")
    lines.append(f"    Gain({feat}) = I(S) - E({feat}, S) = {gain_txt}{star}")
    lines.append("")
    return "\n".join(lines)



__all__ = ["_ID3Steps"]
