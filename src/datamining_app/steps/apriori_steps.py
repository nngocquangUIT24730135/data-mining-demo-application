"""Apriori console steps. Itemset mining stays in ``algorithms.apriori``."""
from __future__ import annotations

from itertools import combinations
from typing import Any

from datamining_app.algorithms._logger import StepLogger
from datamining_app.algorithms.dataset_utils import format_itemset
from datamining_app.fmt import fmt_pct

class _AprioriSteps:
    """Ghi bước Apriori vào StepLogger từ ứng viên, tập phổ biến và luật đã tính."""
    def __init__(self, log: StepLogger) -> None:
        self._log = log
        self._min_count = 0
        self._n = 0
        self._join_explained = False
        self._filter_explained = False

    def init_step(self, n: int, minsup: float, min_count: int, minconf: float) -> None:
        self._min_count = min_count
        self._n = n
        self._log.add(
            "Khởi tạo",
            (
                "① Ý tưởng cốt lõi của Apriori:\n"
                "  Tìm các tập mặt hàng xuất hiện đủ thường xuyên trong\n"
                "  các giao dịch (tập phổ biến), rồi sinh luật kết hợp.\n"
                '  Ví dụ: "70% khách mua {Bánh mì, Sữa} → cũng mua {Bơ}".\n'
                "\n"
                "② Độ hỗ trợ (Support):\n"
                "  support({X}) = số giao dịch chứa X / tổng giao dịch\n"
                "  → Đo mức độ phổ biến. minsup = ngưỡng tối thiểu.\n"
                f"  min_count = ⌈minsup × N⌉ = ⌈{minsup} × {n}⌉ = {min_count}\n"
                "\n"
                "③ Tính chất anti-monotone (nền tảng của Apriori):\n"
                "  Nếu {X} không phổ biến → mọi tập cha {X,Y} cũng không phổ biến.\n"
                "  → Không cần kiểm tra {X,Y} nếu {X} đã bị loại (cắt tỉa).\n"
                "\n"
                "④ Độ tin cậy (Confidence) — dùng khi sinh luật:\n"
                "  conf(X ⇒ Y) = support(X∪Y) / support(X)\n"
                "  → Nếu biết X thì Y xuất hiện với xác suất bao nhiêu?\n"
                f"\nN = {n} giao dịch | minsup = {fmt_pct(minsup)} | minconf = {fmt_pct(minconf)}"
            ),
            {"n": n, "minsup": minsup, "min_count": min_count, "minconf": minconf},
            "STEP_HEADER",
        )

    def level_step(self, k: int, c_rows: list[dict[str, Any]], l_k: list[tuple[str, ...]]) -> None:
        if k == 1:
            title = "Cấp k = 1 — C₁ và L₁"
            conclusion = f"  → L₁ = {self._format_level(l_k)}"
            level = "SUCCESS"
            description = (
                "[Mã giả Bước 1–2] C₁ ← tất cả item; L₁ ← {c ∈ C₁ : support ≥ minsup}\n"
                f"① Với mỗi ứng viên: đếm giao dịch chứa nó; giữ nếu count ≥ {self._min_count}."
            )
        else:
            title = f"Cấp k = {k} — Đếm support → Lọc L_{k}"
            conclusion = f"  → L_{k} = {self._format_level(l_k)}"
            level = "SUCCESS" if l_k else "WARNING"
            if not self._filter_explained:
                self._filter_explained = True
                description = (
                    f"[Mã giả Bước 4] đếm support(c) trên D; L_{k} ← {{c : support ≥ minsup}}\n"
                    "\n"
                    f"① Với mỗi ứng viên trong C_{k}: quét toàn bộ D,\n"
                    "  đếm số giao dịch chứa ứng viên đó.\n"
                    f"  → Giữ lại nếu count ≥ min_count = {self._min_count}."
                )
            else:
                description = (
                    f"[Mã giả Bước 4] đếm support → L_{k} "
                    f"(ngưỡng min_count = {self._min_count})"
                )
        self._log.add(
            title,
            description,
            {
                "k": k,
                "n": self._n,
                "candidates": c_rows,
                "frequent": [list(x) for x in l_k],
                "conclusion": conclusion,
            },
            level,
        )

    def join_prune_step(
        self,
        k: int,
        prev: list[tuple[str, ...]],
        ck: list[tuple[str, ...]],
        pruned: list[tuple[str, ...]],
    ) -> None:
        prev_set = {tuple(sorted(x)) for x in prev}
        join_rows = [[format_itemset(x), "✓ Giữ"] for x in ck]
        for itemset in pruned:
            missing = _missing_subsets(itemset, prev_set, k)
            names = ", ".join(format_itemset(s) for s in missing)
            join_rows.append([format_itemset(itemset), f"✗ Cắt tỉa — thiếu {names} trong L_{k - 1}"])
        data: dict[str, Any] = {
            "k": k,
            "join_candidates": [list(x) for x in ck],
            "pruned": [list(x) for x in pruned],
        }
        if join_rows:
            data.update(
                {
                    "headers": ["Ứng viên Cₖ", "Kết luận"],
                    "rows": join_rows,
                    "alignments": ["left", "left"],
                }
            )
        if not self._join_explained:
            self._join_explained = True
            description = (
                f"[Mã giả Bước 4] Cₖ ← apriori_gen(L_{{k-1}})\n"
                "\n"
                "① Join (Ghép):\n"
                "  Ghép 2 tập (k-1)-phần tử có cùng (k-2) phần tử đầu.\n"
                "  Ví dụ: {A,B} và {A,C} → ghép → {A,B,C}\n"
                "\n"
                "② Prune (Cắt tỉa):\n"
                "  Loại ứng viên nếu bất kỳ tập con (k-1)-phần tử\n"
                "  nào của nó chưa có trong L_{k-1}.\n"
                "  → Áp dụng tính chất anti-monotone.\n"
                f"\nL_{k-1} = {self._format_level(prev)}"
            )
        else:
            description = self._describe_join_prune(prev, k)
        self._log.add(f"Cấp k = {k} — Sinh ứng viên C_{k} — Join & Prune", description, data)

    def stop_step(
        self,
        k: int,
        prev: list[tuple[str, ...]],
        pruned: list[tuple[str, ...]],
    ) -> None:
        self._log.add(
            f"Cấp k = {k} — Dừng vì C_{k} = {{}}",
            self._explain_empty_candidates(prev, k, pruned),
            {
                "k": k,
                "conclusion": f"  → L_{k - 1} ≠ {{}} nhưng C_{k} = {{}}, nên dừng tại k = {k}.",
            },
            "WARNING",
        )

    @staticmethod
    def _explain_empty_candidates(
        prev: list[tuple[str, ...]],
        k: int,
        pruned: list[tuple[str, ...]],
    ) -> str:
        ordered = sorted(tuple(sorted(itemset)) for itemset in prev)
        prefix_len = k - 2
        lines = [
            f"L_{k - 1} vẫn còn {len(ordered)} tập phổ biến, nhưng C_{k} rỗng.",
            f"Dừng không phải vì L_{k - 1} hết tập, mà vì phép Join không tạo được ứng viên.",
            "",
            f"① Muốn có một tập {k} phần tử, phải ghép hai tập trong L_{k - 1}.",
        ]
        if prefix_len <= 0:
            lines.append("   Với k = 2, hai item bất kỳ đều ghép được với nhau.")
            lines.append(f"   L_1 chỉ có {len(ordered)} item, chưa đủ một cặp.")
            lines.append("")
            lines.append(f"→ C_{k} = {{}}, không còn gì để đếm support. L_{k} = {{}}. Thuật toán dừng.")
            return "\n".join(lines)

        head = "phần tử đầu" if prefix_len == 1 else f"{prefix_len} phần tử đầu"
        lines.extend(
            [
                f"   Hai tập chỉ ghép được khi trùng nhau ở {head}",
                "   (đã sắp xếp theo thứ tự chữ cái), còn phần tử cuối thì khác nhau.",
            ]
        )
        if prefix_len == 1:
            lines.extend(
                [
                    "   Ví dụ: {A, B} và {A, C} trùng A → ghép thành {A, B, C}.",
                    "   {A, B} và {C, D} có phần tử đầu A ≠ C → không ghép.",
                ]
            )
        else:
            lines.extend(
                [
                    "   Ví dụ: {A, B, C} và {A, B, D} trùng {A, B} → ghép thành {A, B, C, D}.",
                    "   {A, B, C} và {A, C, D} có tiền tố {A, B} ≠ {A, C} → không ghép.",
                ]
            )
        lines.extend(["", f"② Tiền tố của từng tập trong L_{k - 1}:"])
        groups: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        for itemset in ordered:
            prefix = itemset[:prefix_len]
            groups.setdefault(prefix, []).append(itemset)
            lines.append(
                f"   {format_itemset(itemset)}"
                f"  →  {head} = {format_itemset(prefix)}, phần tử cuối = {itemset[-1]}"
            )
        lines.append("")
        joinable = [sets for sets in groups.values() if len(sets) >= 2]
        if len(ordered) < 2:
            lines.append(f"③ L_{k - 1} chỉ có một tập. Cần ít nhất hai tập khác nhau mới ghép được.")
        elif not joinable:
            lines.append(f"③ Không có hai tập nào cùng {head}.")
            lines.append("   Phép Join không tạo ra cặp nào, nên tập ứng viên rỗng.")
        else:
            lines.append("③ Có cặp trùng tiền tố, nhưng mỗi ứng viên sau khi ghép đều bị cắt tỉa")
            lines.append(f"   vì thiếu ít nhất một tập con {k - 1} phần tử trong L_{k - 1}:")
            prev_set = set(ordered)
            for itemset in pruned:
                missing = ", ".join(format_itemset(s) for s in _missing_subsets(itemset, prev_set, k))
                lines.append(f"   {format_itemset(itemset)} thiếu {missing}.")
            lines.append("   Tính chất anti-monotone: thiếu một tập con phổ biến thì tập cha không phổ biến.")
        lines.append("")
        lines.append(f"→ C_{k} = {{}}, không còn gì để đếm support. L_{k} = {{}}. Thuật toán dừng.")
        return "\n".join(lines)

    def summary_step(self, freq_rows: list[list[Any]], maximal: list[tuple[str, ...]]) -> None:
        n_freq = len(freq_rows)
        self._log.add_table(
            "Tổng hợp tập phổ biến",
            ["Cấp k", "Tập phổ biến", "Count", "Support"],
            freq_rows,
            ["right", "left", "right", "right"],
            description=(
                "[Mã giả Bước 5] L ← ∪ₖ Lₖ\n"
                "① Tập phổ biến tối đại: không bị bao bởi tập phổ biến nào lớn hơn."
            ),
            level="SUCCESS",
            frequent_rows=freq_rows,
            conclusion=(
                f"  → {n_freq} tập phổ biến. "
                f"Tối đại: {', '.join(format_itemset(x) for x in maximal) or '{}'}."
            ),
        )

    def rules_step(self, rules: list[dict[str, Any]], minconf: float) -> None:
        self._log.add(
            "Sinh luật kết hợp",
            (
                "[Mã giả Bước 6] Sinh luật X ⇒ Y với conf ≥ minconf\n"
                "\n"
                "① Luật kết hợp là gì?\n"
                "  X ⇒ Y nghĩa là: nếu mua X thì thường cũng mua Y.\n"
                "  conf(X ⇒ Y) = support(X∪Y) / support(X)."
            ),
            {
                "rules": rules,
                "conclusion": (
                    f"  → {len(rules)} luật đạt minconf = {fmt_pct(minconf)}."
                    if rules
                    else f"  → Không có luật nào đạt minconf = {fmt_pct(minconf)}."
                ),
            },
            "SUCCESS",
        )

    @staticmethod
    def _format_level(level: list[tuple[str, ...]]) -> str:
        if not level:
            return "{}"
        return "{" + ", ".join(format_itemset(x) for x in level) + "}"

    @classmethod
    def _describe_join_prune(cls, prev: list[tuple[str, ...]], k: int) -> str:
        return (
            f"[Mã giả Bước 4] Join L_{k-1} ⋈ L_{k-1} rồi Prune.\n"
            f"L_{k-1} = {cls._format_level(prev)}"
        )


def _missing_subsets(
    candidate: tuple[str, ...],
    prev_set: set[tuple[str, ...]],
    k: int,
) -> list[tuple[str, ...]]:
    ordered = tuple(sorted(candidate))
    return [subset for subset in combinations(ordered, k - 1) if subset not in prev_set]



__all__ = ["_AprioriSteps"]
