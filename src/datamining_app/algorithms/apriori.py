from __future__ import annotations

from itertools import combinations
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm, StepLogger
from datamining_app.algorithms.dataset_utils import (
    extract_transactions,
    format_itemset,
    minsup_count,
    support_count,
)
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef
from datamining_app.fmt import fmt_pct, fmt_support


class AprioriAlgorithm(BaseAlgorithm):
    name = "Apriori"
    description = "Khai phá tập phổ biến theo Join & Prune, sau đó sinh luật kết hợp."
    supports_predict = False
    param_schema = {
        "minsup": ParamDef(
            type="float",
            default=0.5,
            min=0.05,
            max=1.0,
            step=0.05,
            label_vi="Độ hỗ trợ tối thiểu (minsup)",
            label_en="Minimum support (minsup)",
        ),
        "minconf": ParamDef(
            type="float",
            default=0.75,
            min=0.05,
            max=1.0,
            step=0.05,
            label_vi="Độ tin cậy tối thiểu (minconf)",
            label_en="Minimum confidence (minconf)",
        ),
    }

    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult:
        minsup = float(params.get("minsup", 0.5))
        minconf = float(params.get("minconf", 0.75))
        raw_txns = extract_transactions(dataset, params)
        transactions = [set(t) for t in raw_txns]
        n = len(transactions)
        min_count = minsup_count(minsup, n)

        log = self._new_logger()
        steps = _AprioriSteps(log)
        steps.init_step(n, minsup, min_count, minconf)

        items = sorted({item for txn in transactions for item in txn})
        c1 = [(item,) for item in items]
        l1, c1_rows = _filter_frequent(c1, transactions, n, min_count, minsup)
        steps.level_step(1, c1_rows, l1)

        levels: dict[int, list[tuple[str, ...]]] = {1: l1}
        k = 2
        prev = l1
        while prev:
            ck, pruned = apriori_gen(prev)
            steps.join_prune_step(k, prev, ck, pruned)
            if not ck:
                steps.stop_step(k)
                break
            lk, ck_rows = _filter_frequent(ck, transactions, n, min_count, minsup)
            steps.level_step(k, ck_rows, lk)
            if not lk:
                break
            levels[k] = lk
            prev = lk
            k += 1

        all_frequent = {itemset: support_count(itemset, transactions) / n
                        for level in levels.values() for itemset in level}
        maximal = _maximal_itemsets(levels)
        freq_rows = []
        for level_k, itemsets in levels.items():
            for itemset in itemsets:
                sp = all_frequent[itemset]
                freq_rows.append(
                    [str(level_k), format_itemset(itemset), support_count(itemset, transactions), fmt_support(sp)]
                )
        n_freq = len(freq_rows)
        steps.summary_step(freq_rows, maximal)
        rules = generate_association_rules(all_frequent, minconf)
        steps.rules_step(rules, minconf)

        summary = (
            f"Apriori tìm {n_freq} tập phổ biến "
            f"({len(rules)} luật đạt minconf = {fmt_pct(minconf)}). "
            f"Tập phổ biến tối đại: {', '.join(format_itemset(x) for x in maximal) or '∅'}."
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={"minsup": minsup, "minconf": minconf},
            steps=log.steps,
            output={
                "levels": {str(k): [list(s) for s in v] for k, v in levels.items()},
                "supports": {format_itemset(s): sp for s, sp in all_frequent.items()},
                "rules": rules,
                "maximal": [list(s) for s in maximal],
                "transactions": [sorted(t) for t in transactions],
            },
            summary=summary,
        )
        return self._finish(result)


class _AprioriSteps:
    def __init__(self, log: StepLogger) -> None:
        self._log = log
        self._min_count = 0
        self._join_explained = False
        self._filter_explained = False

    def init_step(self, n: int, minsup: float, min_count: int, minconf: float) -> None:
        self._min_count = min_count
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
        join_rows = [[format_itemset(x), "✓ Giữ"] for x in ck]
        join_rows.extend([[format_itemset(x), "✗ Cắt tỉa (thiếu tập con k−1)"] for x in pruned])
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

    def stop_step(self, k: int) -> None:
        self._log.add(
            f"Cấp k = {k} — Dừng",
            (
                f"[Mã giả Bước 4] LẶP khi L_{{k-1}} ≠ ∅ — nhưng C_{k} = ∅.\n"
                f"→ Không còn ứng viên → L_{k} = ∅. Thuật toán dừng."
            ),
            {"k": k, "conclusion": f"  → Dừng tại k = {k}."},
            "WARNING",
        )

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
                f"Tối đại: {', '.join(format_itemset(x) for x in maximal) or '∅'}."
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
            return "∅"
        return "{" + ", ".join(format_itemset(x) for x in level) + "}"

    @classmethod
    def _describe_join_prune(cls, prev: list[tuple[str, ...]], k: int) -> str:
        return (
            f"[Mã giả Bước 4] Join L_{k-1} ⋈ L_{k-1} rồi Prune.\n"
            f"L_{k-1} = {cls._format_level(prev)}"
        )


def apriori_gen(prev: list[tuple[str, ...]]) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Join L_{k-1} ⋈ L_{k-1} then prune candidates whose (k-1)-subset is missing."""
    ordered = [tuple(sorted(itemset)) for itemset in prev]
    ordered.sort()
    k = len(ordered[0]) + 1 if ordered else 1
    prev_set = set(ordered)
    candidates: list[tuple[str, ...]] = []
    pruned: list[tuple[str, ...]] = []
    for i, l1 in enumerate(ordered):
        for l2 in ordered[i + 1 :]:
            if l1[:-1] != l2[:-1]:
                continue
            if l1[-1] >= l2[-1]:
                continue
            candidate = tuple(sorted(l1 + (l2[-1],)))
            subsets = list(combinations(candidate, k - 1))
            if all(subset in prev_set for subset in subsets):
                candidates.append(candidate)
            else:
                pruned.append(candidate)
    return candidates, pruned


def generate_association_rules(
    supports: dict[tuple[str, ...], float],
    minconf: float,
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    support_lookup = {tuple(sorted(k)): v for k, v in supports.items()}
    for itemset, sp in support_lookup.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                ant_sp = support_lookup.get(antecedent)
                if not ant_sp:
                    continue
                confidence = sp / ant_sp
                if confidence + 1e-12 >= minconf:
                    rules.append(
                        {
                            "antecedent": list(antecedent),
                            "consequent": list(consequent),
                            "support": sp,
                            "confidence": confidence,
                            "itemset": list(itemset),
                        }
                    )
    rules.sort(key=lambda r: (-r["confidence"], -r["support"], r["antecedent"], r["consequent"]))
    return rules


def _filter_frequent(
    candidates: list[tuple[str, ...]],
    transactions: list[set[str]],
    n: int,
    min_count: int,
    minsup: float,
) -> tuple[list[tuple[str, ...]], list[dict[str, Any]]]:
    rows = []
    frequent = []
    for itemset in candidates:
        count = support_count(itemset, transactions)
        sp = count / n if n else 0.0
        accepted = count >= min_count and sp + 1e-12 >= minsup
        rows.append(
            {
                "itemset": list(itemset),
                "count": count,
                "support": sp,
                "accepted": accepted,
            }
        )
        if accepted:
            frequent.append(itemset)
    return frequent, rows


def _maximal_itemsets(levels: dict[int, list[tuple[str, ...]]]) -> list[tuple[str, ...]]:
    all_sets = [s for level in levels.values() for s in level]
    maximal = []
    for itemset in all_sets:
        if not any(set(itemset) < set(other) for other in all_sets):
            maximal.append(itemset)
    return maximal
