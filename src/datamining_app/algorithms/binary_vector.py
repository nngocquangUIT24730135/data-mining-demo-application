from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterable

from datamining_app.algorithms.apriori import generate_association_rules
from datamining_app.algorithms.base import BaseAlgorithm, StepLogger
from datamining_app.algorithms.dataset_utils import extract_transactions, format_itemset, minsup_count
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef
from datamining_app.fmt import fmt_pct, fmt_ratio


class BinaryVectorAlgorithm(BaseAlgorithm):
    """Frequent itemsets via binary context matrix and bitwise AND (report Ch.2)."""

    name = "Binary Vector"
    description = "Khai phá tập phổ biến bằng phép giao vector nhị phân."
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
        transactions = [list(dict.fromkeys(t)) for t in extract_transactions(dataset, params)]
        n = len(transactions)
        if n == 0:
            raise ValueError("Không có giao dịch nào.")
        min_count = minsup_count(minsup, n)
        items = _ordered_items(transactions)
        item_vectors = {item: _bit_vector(item, transactions) for item in items}

        log = self._new_logger()
        steps = _BinarySteps(log)
        steps.matrix_step(items, transactions, item_vectors)
        steps.init_step(n, minsup, min_count, minconf, len(items))

        mined = mine_binary_vectors(item_vectors, n, minsup)
        for level in mined.levels_detail:
            steps.level_step(level, mined.vectors)

        supports = mined.supports
        freq_rows = []
        for k_level, sets in mined.levels.items():
            for itemset in sets:
                sp = supports[itemset]
                count = int(round(sp * n))
                freq_rows.append([str(k_level), format_itemset(itemset), count, fmt_ratio(count, n)])
        steps.summary_step(freq_rows)
        rules = generate_association_rules(supports, minconf, n)
        steps.rules_step(rules, minconf)

        levels = {k: [list(s) for s in sets] for k, sets in mined.levels.items()}
        summary = (
            f"Binary Vector: {len(freq_rows)} tập phổ biến, "
            f"{len(rules)} luật (minsup={fmt_pct(minsup)}, minconf={fmt_pct(minconf)})."
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={"minsup": minsup, "minconf": minconf},
            steps=log.steps,
            output={
                "levels": {str(k): v for k, v in levels.items()},
                "vectors": {format_itemset(s): v for s, v in mined.vectors.items()},
                "supports": {format_itemset(s): sp for s, sp in supports.items()},
                "rules": rules,
                "items": items,
                "transactions": transactions,
            },
            summary=summary,
        )
        return self._finish(result)


class _BinarySteps:
    def __init__(self, log: StepLogger) -> None:
        self._log = log
        self._n = 0
        self._and_explained = False

    def matrix_step(
        self,
        items: list[str],
        transactions: list[list[str]],
        item_vectors: dict[str, list[int]],
    ) -> None:
        matrix_headers = ["TID"] + items
        matrix_rows = [
            [f"T{i}"] + ["1" if item in txn else "0" for item in items]
            for i, txn in enumerate(transactions, start=1)
        ]
        example_item = items[0] if items else "?"
        example_vec = item_vectors.get(example_item, [])
        example_sum = sum(example_vec)
        n = len(transactions) or 1
        label = format_itemset([example_item]) if items else "{?}"
        where = _tids(example_vec)
        description = (
            "① Ý tưởng cốt lõi — tại sao Binary Vector nhanh?\n"
            "  Apriori quét D với mỗi ứng viên → O(|Cₖ| × |D|) lần quét.\n"
            "  Binary Vector mã hóa D thành ma trận bit 1 LẦN → chỉ AND để đếm support.\n"
            "\n"
            "② Ma trận ngữ cảnh (O, I, R):\n"
            "  O = hàng = giao dịch (T1..TN)\n"
            "  I = cột  = item\n"
            "  R[t][i]  = 1 nếu item i ∈ T_t,  = 0 nếu không\n"
            "\n"
            "③ Vector nhị phân của một item — đọc theo CỘT:\n"
            "  v({item}) = cột tương ứng trong ma trận R\n"
            f"  Ví dụ: v({label}) = {_fmt_bits(example_vec)} → xuất hiện ở {where}\n"
            f"  → count({label}) = sum(v({label})) = {example_sum}\n"
            f"  → support({label}) = {fmt_ratio(example_sum, n)}"
        )
        self._log.add_table(
            "Xây ma trận nhị phân (O, I, R)",
            matrix_headers,
            matrix_rows,
            ["left"] + ["center"] * len(items),
            description=description,
            level="STEP_HEADER",
            items=items,
            transactions=transactions,
            vectors={k: list(v) for k, v in item_vectors.items()},
        )

    def init_step(self, n: int, minsup: float, min_count: int, minconf: float, n_items: int) -> None:
        self._n = n
        self._log.add(
            "Khởi tạo",
            (
                f"[Mã giả Bước 2] F₁ ← {{i : sum(v({{i}})) / |O| ≥ minsup}}\n"
                f"N = {n} giao dịch, |I| = {n_items}.\n"
                f"minsup = {fmt_pct(minsup)}  →  Count ≥ {min_count} (SP = Count/N)\n"
                f"minconf = {fmt_pct(minconf)}.\n"
                "\n"
                "② Quy ước ký hiệu — hiểu 1 lần, dùng xuyên suốt:\n"
                "  F₁ : Tập các 1 phần tử phổ biến  (item đơn có support ≥ minsup)\n"
                "  F₂ : Tập các 2 phần tử phổ biến  (cặp 2 item có support ≥ minsup)\n"
                "  Fₖ : Tập các k phần tử phổ biến  (tập k item có support ≥ minsup)\n"
                "  v(X): vector nhị phân biểu diễn sự xuất hiện của tập X\n"
                "  ─────────────────────────────────────────────────────────\n"
                "  Thuật toán tăng dần: F₁ → F₂ → F₃ → ... dừng khi Fₖ = {}"
            ),
            {"n": n, "minsup": minsup, "min_count": min_count, "minconf": minconf},
        )

    def level_step(self, level: dict[str, Any], vectors: dict[tuple[str, ...], list[int]] | None = None) -> None:
        k = level["k"]
        rows = level["rows"]
        if k == 1:
            description = (
                "[Mã giả Bước 2] F₁ ← tập các 1 phần tử phổ biến\n"
                "\n"
                "① F₁ là tập 1 phần tử (item đơn) vượt ngưỡng support:\n"
                "  Mỗi item i có vector nhị phân v({i}) = cột i trong ma trận R.\n"
                "  count({i}) = sum(v({i})) = số giao dịch chứa item i\n"
                "  support({i}) = count / N  →  chấp nhận vào F₁ nếu ≥ minsup"
                if rows
                else "Không sinh được tập ứng viên."
            )
        elif not self._and_explained:
            self._and_explained = True
            description = self._first_and_description(level, vectors or {})
        else:
            description = (
                f"[Mã giả Bước 4] F_{k} ← tập các {k} phần tử phổ biến\n"
                f"Áp dụng phép AND: v(X∪Y) = v(X) ∧ v(Y) với X, Y ∈ F_{k - 1}"
                if rows
                else "Không sinh được tập ứng viên."
            )
        frequent = level["frequent"]
        if frequent:
            conclusion = f"  → F_{k} = {_fmt(frequent)}"
        else:
            conclusion = (
                f"  → F_{k} = {{}}  (không có tập {k} phần tử nào đạt minsup)\n"
                f"  ⟹ Thuật toán DỪNG: F_{k} = {{}} → không thể tạo F_{k + 1}."
            )
        suffix = " (tập 1 phần tử phổ biến)" if k == 1 else f" (tập {k} phần tử phổ biến)"
        title = f"Cấp k = {k} — Vector F_{k}{suffix}" + (" — phép AND" if k >= 2 else "")
        self._log.add(
            title,
            description,
            {
                "k": k,
                "n": self._n,
                "candidates": rows,
                "frequent": [list(s) for s in frequent],
                "conclusion": conclusion,
            },
            "SUCCESS" if frequent else "WARNING",
        )

    def _first_and_description(
        self,
        level: dict[str, Any],
        vectors: dict[tuple[str, ...], list[int]],
    ) -> str:
        k = level["k"]
        rows = level["rows"]
        if not rows:
            return "Không sinh được tập ứng viên."
        row = rows[0]
        left = list(row.get("left") or [])
        right = list(row.get("right") or [])
        union = list(row.get("itemset") or [])
        left_vec = vectors.get(tuple(left), [])
        right_vec = vectors.get(tuple(right), [])
        union_vec = list(row.get("vector") or [])
        count = int(row.get("count") if row.get("count") is not None else sum(union_vec))
        if len(union_vec) <= 12:
            bits = "+".join(str(bit) for bit in union_vec)
            count_text = f"{bits} = {count}"
        else:
            count_text = f"sum(v) = {count}"
        verdict = f"Chấp nhận vào F_{k}" if row.get("accepted") else f"Loại, không vào F_{k}"
        x = format_itemset(left)
        y = format_itemset(right)
        both = format_itemset(union)
        return (
            f"[Mã giả Bước 4] F_{k} ← tập các {k} phần tử phổ biến\n"
            "\n"
            f"① F_{k} = tập {k} phần tử được xây từ F_{k - 1} bằng phép AND:\n"
            f"  Với mỗi cặp (X, Y) ∈ F_{k - 1} × F_{k - 1} có |X∪Y| = {k}:\n"
            "    v(X∪Y) = v(X) AND v(Y)\n"
            "    v(X∪Y)[t] = 1  ⟺  T_t chứa ĐỒNG THỜI cả X lẫn Y\n"
            "    → count(X∪Y) = sum(v(X∪Y))\n"
            "\n"
            "② Ví dụ minh họa (cặp đầu tiên):\n"
            f"  v({x}) = {_fmt_bits(left_vec)}\n"
            f"  v({y}) = {_fmt_bits(right_vec)}\n"
            f"  v({both}) = v({x}) ∧ v({y}) = {_fmt_bits(union_vec)}\n"
            f"  → count = {count_text}  →  support = {fmt_ratio(count, self._n or 1)}"
            f"  →  {verdict}"
        )

    def summary_step(self, freq_rows: list[list[Any]]) -> None:
        self._log.add_table(
            "Tổng hợp tập phổ biến",
            ["Cấp k", "Tập phổ biến", "Count", "Support"],
            freq_rows,
            ["right", "left", "right", "right"],
            level="SUCCESS",
            conclusion=_frequent_distribution(freq_rows),
        )

    def rules_step(self, rules: list[dict[str, Any]], minconf: float) -> None:
        self._log.add(
            "Sinh luật kết hợp từ tập phổ biến",
            (
                "[Mã giả Bước 5] Sinh luật từ ∪ Fₖ với conf ≥ minconf\n"
                "\n"
                "① Với mỗi tập phổ biến Z có |Z| ≥ 2, xét mọi cách tách X ⇒ Y:\n"
                "  conf(X ⇒ Y) = support(X∪Y) / support(X) = count(X∪Y) / count(X)\n"
                f"② Chấp nhận luật nếu conf ≥ minconf = {fmt_pct(minconf)}"
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


class BinaryMineResult:
    def __init__(self) -> None:
        self.levels: dict[int, list[tuple[str, ...]]] = {}
        self.levels_detail: list[dict[str, Any]] = []
        self.vectors: dict[tuple[str, ...], list[int]] = {}
        self.supports: dict[tuple[str, ...], float] = {}


def mine_binary_vectors(
    item_vectors: dict[str, list[int]],
    n: int,
    minsup: float,
) -> BinaryMineResult:
    """Level-wise: F1 from 1-item vectors, Fk from pairs in F(k-1) with |union|=k, bitwise AND."""
    out = BinaryMineResult()
    items = list(item_vectors)
    f1: list[tuple[str, ...]] = []
    f1_rows = []
    for item in items:
        vec = list(item_vectors[item])
        row = _eval_vector((item,), vec, n, minsup)
        f1_rows.append(row)
        key = (item,)
        out.vectors[key] = vec
        if row["accepted"]:
            f1.append(key)
            out.supports[key] = row["support"]
    out.levels[1] = f1
    out.levels_detail.append({"k": 1, "rows": f1_rows, "frequent": f1})

    prev = f1
    k = 2
    while prev:
        rows = []
        fk: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for left, right in combinations(prev, 2):
            union = tuple(sorted(set(left) | set(right)))
            if len(union) != k or union in seen:
                continue
            seen.add(union)
            vec = _and(out.vectors[left], out.vectors[right])
            row = _eval_vector(union, vec, n, minsup)
            row["left"] = list(left)
            row["right"] = list(right)
            rows.append(row)
            out.vectors[union] = vec
            if row["accepted"]:
                fk.append(union)
                out.supports[union] = row["support"]
        out.levels_detail.append({"k": k, "rows": rows, "frequent": fk})
        if not fk:
            break
        out.levels[k] = fk
        prev = fk
        k += 1
    return out


def _eval_vector(itemset: tuple[str, ...], vec: list[int], n: int, minsup: float) -> dict[str, Any]:
    count = sum(vec)
    sp = count / n if n else 0.0
    return {
        "itemset": list(itemset),
        "vector": list(vec),
        "count": count,
        "support": sp,
        "accepted": sp + 1e-12 >= minsup,
    }


def _ordered_items(transactions: list[list[str]]) -> list[str]:
    seen: list[str] = []
    found: set[str] = set()
    for txn in transactions:
        for item in txn:
            if item not in found:
                found.add(item)
                seen.append(item)
    return sorted(seen)


def _bit_vector(item: str, transactions: list[list[str]]) -> list[int]:
    return [1 if item in txn else 0 for txn in transactions]


def _and(left: Iterable[int], right: Iterable[int]) -> list[int]:
    return [int(a) & int(b) for a, b in zip(left, right)]


def _frequent_distribution(freq_rows: list[list[Any]]) -> str:
    counts = Counter(int(row[0]) for row in freq_rows)
    if not counts:
        return "  → 0 tập phổ biến  (F₁: {})."
    highest = max(counts)
    parts = []
    for k in range(1, highest + 2):
        found = counts.get(k, 0)
        parts.append(f"F_{k}: {found} tập" if found else f"F_{k}: {{}}")
    return f"  → {len(freq_rows)} tập phổ biến  ({', '.join(parts)})."


def _fmt_bits(vec: list[int]) -> str:
    return "(" + ", ".join(str(bit) for bit in vec) + ")"


def _tids(vec: list[int]) -> str:
    names = [f"T{i}" for i, bit in enumerate(vec, start=1) if bit]
    return ",".join(names) if names else "không giao dịch nào"


def _fmt(level: list[tuple[str, ...]]) -> str:
    if not level:
        return "{}"
    return "{" + ", ".join(format_itemset(x) for x in level) + "}"
