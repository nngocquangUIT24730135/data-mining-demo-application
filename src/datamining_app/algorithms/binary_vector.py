from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from datamining_app.algorithms.apriori import generate_association_rules
from datamining_app.algorithms.base import BaseAlgorithm
from datamining_app.algorithms.dataset_utils import extract_transactions, format_itemset, minsup_count
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef


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
        matrix_headers = ["TID"] + items
        matrix_rows = [
            [f"T{i}"] + ["1" if item in txn else "0" for item in items]
            for i, txn in enumerate(transactions, start=1)
        ]
        log.add(
            "Ma trận ngữ cảnh nhị phân (O, I, R)",
            "",
            {
                "items": items,
                "transactions": transactions,
                "vectors": {k: list(v) for k, v in item_vectors.items()},
                "headers": matrix_headers,
                "rows": matrix_rows,
                "alignments": ["left"] + ["center"] * len(items),
            },
            "STEP_HEADER",
        )
        log.add(
            "Khởi tạo",
            f"N = {n} giao dịch, |I| = {len(items)}.\n"
            f"minsup = {minsup:.2%}  →  Count ≥ {min_count} (SP = Count/N)\n"
            f"minconf = {minconf:.2%}.",
            {"n": n, "minsup": minsup, "min_count": min_count, "minconf": minconf},
        )

        mined = mine_binary_vectors(item_vectors, n, minsup)
        for level in mined.levels_detail:
            k = level["k"]
            log.add(
                f"Cấp k = {k} — Vector F_{k}",
                "" if level["rows"] else "Không sinh được tập ứng viên.",
                {
                    "k": k,
                    "candidates": level["rows"],
                    "frequent": [list(s) for s in level["frequent"]],
                    "conclusion": f"  → F_{k} = {_fmt(level['frequent'])}",
                },
                "SUCCESS" if level["frequent"] else "WARNING",
            )

        supports = mined.supports
        freq_rows = []
        for k_level, sets in mined.levels.items():
            for itemset in sets:
                sp = supports[itemset]
                count = int(round(sp * n))
                freq_rows.append([str(k_level), format_itemset(itemset), count, f"{sp * 100:.2f}%"])
        log.add(
            "Tổng hợp tập phổ biến",
            "",
            {
                "headers": ["Cấp k", "Tập phổ biến", "Count", "Support"],
                "rows": freq_rows,
                "alignments": ["right", "left", "right", "right"],
                "conclusion": f"  → {len(freq_rows)} tập phổ biến.",
            },
            "SUCCESS",
        )
        rules = generate_association_rules(supports, minconf)
        log.add(
            "Sinh luật kết hợp từ tập phổ biến",
            "",
            {
                "rules": rules,
                "conclusion": (
                    f"  → {len(rules)} luật đạt minconf = {minconf:.2%}."
                    if rules
                    else f"  → Không có luật nào đạt minconf = {minconf:.2%}."
                ),
            },
            "SUCCESS",
        )

        levels = {k: [list(s) for s in sets] for k, sets in mined.levels.items()}
        summary = (
            f"Binary Vector: {len(freq_rows)} tập phổ biến, "
            f"{len(rules)} luật (minsup={minsup:.0%}, minconf={minconf:.0%})."
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
        self._last_result = result
        self._trained = True
        return result


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


def _fmt(level: list[tuple[str, ...]]) -> str:
    if not level:
        return "∅"
    return "{" + ", ".join(format_itemset(x) for x in level) + "}"


