from __future__ import annotations

import math
from collections import Counter
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm
from datamining_app.algorithms.dataset_utils import excluded_headers
from datamining_app.algorithms.tree_utils import extract_tree_rules, predict_on_tree
from datamining_app.steps.id3_steps import _ID3Steps
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult
from datamining_app.fmt import fmt_score


class ID3Algorithm(BaseAlgorithm):
    name = "ID3"
    description = "Xây dựng cây quyết định theo Information Gain."
    supports_predict = True
    supports_visualization = True
    param_schema = {
        "decision_attr": ParamDef(
            type="choice",
            options="headers",
            default=None,
            label_vi="Thuộc tính quyết định (lớp)",
            label_en="Decision attribute (class)",
        ),
        "max_depth": ParamDef(
            type="int",
            default=6,
            min=1,
            max=20,
            step=1,
            label_vi="Độ sâu tối đa",
            label_en="Maximum depth",
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        self._tree: dict[str, Any] | None = None
        self._features: list[str] = []
        self._decision: str = ""

    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult:
        headers = excluded_headers(dataset, params)
        decision = params.get("decision_attr") or headers[-1]
        if decision not in headers:
            raise ValueError(f"Thuộc tính quyết định '{decision}' không tồn tại.")
        features = [h for h in headers if h != decision]
        max_depth = int(params.get("max_depth", 6))
        rows = [
            {"_id": str(i), **{h: str(row.get(h, "")).strip() for h in headers}}
            for i, row in enumerate(dataset.rows, start=1)
        ]

        log = self._new_logger()
        counts = Counter(r[decision] for r in rows)
        info_d = entropy(rows, decision)
        step_builder = _ID3Steps(log, len(rows), dict(counts), entropy)
        step_builder.initial_entropy_step(rows, decision, info_d, counts)

        tree = self._build(rows, features, decision, max_depth, 0, step_builder, "Gốc")
        self._tree = tree
        self._features = features
        self._decision = decision

        rules = extract_tree_rules(tree)
        summary = (
            f"Cây ID3 gốc = '{tree.get('attribute', tree.get('label'))}'. "
            f"I(S) = {fmt_score(info_d)}. Độ sâu tối đa cho phép = {max_depth}."
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={"decision_attr": decision, "max_depth": max_depth},
            steps=log.steps,
            output={
                "tree": tree,
                "features": features,
                "decision": decision,
                "entropy": info_d,
                "class_counts": dict(counts),
                "n_rows": len(rows),
                "rules": rules,
                "rows": rows,
            },
            summary=summary,
        )
        return self._finish(result)

    def _build(
        self,
        rows: list[dict[str, str]],
        features: list[str],
        decision: str,
        max_depth: int,
        depth: int,
        step_builder: _ID3Steps,
        path: str,
    ) -> dict[str, Any]:
        counts = Counter(r[decision] for r in rows)
        majority = counts.most_common(1)[0][0]
        node: dict[str, Any] = {
            "samples": len(rows),
            "class_counts": dict(counts),
            "path": path,
        }

        if depth > 0:
            step_builder.sub_dataset_step(path, rows, features, decision)

        if len(counts) == 1 or not features or depth >= max_depth:
            if len(counts) == 1:
                stop_reason = "pure"
            elif not features:
                stop_reason = "no_attrs"
            else:
                stop_reason = "max_depth"
            node.update({"is_leaf": True, "label": majority, "pending": False})
            step_builder.leaf_step(path, majority, counts, node, stop_reason, max_depth)
            return node

        info_d = entropy(rows, decision)
        gains = []
        for feat in features:
            split_info, gain = information_gain(rows, feat, decision, info_d)
            gains.append((feat, gain, split_info))
        gains.sort(key=lambda x: -x[1])
        best, best_gain, _ = gains[0]
        values = sorted({r[best] for r in rows})
        split_node = {
            "is_leaf": False,
            "attribute": best,
            "gain": best_gain,
            "samples": len(rows),
            "class_counts": dict(counts),
            "path": path,
            "pending": False,
            "children": {
                value: {
                    "is_leaf": True,
                    "label": "…",
                    "pending": True,
                    "samples": 0,
                    "path": f"{path} / {best}={value}",
                }
                for value in values
            },
        }
        step_builder.gain_step(path, rows, decision, info_d, gains, best, split_node)
        remaining = [f for f in features if f != best]
        children = {}
        for value in values:
            subset = [r for r in rows if r[best] == value]
            children[value] = self._build(
                subset, remaining, decision, max_depth, depth + 1, step_builder, f"{path} / {best}={value}"
            )
        node.update({"is_leaf": False, "attribute": best, "gain": best_gain, "children": children})
        return node

    def predict(self, sample: dict[str, Any]) -> PredictResult:
        if not self._trained or self._tree is None:
            raise ValueError("Chưa huấn luyện cây ID3.")
        return predict_on_tree(self._tree, sample, algo_name="cây ID3")


def entropy(rows: list[dict[str, str]], decision: str) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    counts = Counter(r[decision] for r in rows)
    total = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            total -= p * math.log2(p)
    return total


def information_gain(
    rows: list[dict[str, str]], feature: str, decision: str, info_d: float
) -> tuple[float, float]:
    n = len(rows)
    split = 0.0
    for value in {r[feature] for r in rows}:
        subset = [r for r in rows if r[feature] == value]
        split += (len(subset) / n) * entropy(subset, decision)
    return split, info_d - split

__all__ = ["ID3Algorithm", "entropy", "information_gain"]
