from __future__ import annotations

from collections import Counter
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm
from datamining_app.algorithms.dataset_utils import excluded_headers
from datamining_app.algorithms.tree_utils import extract_tree_rules, predict_on_tree
from datamining_app.steps.cart_steps import _CARTSteps
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult
from datamining_app.fmt import fmt_score


class CARTGiniAlgorithm(BaseAlgorithm):
    name = "CART (Gini Index)"
    description = "Xây dựng cây quyết định theo tiêu chí Gini Impurity."
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
        gini_s = gini_impurity(rows, decision)
        step_builder = _CARTSteps(log, len(rows), dict(counts), gini_impurity)
        step_builder.initial_gini_step(rows, decision, gini_s, counts)

        tree = self._build(rows, features, decision, max_depth, 0, step_builder, "Gốc")
        self._tree = tree
        self._features = features
        self._decision = decision

        rules = extract_tree_rules(tree)
        summary = (
            f"Cây CART (Gini) gốc = '{tree.get('attribute', tree.get('label'))}'. "
            f"G(S) = {fmt_score(gini_s)}. Độ sâu tối đa = {max_depth}."
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={"decision_attr": decision, "max_depth": max_depth},
            steps=log.steps,
            output={
                "tree": tree,
                "features": features,
                "decision": decision,
                "gini": gini_s,
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
        step_builder: "_CARTSteps",
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

        gini_s = gini_impurity(rows, decision)
        scores: list[tuple[str, float, float]] = []
        for feat in features:
            gini_a, delta = gini_gain(rows, feat, decision, gini_s)
            scores.append((feat, delta, gini_a))
        scores.sort(key=lambda x: -x[1])
        best, best_delta, _ = scores[0]
        values = sorted({r[best] for r in rows})
        split_node = {
            "is_leaf": False,
            "attribute": best,
            "gain": best_delta,
            "delta_gini": best_delta,
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
        step_builder.split_step(path, rows, decision, gini_s, scores, best, split_node)
        remaining = [f for f in features if f != best]
        children = {}
        for value in values:
            subset = [r for r in rows if r[best] == value]
            children[value] = self._build(
                subset,
                remaining,
                decision,
                max_depth,
                depth + 1,
                step_builder,
                f"{path} / {best}={value}",
            )
        node.update(
            {
                "is_leaf": False,
                "attribute": best,
                "gain": best_delta,
                "delta_gini": best_delta,
                "children": children,
            }
        )
        return node

    def predict(self, sample: dict[str, Any]) -> PredictResult:
        if not self._trained or self._tree is None:
            raise ValueError("Chưa huấn luyện cây CART.")
        return predict_on_tree(self._tree, sample, algo_name="cây CART (Gini)")


def gini_impurity(rows: list[dict[str, str]], decision: str) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    counts = Counter(r[decision] for r in rows)
    return 1.0 - sum((count / n) ** 2 for count in counts.values())


def gini_gain(
    rows: list[dict[str, str]],
    feature: str,
    decision: str,
    gini_s: float,
) -> tuple[float, float]:
    """Return (Gini_A(S), ΔGini(A))."""
    n = len(rows)
    weighted = 0.0
    for value in {r[feature] for r in rows}:
        subset = [r for r in rows if r[feature] == value]
        weighted += (len(subset) / n) * gini_impurity(subset, decision)
    return weighted, gini_s - weighted

__all__ = ["CARTGiniAlgorithm", "gini_gain", "gini_impurity"]
