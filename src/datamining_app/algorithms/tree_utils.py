from __future__ import annotations

from typing import Any

from datamining_app.core.models import PredictResult


def extract_tree_rules(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk a decision tree and emit IF-THEN style rules from root to each leaf."""
    rules: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], conds: list[tuple[str, str]]) -> None:
        if node.get("is_leaf"):
            rules.append(
                {
                    "conditions": list(conds),
                    "label": node.get("label"),
                    "samples": node.get("samples", 0),
                }
            )
            return
        attr = node.get("attribute", "?")
        for value, child in (node.get("children") or {}).items():
            walk(child, conds + [(attr, str(value))])

    walk(tree, [])
    return rules


def majority_class(class_counts: dict[str, int]) -> str:
    return max(class_counts, key=class_counts.get)


def predict_on_tree(
    tree: dict[str, Any],
    sample: dict[str, Any],
    *,
    algo_name: str = "cây quyết định",
) -> PredictResult:
    """Traverse a trained tree for a new sample (shared by ID3 and CART)."""
    normalized = {k: str(v).strip() for k, v in sample.items()}
    path: list[str] = []
    node = tree
    while not node.get("is_leaf"):
        attr = node["attribute"]
        value = normalized.get(attr, "")
        path.append(f"{attr} = {value}")
        children = node.get("children") or {}
        if value not in children:
            label = majority_class(node["class_counts"])
            explanation = (
                " → ".join(path)
                + f"\nGiá trị '{value}' không có nhánh. Lớp đa số = {label}."
            )
            return PredictResult(
                label=label,
                explanation=explanation,
                details={"path": path, "fallback": True},
                sample=normalized,
            )
        node = children[value]
    label = node["label"]
    path.append(f"Lá = {label}")
    return PredictResult(
        label=label,
        explanation=f"Suy luận trên {algo_name}:\n  " + "\n  → ".join(path),
        details={"path": path, "leaf": node},
        sample=normalized,
    )
