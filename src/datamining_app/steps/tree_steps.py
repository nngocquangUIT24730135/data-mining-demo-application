"""Shared decision-tree prediction text for ID3 and CART."""
from __future__ import annotations


def missing_branch_explanation(path: list[str], value: str, label: str) -> str:
    """Explain a fallback when the sample value has no child."""
    return " → ".join(path) + f"\nGiá trị '{value}' không có nhánh. Lớp đa số = {label}."


def leaf_explanation(algo_name: str, path: list[str]) -> str:
    """Explain the path from the root to a leaf."""
    return f"Suy luận trên {algo_name}:\n  " + "\n  → ".join(path)


__all__ = ["leaf_explanation", "missing_branch_explanation"]
