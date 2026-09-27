"""Step log collector used by algorithm runs.

Algorithms create a logger; step builders in ``datamining_app.steps`` append to it.
"""
from __future__ import annotations

from typing import Any

from datamining_app.core.models import StepLog


class StepLogger:
    def __init__(self) -> None:
        self.steps: list[StepLog] = []

    def add(
        self,
        title: str,
        description: str,
        data: dict[str, Any] | None = None,
        level: str = "INFO",
        kind: str = "text",
    ) -> StepLog:
        payload = data or {}
        inferred = infer_step_kind(payload)
        if kind == "text" and inferred != "text":
            kind = inferred
        step = StepLog(
            step_number=len(self.steps) + 1,
            title=title,
            description=description,
            data=payload,
            level=level,
            kind=kind or infer_step_kind(payload),
        )
        self.steps.append(step)
        return step

    def add_table(
        self,
        title: str,
        headers: list[Any],
        rows: list[Any],
        alignments: list[str] | None = None,
        description: str = "",
        level: str = "INFO",
        **extra: Any,
    ) -> StepLog:
        data: dict[str, Any] = {"headers": headers, "rows": rows, **extra}
        if alignments is not None:
            data["alignments"] = alignments
        return self.add(title, description, data, level)


def infer_step_kind(data: dict[str, Any]) -> str:
    if data.get("headers") is not None and data.get("rows") is not None:
        return "table"
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) and "itemset" in candidates[0]:
        return "table"
    rules = data.get("rules")
    if isinstance(rules, list) and rules and isinstance(rules[0], dict) and (
        "antecedent" in rules[0] or "lhs" in rules[0]
    ):
        return "rules"
    if data.get("cells"):
        return "matrix"
    return "text"


__all__ = ["StepLogger", "infer_step_kind"]
