from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult, StepLog


class BaseAlgorithm(ABC):
    name: str = ""
    description: str = ""
    param_schema: dict[str, ParamDef] = {}
    supports_predict: bool = False
    supports_visualization: bool = False

    def __init__(self) -> None:
        self._last_result: AlgorithmResult | None = None
        self._trained = False

    @abstractmethod
    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult:
        raise NotImplementedError

    def predict(self, sample: dict[str, Any]) -> PredictResult:
        raise NotImplementedError("Thuật toán này không hỗ trợ dự đoán.")

    def _new_logger(self) -> StepLogger:
        return StepLogger()

    def _finish(self, result: AlgorithmResult, *, trained: bool = True) -> AlgorithmResult:
        self._last_result = result
        self._trained = trained
        return result


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
