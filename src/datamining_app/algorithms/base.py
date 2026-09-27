"""Algorithm lifecycle. Presentation lives in ``datamining_app.steps``."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from datamining_app.algorithms._logger import StepLogger, infer_step_kind
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult

__all__ = ["BaseAlgorithm", "StepLogger", "infer_step_kind"]


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
