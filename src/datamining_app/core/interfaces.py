from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult


@runtime_checkable
class IAlgorithm(Protocol):
    name: str
    description: str
    param_schema: dict[str, ParamDef]
    supports_predict: bool

    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult: ...


@runtime_checkable
class PredictableAlgorithm(Protocol):
    def predict(self, sample: dict[str, Any]) -> PredictResult: ...


IPredictable = PredictableAlgorithm


@runtime_checkable
class IDataLoader(Protocol):
    def load(self, source: str) -> Dataset: ...


@runtime_checkable
class IVisualizer(Protocol):
    def draw(self, result: AlgorithmResult, ax: Any) -> None: ...


@runtime_checkable
class IExporter(Protocol):
    def export(self, result: AlgorithmResult, path: str) -> None: ...



class IStepBuilder(Protocol):
    """Console step builder living in ``datamining_app.steps``.

    Receives a step logger plus data the algorithm already computed, and appends
    StepLog entries. Implementations must not change numeric results.
    """
