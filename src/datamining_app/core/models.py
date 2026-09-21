from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Dataset:
    name: str
    headers: list[str]
    rows: list[dict[str, Any]]
    source: str = "csv"

    def copy(self) -> Dataset:
        return Dataset(
            name=self.name,
            headers=list(self.headers),
            rows=[dict(row) for row in self.rows],
            source=self.source,
        )

    def column_values(self, header: str) -> list[Any]:
        return [row.get(header) for row in self.rows]


@dataclass
class StepLog:
    step_number: int
    title: str
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    level: str = "INFO"
    kind: str = "text"  # text | table | matrix | rules | prediction


@dataclass
class AlgorithmResult:
    algorithm_name: str
    parameters: dict[str, Any]
    steps: list[StepLog]
    output: Any
    summary: str
    predictions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParamDef:
    type: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | str | None = None
    label_vi: str = ""
    label_en: str = ""


@dataclass
class PredictResult:
    label: str
    explanation: str
    details: dict[str, Any] = field(default_factory=dict)
    sample: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMeta:
    algorithm: str
    table_name: str
    display_name: str
    description: str
    default_config: dict[str, Any] = field(default_factory=dict)
