from __future__ import annotations

from pathlib import Path

from datamining_app.console.formatter import ConsoleFormatter
from datamining_app.core.models import AlgorithmResult


class TxtExporter:
    def __init__(self) -> None:
        self.formatter = ConsoleFormatter()

    def export(self, result: AlgorithmResult, path: str) -> None:
        Path(path).write_text(self.render(result), encoding="utf-8")

    def render(self, result: AlgorithmResult) -> str:
        return self.formatter.render_result(result)
