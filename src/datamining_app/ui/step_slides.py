from __future__ import annotations

from datamining_app.core.models import AlgorithmResult, StepLog


def console_slides(result: AlgorithmResult) -> list[StepLog]:
    """One visualization slide per console StepLog (1:1)."""
    return list(result.steps)
