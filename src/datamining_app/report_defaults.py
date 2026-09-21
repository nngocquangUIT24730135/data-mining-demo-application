"""Defaults come from dataset_registry (see embedded_db/catalog.py)."""

from __future__ import annotations

from typing import Any

from datamining_app.data.embedded_db.catalog import ALGO_FALLBACK_PARAMS, DEFAULT_TABLES, params_for as catalog_params
from datamining_app.data.preprocessor import identifier_headers


def params_for(algorithm_key: str, dataset_name: str) -> dict[str, Any]:
    specific = catalog_params(algorithm_key, dataset_name)
    if specific:
        return dict(specific)
    return dict(ALGO_FALLBACK_PARAMS.get(algorithm_key, {}))


def exclude_for(dataset_name: str, headers: list[str] | None = None) -> list[str]:
    if headers:
        return identifier_headers(headers)
    return ["pid", "tid", "id", "RID", "Instance"]
