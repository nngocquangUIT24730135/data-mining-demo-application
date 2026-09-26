from __future__ import annotations

from typing import Any

from datamining_app.core.models import Dataset

IDENTIFIER_NAMES = {
    "pid",
    "tid",
    "id",
    "rid",
    "object_id",
    "object",
    "instance",
    "point",
    "point id",
    "student id",
    "tuple",
    "transaction",
    "transaction id",
    "trans.",
    "tr.",
}


def is_identifier_header(header: str) -> bool:
    return header.strip().lower() in IDENTIFIER_NAMES


def identifier_headers(headers: list[str]) -> list[str]:
    return [h for h in headers if is_identifier_header(h)]


def remaining_headers(headers: list[str], exclude_cols: list[str] | None) -> list[str]:
    exclude = set(exclude_cols or [])
    return [h for h in headers if h not in exclude]


def _is_numeric_column(rows: list[dict[str, Any]], header: str) -> bool:
    from datamining_app.algorithms.dataset_utils import to_float

    values = [row.get(header) for row in rows]
    parsed = [to_float(v) for v in values]
    return bool(values) and all(p is not None for p in parsed)


class ExcludeValidationError(ValueError):
    pass


def validate_exclude(
    headers: list[str],
    rows: list[dict[str, Any]],
    exclude_cols: list[str],
    algorithm: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    extra = extra or {}
    unknown = [c for c in exclude_cols if c not in headers]
    if unknown:
        raise ExcludeValidationError(f"Cột không tồn tại: {', '.join(unknown)}")
    remaining = remaining_headers(headers, exclude_cols)
    if not remaining:
        raise ExcludeValidationError("Không được loại tất cả các cột. Phải còn ít nhất 1 cột.")
    if not rows:
        raise ExcludeValidationError("Dataset không còn dòng dữ liệu.")

    if algorithm in {"apriori", "binary_vector"}:
        item_cols = [h for h in remaining if not is_identifier_header(h)]
        if not item_cols:
            raise ExcludeValidationError("Apriori/Binary Vector cần còn ít nhất 1 cột item sau khi loại cột.")

    if algorithm in {"id3", "cart_gini", "naive_bayes", "naive_bayes_laplace", "rough_set"}:
        if len(remaining) < 2:
            raise ExcludeValidationError("Cần còn ít nhất 2 cột (điều kiện + quyết định).")
        decision = extra.get("decision_attr")
        if decision and decision in exclude_cols:
            raise ExcludeValidationError(f"Không được loại thuộc tính quyết định '{decision}'.")

    if algorithm == "kmeans":
        numeric = [h for h in remaining if _is_numeric_column(rows, h)]
        if not numeric:
            raise ExcludeValidationError("K-Means cần còn ít nhất 1 cột số sau khi loại cột.")
        k = extra.get("k")
        if k is not None:
            try:
                k_val = int(k)
            except (TypeError, ValueError):
                k_val = None
            if k_val is not None and k_val > len(rows):
                raise ExcludeValidationError(f"k={k_val} lớn hơn số dòng ({len(rows)}).")


class Preprocessor:
    def transform(
        self,
        dataset: Dataset,
        exclude_cols: list[str] | None = None,
        strip_whitespace: bool = True,
        lowercase: bool = False,
    ) -> Dataset:
        exclude = set(exclude_cols or [])
        headers = [h for h in dataset.headers if h not in exclude]
        rows = []
        for row in dataset.rows:
            new_row = {}
            for header in headers:
                value = row.get(header, "")
                if value is None:
                    value = ""
                text = str(value)
                if strip_whitespace:
                    text = text.strip()
                if lowercase:
                    text = text.lower()
                new_row[header] = text
            rows.append(new_row)
        return Dataset(
            name=dataset.name,
            headers=headers,
            rows=rows,
            source=dataset.source,
        )
