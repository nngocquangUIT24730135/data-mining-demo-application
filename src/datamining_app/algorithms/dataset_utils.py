from __future__ import annotations

from itertools import combinations
from math import ceil
from typing import Any, Iterable

from datamining_app.core.models import Dataset


ID_LIKE = {
    "tid",
    "id",
    "pid",
    "rid",
    "transaction",
    "instance",
    "object",
    "object_id",
    "ứng viên",
    "ung vien",
    "điểm",
    "diem",
}


def is_truthy(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "x", "t"}


def looks_binary_column(values: Iterable[Any]) -> bool:
    allowed = {"0", "1", "true", "false", "yes", "no", "y", "n", "x", "", "none"}
    seen = set()
    for value in values:
        text = str(value).strip().lower() if value is not None else ""
        seen.add(text)
        if text not in allowed:
            return False
    return bool(seen - {"", "none"})


def extract_transactions(dataset: Dataset, params: dict[str, Any]) -> list[list[str]]:
    exclude = {str(c) for c in (params.get("exclude_cols") or [])}
    headers = [h for h in dataset.headers if h not in exclude]
    if not headers:
        raise ValueError("Không còn cột nào sau khi loại bỏ.")

    item_col = params.get("items_col")
    if item_col and item_col in dataset.headers:
        return [_split_items(row.get(item_col, "")) for row in dataset.rows]

    for header in headers:
        if header.lower() in {"items", "itemset", "itemsets", "list of items"}:
            return [_split_items(row.get(header, "")) for row in dataset.rows]

    data_headers = [h for h in headers if h.lower() not in ID_LIKE]
    if data_headers and all(
        looks_binary_column(dataset.column_values(h)) for h in data_headers
    ):
        transactions: list[list[str]] = []
        for row in dataset.rows:
            transactions.append([h for h in data_headers if is_truthy(row.get(h))])
        return transactions

    if len(data_headers) == 1:
        return [_split_items(row.get(data_headers[0], "")) for row in dataset.rows]

    raise ValueError(
        "Không nhận diện được dữ liệu giao dịch. Dùng cột Items (phẩy) hoặc ma trận 0/1."
    )


def _split_items(raw: Any) -> list[str]:
    text = str(raw or "").replace(";", ",")
    items = [part.strip() for part in text.split(",") if part.strip()]
    return items


def support_count(itemset: Iterable[str], transactions: list[set[str]]) -> int:
    frozen = set(itemset)
    return sum(1 for txn in transactions if frozen <= txn)


def minsup_count(minsup: float, n: int) -> int:
    if n <= 0:
        return 0
    return max(1, int(ceil(minsup * n - 1e-12)))


def format_itemset(itemset: Iterable[str]) -> str:
    return "{" + ", ".join(itemset) + "}"


def excluded_headers(dataset: Dataset, params: dict[str, Any]) -> list[str]:
    exclude = {str(c) for c in (params.get("exclude_cols") or [])}
    return [h for h in dataset.headers if h not in exclude]


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None
