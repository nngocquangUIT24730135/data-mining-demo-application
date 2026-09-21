from __future__ import annotations

import csv
from pathlib import Path

from datamining_app.core.models import Dataset


class CSVDataLoader:
    def load(self, source: str) -> Dataset:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(text.splitlines(), dialect=dialect)
        if not reader.fieldnames:
            raise ValueError("CSV không có header.")
        headers = [h.strip() for h in reader.fieldnames if h and h.strip()]
        rows = []
        for row in reader:
            rows.append({h: (row.get(h) if row.get(h) is not None else "") for h in headers})
        return Dataset(name=path.stem, headers=headers, rows=rows, source="csv")
