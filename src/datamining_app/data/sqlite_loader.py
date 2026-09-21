from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from datamining_app.core.models import Dataset, DatasetMeta


class SQLiteDataLoader:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)

    def tables(self) -> list[str]:
        path = Path(self.db_path)
        if not path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "AND name != 'dataset_registry' ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]

    def list_datasets(self, algorithm: str | None = None) -> list[DatasetMeta]:
        path = Path(self.db_path)
        if not path.exists():
            return []
        sql = "SELECT algorithm, table_name, display_name, description, default_config FROM dataset_registry"
        params: tuple = ()
        if algorithm:
            sql += " WHERE algorithm = ?"
            params = (algorithm,)
        sql += " ORDER BY id"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        result = []
        for algorithm_name, table_name, display_name, description, config in rows:
            try:
                default_config = json.loads(config) if config else {}
            except json.JSONDecodeError:
                default_config = {}
            result.append(
                DatasetMeta(
                    algorithm=algorithm_name,
                    table_name=table_name,
                    display_name=display_name,
                    description=description,
                    default_config=default_config,
                )
            )
        return result

    def load(self, table: str) -> Dataset:
        if table not in self.tables() and table != "dataset_registry":
            raise ValueError(f"Không có bảng '{table}'.")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(f'SELECT * FROM "{table}"')
            fetched = cur.fetchall()
            headers = [d[0] for d in cur.description]
        rows = [{h: row[h] if row[h] is not None else "" for h in headers} for row in fetched]
        return Dataset(name=table, headers=headers, rows=rows, source="sqlite")
