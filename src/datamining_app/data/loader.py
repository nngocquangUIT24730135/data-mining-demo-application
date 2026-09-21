from __future__ import annotations

from pathlib import Path

from datamining_app.core.models import Dataset, DatasetMeta
from datamining_app.data.csv_loader import CSVDataLoader
from datamining_app.data.embedded_db.seed import default_db_path, ensure_seeded
from datamining_app.data.sqlite_loader import SQLiteDataLoader


def _loader(db_path: str | Path | None = None) -> SQLiteDataLoader:
    path = Path(db_path) if db_path else default_db_path()
    ensure_seeded(path)
    return SQLiteDataLoader(str(path))


def list_sqlite_tables(db_path: str | Path | None = None) -> list[str]:
    return _loader(db_path).tables()


def list_datasets(algorithm: str | None = None, db_path: str | Path | None = None) -> list[DatasetMeta]:
    return _loader(db_path).list_datasets(algorithm)


def load_dataset(source: str, table: str | None = None) -> Dataset:
    if source.lower().endswith(".csv"):
        return CSVDataLoader().load(source)
    db_path = Path(source) if source.lower().endswith(".db") else default_db_path()
    ensure_seeded(db_path)
    if not table:
        raise ValueError("Cần tên bảng SQLite.")
    return SQLiteDataLoader(str(db_path)).load(table)
