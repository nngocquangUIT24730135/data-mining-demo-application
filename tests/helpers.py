from __future__ import annotations

from pathlib import Path

from datamining_app.data.csv_loader import CSVDataLoader
from datamining_app.data.preprocessor import Preprocessor, identifier_headers
from datamining_app.data.sqlite_loader import SQLiteDataLoader
from datamining_app.report_defaults import params_for

_CSV_DIR = Path(__file__).resolve().parents[1] / "data"


def load_report_dataset(db_path, table: str):
    csv_path = _CSV_DIR / f"{table}.csv"
    if csv_path.exists():
        raw = CSVDataLoader().load(str(csv_path))
    else:
        raw = SQLiteDataLoader(str(db_path)).load(table)
    exclude = identifier_headers(raw.headers)
    return Preprocessor().transform(raw, exclude_cols=exclude, strip_whitespace=True)


def report_params(algorithm_key: str, table: str) -> dict:
    return params_for(algorithm_key, table)
