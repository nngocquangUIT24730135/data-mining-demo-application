from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from datamining_app.data.embedded_db.catalog import REGISTRY, config_json
from datamining_app.data.embedded_db.tables import TABLES

LEGACY_TABLES = {
    "transactions_s1",
    "transactions_s2",
    "recruitment_ds",
    "weather_play",
    "buys_computer",
    "iris_2d",
    "customer_segments",
    "kmeans_points",
}


def default_db_path() -> Path:
    return Path(__file__).resolve().parent / "datasets.db"


def _user_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row[0] for row in rows}


def _drop_all(conn: sqlite3.Connection) -> None:
    for name in _user_tables(conn):
        conn.execute(f'DROP TABLE IF EXISTS "{name}"')


def _needs_reseed(path: Path) -> bool:
    if not path.exists():
        return True
    with sqlite3.connect(path) as conn:
        names = _user_tables(conn)
    if "dataset_registry" not in names:
        return True
    if names & LEGACY_TABLES:
        return True
    expected = {table for table, _, _ in TABLES} | {"dataset_registry"}
    if not expected.issubset(names):
        return True
    return False


def ensure_seeded(path: Path | None = None) -> Path:
    db_path = Path(path) if path else default_db_path()
    if _needs_reseed(db_path):
        seed_database(db_path)
    return db_path


def seed_database(path: Path | None = None) -> Path:
    db_path = Path(path) if path else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _drop_all(conn)
        _create_and_insert(conn)
        conn.commit()
    return db_path


def _sql_type(value: Any) -> str:
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int) and not isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _create_and_insert(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE dataset_registry (
            id INTEGER PRIMARY KEY,
            algorithm TEXT NOT NULL,
            table_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            description TEXT NOT NULL,
            default_config TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO dataset_registry (algorithm, table_name, display_name, description, default_config)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                entry["algorithm"],
                entry["table_name"],
                entry["display_name"],
                entry["description"],
                config_json(entry["default_config"]),
            )
            for entry in REGISTRY
        ],
    )

    for name, headers, rows in TABLES:
        sample = rows[0]
        cols = ", ".join(f'"{h}" {_sql_type(sample[i])}' for i, h in enumerate(headers))
        conn.execute(f'CREATE TABLE "{name}" ({cols})')
        placeholders = ", ".join("?" for _ in headers)
        quoted = ", ".join(f'"{h}"' for h in headers)
        conn.executemany(f'INSERT INTO "{name}" ({quoted}) VALUES ({placeholders})', rows)
