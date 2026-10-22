from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, event, inspect, text
from sqlmodel import SQLModel, create_engine

from holywater.storage import models as _models  # noqa: F401 - registers tables on the metadata


def make_engine(db_path: Path | str) -> Engine:
    """SQLite engine with foreign keys and WAL enabled. `":memory:"` is accepted for tests."""
    if str(db_path) == ":memory:":
        url = "sqlite://"
    else:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        if url != "sqlite://":
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create missing tables, then add columns that newer models declare but an older DB lacks."""
    SQLModel.metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """Forward-only, additive migration: `ALTER TABLE ... ADD COLUMN` with the model's default.

    Enough for a PoC whose schema only grows; anything destructive needs a real migration tool.
    """
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.tables.values():  # order is irrelevant for ADD COLUMN
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = column.type.compile(dialect=engine.dialect)
                default = getattr(column.default, "arg", None)
                clause = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl}"
                if default is not None and not callable(default):
                    clause += f" DEFAULT {default!r}"
                elif not column.nullable:
                    msg = f"cannot add NOT NULL column {table.name}.{column.name} without default"
                    raise RuntimeError(msg)
                conn.execute(text(clause))
