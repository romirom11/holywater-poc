from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, Engine, event, inspect, text
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, create_engine

from holywater.observability.errors import ConfigError
from holywater.observability.logging import get_logger
from holywater.storage import models as _models  # noqa: F401 - registers tables on the metadata

log = get_logger(__name__)


def make_engine(db_path: Path | str) -> Engine:
    """SQLite engine with foreign keys and WAL enabled. `":memory:"` is accepted for tests.

    A file DB uses `NullPool`: every session opens its own connection and closes it on exit, so
    no connection outlives its session (no `ResourceWarning: unclosed database` on any exit path,
    including errors raised before the caller could dispose the engine). The in-memory DB keeps
    SQLAlchemy's default pool, which holds the one connection the data lives in.
    """
    if str(db_path) == ":memory:":
        url = "sqlite://"
        engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        engine = create_engine(url, connect_args={"check_same_thread": False}, poolclass=NullPool)

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
    added = _add_missing_columns(engine)
    if ("runs", "dry_run") in added:
        _backfill_dry_run(engine)


def _add_missing_columns(engine: Engine) -> set[tuple[str, str]]:
    """Forward-only, additive migration: `ALTER TABLE ... ADD COLUMN` with the model's default.

    Enough for a PoC whose schema only grows; anything destructive needs a real migration tool.
    Returns the `(table, column)` pairs it added.
    """
    inspector = inspect(engine)
    added: set[tuple[str, str]] = set()
    with engine.begin() as conn:
        for table in SQLModel.metadata.tables.values():  # order is irrelevant for ADD COLUMN
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = column.type.compile(dialect=engine.dialect)
                clause = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl}"
                default = _default_literal(column)
                if default is not None:
                    clause += f" DEFAULT {default}"
                elif not column.nullable:
                    msg = (
                        f"the database at {engine.url.database} predates column "
                        f"{table.name}.{column.name}, which has no default and cannot be added; "
                        "move the old DB aside or point storage.db_path elsewhere"
                    )
                    raise ConfigError(msg, context={"table": table.name, "column": column.name})
                conn.execute(text(clause))
                added.add((table.name, column.name))
                log.info("db.column_added", table=table.name, column=column.name, ddl=clause)
    return added


def _default_literal(column: Column[object]) -> str | None:
    """SQL literal for a column's scalar Python default, or None when there is none."""
    value = getattr(column.default, "arg", None)
    if value is None or callable(value):
        return None
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "1" if value else "0"
    if isinstance(value, int | float):
        return repr(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _backfill_dry_run(engine: Engine) -> None:
    """Runs recorded before `runs.dry_run` existed default to real. Mark those whose every cost row
    names a fake model (`fake/...`, as every fake provider does) as dry runs, so old dry-run
    smoke tests stop counting as real runs in `hw eval compare` and tag checks."""
    sql = text(
        "UPDATE runs SET dry_run = 1 WHERE id IN ("
        " SELECT run_id FROM costs GROUP BY run_id"
        " HAVING SUM(CASE WHEN model LIKE 'fake/%' THEN 0 ELSE 1 END) = 0)"
    )
    with engine.begin() as conn:
        marked = conn.execute(sql).rowcount
    log.info("db.dry_run_backfilled", runs=marked)
