"""The Alembic migrations must produce exactly the schema the ORM models describe."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from agent_backend.db.models import Base

ROOT = Path(__file__).resolve().parents[2]
PG_ONLY_INDEX = "ix_document_chunks_embedding_hnsw"


@pytest.fixture
def alembic_config(tmp_path: Path) -> tuple[Config, str]:
    db_file = (tmp_path / "migrations.db").as_posix()
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_file}")
    config.attributes["configure_logger"] = False
    return config, f"sqlite:///{db_file}"


def test_upgrade_head_matches_models(alembic_config: tuple[Config, str]) -> None:
    config, sync_url = alembic_config
    command.upgrade(config, "head")

    engine = create_engine(sync_url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    engine.dispose()
    # The HNSW vector index is PostgreSQL-only (it is never created on SQLite).
    diff = [d for d in diff if not (d[0] == "add_index" and d[1].name == PG_ONLY_INDEX)]
    assert diff == []


def test_downgrade_to_base_removes_tables(alembic_config: tuple[Config, str]) -> None:
    config, sync_url = alembic_config
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_engine(sync_url)
    tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
    engine.dispose()
    assert tables == set()
