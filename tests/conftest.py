"""
Shared pytest fixtures for CensorBot tests.

An in-memory SQLite database is created per test session and its URL is
injected into the environment before db/settings modules are imported, so
every test works against a clean, isolated schema without touching the
production database file.
"""

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Point all db/settings code at an in-memory SQLite database for the test run.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from db import Base, CensorEvent, GuildSettings  # noqa: E402


@pytest.fixture(scope="session")
def db_engine():
    """Create the in-memory engine and schema once for the entire test session."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    """Yield a transactional session that is rolled back after each test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection, expire_on_commit=False)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
