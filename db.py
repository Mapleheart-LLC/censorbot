"""
Database models and engine for CensorBot.

``GuildSettings`` stores per-guild configuration that the dashboard can edit.
``CensorEvent`` records every censor action the bot performs, providing an
audit trail accessible through the dashboard.

The module creates the SQLAlchemy engine from the ``DATABASE_URL`` environment
variable (default: ``sqlite:///censorbot.db`` next to the working directory).
Call :func:`init_db` once at application startup to create tables that do not
yet exist.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///censorbot.db")

_connect_args: dict = (
    {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)

# ---------------------------------------------------------------------------
# ORM base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GuildSettings(Base):
    """Per-guild configuration that overrides the global environment defaults."""

    __tablename__ = "guild_settings"

    guild_id = Column(BigInteger, primary_key=True)

    # Comma-separated channel IDs; empty string ⇒ watch all channels.
    monitored_channels = Column(Text, nullable=False, server_default="")

    pixel_block_size = Column(Integer, nullable=False, server_default="20")
    min_confidence = Column(Float, nullable=False, server_default="0.0")

    # Comma-separated NudeNet class labels; empty string ⇒ use built-in defaults.
    censor_classes = Column(Text, nullable=False, server_default="")

    video_frame_sample_rate = Column(Integer, nullable=False, server_default="1")
    max_file_mb = Column(Float, nullable=False, server_default="8.0")
    dm_on_censor = Column(Boolean, nullable=False, server_default="1")
    log_channel_id = Column(BigInteger, nullable=True)


class CensorEvent(Base):
    """One censor action performed by the bot."""

    __tablename__ = "censor_events"
    __table_args__ = (Index("ix_censor_events_guild_ts", "guild_id", "ts"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    guild_id = Column(BigInteger, nullable=False)
    channel_id = Column(BigInteger, nullable=False)
    author_id = Column(BigInteger, nullable=False)

    # JSON-encoded list of censored / oversized filenames.
    filenames = Column(Text, nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create all tables that do not yet exist.  Safe to call repeatedly."""
    Base.metadata.create_all(engine)
