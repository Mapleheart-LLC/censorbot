"""
Per-guild settings loader for the bot.

Loads :class:`GuildSettings` from the database, falling back to environment-
variable defaults when no row exists for a guild.  Also provides
:func:`record_censor_event` for writing audit rows.

The module initialises the database schema on first import so that bot.py and
the dashboard both see tables without an explicit bootstrap step.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from db import CensorEvent, GuildSettings, SessionFactory, init_db

log = logging.getLogger("censorbot.settings")

# ---------------------------------------------------------------------------
# Track whether the DB came up cleanly
# ---------------------------------------------------------------------------

_db_available = False

try:
    init_db()
    _db_available = True
except Exception:
    log.warning(
        "Database initialisation failed – per-guild settings disabled, "
        "falling back to environment-variable defaults.",
        exc_info=True,
    )

# ---------------------------------------------------------------------------
# Environment-variable defaults  (mirrors the original bot.py constants)
# ---------------------------------------------------------------------------

_raw_channels = os.getenv("MONITORED_CHANNELS", "").strip()
_DEFAULT_MONITORED_CHANNELS: set[int] = (
    {int(c.strip()) for c in _raw_channels.split(",") if c.strip()}
    if _raw_channels
    else set()
)

_DEFAULT_PIXEL_BLOCK_SIZE: int = int(os.getenv("PIXEL_BLOCK_SIZE", "20"))
_DEFAULT_MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.0"))
_DEFAULT_VIDEO_FRAME_SAMPLE_RATE: int = int(os.getenv("VIDEO_FRAME_SAMPLE_RATE", "1"))
_DEFAULT_MAX_FILE_MB: float = float(os.getenv("MAX_FILE_MB", "8.0"))
_DEFAULT_DM_ON_CENSOR: bool = (
    os.getenv("DM_ON_CENSOR", "true").strip().lower() in ("1", "true", "yes")
)

_raw_log_channel = os.getenv("LOG_CHANNEL_ID", "").strip()
_DEFAULT_LOG_CHANNEL_ID: Optional[int] = int(_raw_log_channel) if _raw_log_channel else None

_raw_classes = os.getenv("CENSOR_CLASSES", "").strip()
_DEFAULT_CENSOR_CLASSES: Optional[frozenset[str]] = (
    frozenset(c.strip() for c in _raw_classes.split(",") if c.strip())
    if _raw_classes
    else None
)

# ---------------------------------------------------------------------------
# GuildConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class GuildConfig:
    """All per-guild settings consumed by ``_process_message`` in bot.py."""

    monitored_channels: set[int] = field(default_factory=set)
    pixel_block_size: int = 20
    min_confidence: float = 0.0
    censor_classes: Optional[frozenset[str]] = None
    video_frame_sample_rate: int = 1
    max_file_mb: float = 8.0
    dm_on_censor: bool = True
    log_channel_id: Optional[int] = None

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)


# ---------------------------------------------------------------------------
# Sync helpers  (called via run_in_executor from async code)
# ---------------------------------------------------------------------------


def _env_defaults() -> GuildConfig:
    return GuildConfig(
        monitored_channels=set(_DEFAULT_MONITORED_CHANNELS),
        pixel_block_size=_DEFAULT_PIXEL_BLOCK_SIZE,
        min_confidence=_DEFAULT_MIN_CONFIDENCE,
        censor_classes=_DEFAULT_CENSOR_CLASSES,
        video_frame_sample_rate=_DEFAULT_VIDEO_FRAME_SAMPLE_RATE,
        max_file_mb=_DEFAULT_MAX_FILE_MB,
        dm_on_censor=_DEFAULT_DM_ON_CENSOR,
        log_channel_id=_DEFAULT_LOG_CHANNEL_ID,
    )


def _load_sync(guild_id: int) -> GuildConfig:
    """Read GuildSettings from the database; return env defaults on any error."""
    if not _db_available:
        return _env_defaults()

    try:
        with SessionFactory() as session:
            row: Optional[GuildSettings] = session.get(GuildSettings, guild_id)
    except Exception:
        log.exception("DB read failed for guild %s, using env defaults", guild_id)
        return _env_defaults()

    if row is None:
        return _env_defaults()

    raw_channels = (row.monitored_channels or "").strip()
    monitored: set[int] = (
        {int(c.strip()) for c in raw_channels.split(",") if c.strip()}
        if raw_channels
        else set()
    )

    raw_classes = (row.censor_classes or "").strip()
    censor_classes: Optional[frozenset[str]] = (
        frozenset(c.strip() for c in raw_classes.split(",") if c.strip())
        if raw_classes
        else None
    )

    return GuildConfig(
        monitored_channels=monitored,
        pixel_block_size=row.pixel_block_size,
        min_confidence=row.min_confidence,
        censor_classes=censor_classes,
        video_frame_sample_rate=row.video_frame_sample_rate,
        max_file_mb=row.max_file_mb,
        dm_on_censor=row.dm_on_censor,
        log_channel_id=row.log_channel_id,
    )


def _record_sync(
    guild_id: int,
    channel_id: int,
    author_id: int,
    filenames: list[str],
) -> None:
    """Insert a CensorEvent row; silently swallows errors so the bot keeps running."""
    if not _db_available:
        return

    try:
        with SessionFactory() as session:
            with session.begin():
                session.add(
                    CensorEvent(
                        guild_id=guild_id,
                        channel_id=channel_id,
                        author_id=author_id,
                        filenames=json.dumps(filenames),
                        ts=datetime.now(timezone.utc),
                    )
                )
    except Exception:
        log.exception("Failed to record censor event for guild %s", guild_id)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def load_guild_settings(guild_id: int) -> GuildConfig:
    """Return the :class:`GuildConfig` for *guild_id*, falling back to env defaults."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _load_sync, guild_id)


async def record_censor_event(
    guild_id: int,
    channel_id: int,
    author_id: int,
    filenames: list[str],
) -> None:
    """Write a :class:`CensorEvent` row for the completed censor action."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _record_sync, guild_id, channel_id, author_id, filenames)
