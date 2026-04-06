"""
Unit tests for db.py and settings.py.

Tests cover:
- GuildSettings and CensorEvent ORM models
- init_db() idempotency
- settings._load_sync / load_guild_settings
- settings._record_sync / record_censor_event
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# db imports
# ---------------------------------------------------------------------------

from db import Base, CensorEvent, GuildSettings, SessionFactory, init_db


# ---------------------------------------------------------------------------
# GuildSettings – CRUD via db_session fixture
# ---------------------------------------------------------------------------


class TestGuildSettings:
    def test_insert_and_retrieve(self, db_session):
        row = GuildSettings(
            guild_id=111,
            monitored_channels="100,200",
            pixel_block_size=10,
            min_confidence=0.5,
            censor_classes="FEMALE_BREAST_EXPOSED",
            video_frame_sample_rate=2,
            max_file_mb=16.0,
            dm_on_censor=False,
            log_channel_id=999,
        )
        db_session.add(row)
        db_session.flush()

        fetched: GuildSettings = db_session.get(GuildSettings, 111)
        assert fetched is not None
        assert fetched.guild_id == 111
        assert fetched.monitored_channels == "100,200"
        assert fetched.pixel_block_size == 10
        assert fetched.min_confidence == pytest.approx(0.5)
        assert fetched.censor_classes == "FEMALE_BREAST_EXPOSED"
        assert fetched.video_frame_sample_rate == 2
        assert fetched.max_file_mb == pytest.approx(16.0)
        assert fetched.dm_on_censor is False
        assert fetched.log_channel_id == 999

    def test_defaults(self, db_session):
        """Columns with server_default values must be set after flush."""
        row = GuildSettings(guild_id=222)
        db_session.add(row)
        db_session.flush()
        db_session.expire(row)

        fetched: GuildSettings = db_session.get(GuildSettings, 222)
        assert fetched is not None
        # server_defaults are applied by the DB; verify the column exists.
        assert fetched.monitored_channels is not None
        assert fetched.log_channel_id is None

    def test_update(self, db_session):
        row = GuildSettings(guild_id=333, pixel_block_size=5)
        db_session.add(row)
        db_session.flush()

        row.pixel_block_size = 30
        db_session.flush()

        fetched: GuildSettings = db_session.get(GuildSettings, 333)
        assert fetched.pixel_block_size == 30

    def test_missing_guild_returns_none(self, db_session):
        result = db_session.get(GuildSettings, 99999)
        assert result is None


# ---------------------------------------------------------------------------
# CensorEvent – CRUD
# ---------------------------------------------------------------------------


class TestCensorEvent:
    def test_insert_and_retrieve(self, db_session):
        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        event = CensorEvent(
            guild_id=111,
            channel_id=222,
            author_id=333,
            filenames=json.dumps(["file1.jpg", "file2.mp4"]),
            ts=ts,
        )
        db_session.add(event)
        db_session.flush()

        fetched: CensorEvent = db_session.get(CensorEvent, event.id)
        assert fetched is not None
        assert fetched.guild_id == 111
        assert fetched.channel_id == 222
        assert fetched.author_id == 333
        assert json.loads(fetched.filenames) == ["file1.jpg", "file2.mp4"]
        assert fetched.ts == ts

    def test_multiple_events_for_guild(self, db_session):
        for i in range(5):
            db_session.add(
                CensorEvent(
                    guild_id=444,
                    channel_id=555,
                    author_id=666 + i,
                    filenames=json.dumps([f"img{i}.png"]),
                    ts=datetime.now(timezone.utc),
                )
            )
        db_session.flush()

        events = (
            db_session.query(CensorEvent)
            .filter_by(guild_id=444)
            .all()
        )
        assert len(events) == 5

    def test_autoincrement_id(self, db_session):
        ev1 = CensorEvent(
            guild_id=777,
            channel_id=1,
            author_id=1,
            filenames="[]",
            ts=datetime.now(timezone.utc),
        )
        ev2 = CensorEvent(
            guild_id=777,
            channel_id=2,
            author_id=2,
            filenames="[]",
            ts=datetime.now(timezone.utc),
        )
        db_session.add_all([ev1, ev2])
        db_session.flush()
        assert ev1.id != ev2.id


# ---------------------------------------------------------------------------
# init_db idempotency
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_init_db_is_idempotent(self):
        """Calling init_db() twice must not raise."""
        init_db()
        init_db()  # second call should be a no-op


# ---------------------------------------------------------------------------
# settings._load_sync
# ---------------------------------------------------------------------------


class TestLoadSync:
    def test_returns_env_defaults_when_no_db_row(self):
        """When no GuildSettings row exists, env defaults are returned."""
        import settings as settings_mod

        cfg = settings_mod._load_sync(guild_id=88888)
        # Must return a GuildConfig instance without raising.
        from settings import GuildConfig
        assert isinstance(cfg, GuildConfig)

    def test_returns_env_defaults_when_db_unavailable(self):
        """When _db_available is False, _load_sync returns env defaults."""
        import settings as settings_mod

        with patch.object(settings_mod, "_db_available", False):
            cfg = settings_mod._load_sync(guild_id=12345)
        from settings import GuildConfig
        assert isinstance(cfg, GuildConfig)

    def test_loads_row_from_db(self):
        """When a GuildSettings row exists, _load_sync maps it to GuildConfig."""
        import settings as settings_mod

        # Write a row directly via the real SessionFactory (in-memory DB from conftest).
        with settings_mod.SessionFactory() as session:
            with session.begin():
                row = GuildSettings(
                    guild_id=55555,
                    monitored_channels="100,200",
                    pixel_block_size=15,
                    min_confidence=0.7,
                    censor_classes="MALE_GENITALIA_EXPOSED",
                    video_frame_sample_rate=3,
                    max_file_mb=4.0,
                    dm_on_censor=False,
                    log_channel_id=9876,
                )
                session.add(row)

        cfg = settings_mod._load_sync(guild_id=55555)

        assert cfg.monitored_channels == {100, 200}
        assert cfg.pixel_block_size == 15
        assert cfg.min_confidence == pytest.approx(0.7)
        assert cfg.censor_classes == frozenset({"MALE_GENITALIA_EXPOSED"})
        assert cfg.video_frame_sample_rate == 3
        assert cfg.max_file_mb == pytest.approx(4.0)
        assert cfg.dm_on_censor is False
        assert cfg.log_channel_id == 9876

    def test_empty_censor_classes_returns_none(self):
        """An empty censor_classes string in the DB row maps to None."""
        import settings as settings_mod

        with settings_mod.SessionFactory() as session:
            with session.begin():
                session.add(GuildSettings(
                    guild_id=55556,
                    censor_classes="",
                ))

        cfg = settings_mod._load_sync(guild_id=55556)
        assert cfg.censor_classes is None

    def test_empty_monitored_channels_returns_empty_set(self):
        """An empty monitored_channels string in the DB row maps to an empty set."""
        import settings as settings_mod

        with settings_mod.SessionFactory() as session:
            with session.begin():
                session.add(GuildSettings(guild_id=55557, monitored_channels=""))

        cfg = settings_mod._load_sync(guild_id=55557)
        assert cfg.monitored_channels == set()

    def test_db_exception_returns_env_defaults(self):
        """A DB read error must be swallowed and env defaults returned."""
        import settings as settings_mod

        with patch.object(settings_mod, "SessionFactory", side_effect=Exception("db down")):
            cfg = settings_mod._load_sync(guild_id=77777)
        from settings import GuildConfig
        assert isinstance(cfg, GuildConfig)


# ---------------------------------------------------------------------------
# settings._record_sync
# ---------------------------------------------------------------------------


class TestRecordSync:
    def test_inserts_event_row(self):
        """_record_sync should insert a CensorEvent that can then be queried."""
        import settings as settings_mod

        settings_mod._record_sync(
            guild_id=10001,
            channel_id=20001,
            author_id=30001,
            filenames=["test.jpg"],
        )

        with settings_mod.SessionFactory() as session:
            events = (
                session.query(CensorEvent)
                .filter_by(guild_id=10001)
                .all()
            )
        assert len(events) == 1
        assert json.loads(events[0].filenames) == ["test.jpg"]

    def test_silently_ignores_db_error(self):
        """_record_sync must not raise even when the DB is unreachable."""
        import settings as settings_mod

        with patch.object(settings_mod, "_db_available", False):
            settings_mod._record_sync(
                guild_id=99999,
                channel_id=1,
                author_id=1,
                filenames=["x.png"],
            )  # must not raise

    def test_db_write_exception_is_swallowed(self):
        """An unexpected DB exception must be logged and swallowed."""
        import settings as settings_mod

        with patch.object(
            settings_mod, "SessionFactory", side_effect=Exception("write error")
        ):
            settings_mod._record_sync(
                guild_id=88888,
                channel_id=1,
                author_id=1,
                filenames=["y.png"],
            )  # must not raise


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------


class TestAsyncAPI:
    def test_load_guild_settings_is_awaitable(self):
        """load_guild_settings must return a coroutine that resolves."""
        import settings as settings_mod
        from settings import GuildConfig

        result = asyncio.run(settings_mod.load_guild_settings(99991))
        assert isinstance(result, GuildConfig)

    def test_record_censor_event_is_awaitable(self):
        """record_censor_event must return a coroutine without raising."""
        import settings as settings_mod

        asyncio.run(
            settings_mod.record_censor_event(
                guild_id=99992,
                channel_id=1,
                author_id=1,
                filenames=["z.png"],
            )
        )  # must not raise


# ---------------------------------------------------------------------------
# GuildConfig dataclass
# ---------------------------------------------------------------------------


class TestGuildConfig:
    def test_max_file_bytes_conversion(self):
        from settings import GuildConfig

        cfg = GuildConfig(max_file_mb=8.0)
        assert cfg.max_file_bytes == 8 * 1024 * 1024

    def test_defaults(self):
        from settings import GuildConfig

        cfg = GuildConfig()
        assert cfg.pixel_block_size == 20
        assert cfg.min_confidence == 0.0
        assert cfg.censor_classes is None
        assert cfg.video_frame_sample_rate == 1
        assert cfg.max_file_mb == 8.0
        assert cfg.dm_on_censor is True
        assert cfg.log_channel_id is None
        assert cfg.monitored_channels == set()
