"""
CensorBot – a Discord bot that censors images and videos.

For every message that contains image or video attachments the bot will:
  1. Download each attachment.
  2. Run NudeNet detection and pixelate any explicit regions found.
  3. Delete the original message.
  4. Re-post the message content together with the censored media.

Per-guild configuration is loaded from the database (see ``settings.py``).
Environment variables serve as global defaults when no database row exists for
a guild, and as the sole configuration source when the database is unavailable.

Required environment variable
------------------------------
DISCORD_TOKEN   Discord bot token.

Optional environment variables (global defaults / fallbacks)
--------------------------------------------------------------
MONITORED_CHANNELS     Comma-separated list of channel IDs to watch.
                       Leave empty (the default) to watch every channel.
PIXEL_BLOCK_SIZE       Integer pixel block size for pixelation (default: 20).
MIN_CONFIDENCE         Minimum NudeNet detection confidence 0–1 (default: 0.0).
CENSOR_CLASSES         Comma-separated NudeNet labels to censor; leave empty
                       to use the built-in defaults.
VIDEO_FRAME_SAMPLE_RATE  Run NudeNet every N frames for video (default: 1).
MAX_FILE_MB            Max censored-file size in MB before it is dropped
                       (default: 8.0).
DM_ON_CENSOR           Send the original author a DM explaining the removal
                       (default: true).
LOG_CHANNEL_ID         Channel ID for a moderator audit log (default: none).
HEALTH_PORT            Port for the HTTP health-check endpoint; 0 to disable
                       (default: 8080).
"""

import asyncio
import functools
import io
import logging
import os
from pathlib import PurePosixPath

import discord
from discord.ext import commands
from dotenv import load_dotenv

from censor import (
    censor_image,
    censor_video,
    is_image,
    is_video,
)
from settings import GuildConfig, load_guild_settings, record_censor_event

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("censorbot")

# ---------------------------------------------------------------------------
# Bot-level configuration  (not per-guild)
# ---------------------------------------------------------------------------

TOKEN: str = os.environ["DISCORD_TOKEN"]
HEALTH_PORT: int = int(os.getenv("HEALTH_PORT", "8080"))

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Guard against the health-check server being started more than once on
# repeated on_ready events (e.g. after a reconnect).
_health_server_started = False


# ---------------------------------------------------------------------------
# Health-check HTTP server
# ---------------------------------------------------------------------------


async def _health_handler(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Minimal HTTP/1.1 handler that returns 200 OK for any request."""
    try:
        await reader.read(1024)  # consume the request
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def _start_health_server(port: int) -> None:
    server = await asyncio.start_server(_health_handler, "0.0.0.0", port)
    log.info("Health-check server listening on port %d", port)
    async with server:
        await server.serve_forever()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_censor_image(
    data: bytes,
    cfg: GuildConfig,
) -> bytes:
    """Offload the CPU-bound image censoring to a thread so the event loop stays responsive."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(
            censor_image,
            data,
            block_size=cfg.pixel_block_size,
            min_confidence=cfg.min_confidence,
            censor_classes=cfg.censor_classes,
        ),
    )


async def _run_censor_video(
    data: bytes,
    input_extension: str,
    cfg: GuildConfig,
) -> bytes:
    """Offload the CPU-bound video censoring to a thread so the event loop stays responsive."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(
            censor_video,
            data,
            input_extension=input_extension,
            block_size=cfg.pixel_block_size,
            min_confidence=cfg.min_confidence,
            censor_classes=cfg.censor_classes,
            frame_sample_rate=cfg.video_frame_sample_rate,
        ),
    )


async def _dm_author(author: discord.User | discord.Member, channel_name: str) -> None:
    """Attempt to DM the original author explaining why their message was removed."""
    try:
        await author.send(
            f"👋 Your message in **{channel_name}** was removed because it "
            "contained media with explicit content.  A censored version has "
            "been posted in its place."
        )
    except discord.Forbidden:
        log.debug("Could not DM %s (DMs disabled or bot blocked)", author)
    except discord.HTTPException:
        log.debug("Failed to send DM to %s", author)


async def _post_to_log_channel(
    message: discord.Message,
    filenames: list[str],
    log_channel_id: int,
) -> None:
    """Post a moderation audit embed to the configured log channel."""
    channel = bot.get_channel(log_channel_id)
    if channel is None:
        log.warning("Log channel %s not found or bot lacks access", log_channel_id)
        return
    if not isinstance(channel, discord.TextChannel):
        return

    embed = discord.Embed(
        title="Content censored",
        color=discord.Color.orange(),
    )
    embed.add_field(name="Author", value=f"{message.author} ({message.author.id})", inline=True)
    embed.add_field(name="Channel", value=f"<#{message.channel.id}>", inline=True)
    embed.add_field(name="Files", value="\n".join(filenames) or "—", inline=False)
    if message.content:
        embed.add_field(name="Original text", value=message.content[:1024], inline=False)
    embed.set_footer(text=f"Message ID: {message.id}")

    try:
        await channel.send(embed=embed)
    except discord.HTTPException:
        log.exception("Failed to post to log channel %s", log_channel_id)


async def _process_message(message: discord.Message) -> None:
    """Core pipeline: censor attachments, delete original, repost censored version."""
    # Ignore messages from bots (including ourselves) to prevent loops.
    if message.author.bot:
        return

    # DMs have no guild context; skip them.
    if message.guild is None:
        return

    # Load per-guild settings from the database (falls back to env defaults).
    cfg = await load_guild_settings(message.guild.id)

    # Honour channel allowlist when configured.
    if cfg.monitored_channels and message.channel.id not in cfg.monitored_channels:
        return

    media_attachments = [
        att
        for att in message.attachments
        if att.content_type and (is_image(att.content_type) or is_video(att.content_type))
    ]

    if not media_attachments:
        return

    censored_files: list[discord.File] = []
    censored_filenames: list[str] = []
    oversized_names: list[str] = []

    for attachment in media_attachments:
        try:
            raw_data = await attachment.read()
            content_type = attachment.content_type or ""
            filename = attachment.filename

            if is_image(content_type):
                censored_data = await _run_censor_image(raw_data, cfg)
                stem = PurePosixPath(filename).stem
                out_name = f"{stem}_censored.png"
            else:
                ext = PurePosixPath(filename).suffix or ".mp4"
                censored_data = await _run_censor_video(raw_data, ext, cfg)
                stem = PurePosixPath(filename).stem
                out_name = f"{stem}_censored.mp4"

            if len(censored_data) > cfg.max_file_bytes:
                log.warning(
                    "Censored file %s is %.1f MB, exceeds limit of %.1f MB – skipping attachment",
                    out_name,
                    len(censored_data) / 1024 / 1024,
                    cfg.max_file_mb,
                )
                oversized_names.append(out_name)
                continue

            censored_files.append(discord.File(io.BytesIO(censored_data), filename=out_name))
            censored_filenames.append(out_name)
            log.info(
                "Censored %s from %s in channel %s",
                filename,
                message.author,
                message.channel,
            )
        except Exception:
            log.exception("Failed to censor attachment %s", attachment.filename)

    if not censored_files and not oversized_names:
        # Nothing was successfully processed – leave the original message alone.
        return

    # Delete the original message before re-posting the censored version.
    try:
        await message.delete()
    except discord.Forbidden:
        log.warning(
            "Missing permission to delete message %s in channel %s",
            message.id,
            message.channel,
        )
        # Still try to send the censored version even if we couldn't delete.

    # Build the replacement message content.
    author_mention = message.author.mention
    original_text = message.content or ""
    notice = f"{author_mention} (censored media)"
    if oversized_names:
        notice += (
            f"\n⚠️ The following file(s) could not be re-attached because they exceed "
            f"the {cfg.max_file_mb:.0f} MB limit: {', '.join(oversized_names)}"
        )
    send_content = f"{notice}\n{original_text}".strip()

    try:
        if censored_files:
            await message.channel.send(content=send_content, files=censored_files)
        else:
            await message.channel.send(content=send_content)
    except discord.HTTPException:
        log.exception("Failed to send censored message in channel %s", message.channel)
    finally:
        for f in censored_files:
            f.fp.close()

    # Record the censor event to the database for the dashboard audit log.
    await record_censor_event(
        message.guild.id,
        message.channel.id,
        message.author.id,
        censored_filenames + oversized_names,
    )

    # Optional: DM the original author.
    if cfg.dm_on_censor:
        await _dm_author(message.author, str(message.channel))

    # Optional: post to the moderator log channel.
    if cfg.log_channel_id is not None:
        await _post_to_log_channel(message, censored_filenames + oversized_names, cfg.log_channel_id)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


@bot.event
async def on_ready() -> None:
    global _health_server_started
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    log.info("Per-guild settings loaded from database on each message.")
    if HEALTH_PORT and not _health_server_started:
        _health_server_started = True
        asyncio.create_task(_start_health_server(HEALTH_PORT))


@bot.event
async def on_message(message: discord.Message) -> None:
    await _process_message(message)
    await bot.process_commands(message)


@bot.event
async def on_message_edit(before: discord.Message, after: discord.Message) -> None:
    # Only act if the set of media attachments actually changed.
    before_ids = {att.id for att in before.attachments}
    after_ids = {att.id for att in after.attachments}
    if after_ids == before_ids:
        return
    await _process_message(after)
    await bot.process_commands(after)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot.run(TOKEN)
