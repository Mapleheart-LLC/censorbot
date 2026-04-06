"""
CensorBot – a Discord bot that censors images and videos.

For every message that contains image or video attachments the bot will:
  1. Download each attachment.
  2. Apply a configurable censoring effect (pixelation or blur).
  3. Delete the original message.
  4. Re-post the message content together with the censored media.

Configuration is loaded from a ``.env`` file (see ``.env.example``).

Required environment variable
------------------------------
DISCORD_TOKEN   Discord bot token.

Optional environment variables
-------------------------------
MONITORED_CHANNELS   Comma-separated list of channel IDs to watch.
                     Leave empty (the default) to watch every channel.
CENSOR_MODE          ``pixelate`` (default) or ``blur``.
PIXEL_BLOCK_SIZE     Integer block size for pixelation (default: 20).
BLUR_RADIUS          Integer blur radius (default: 20).
"""

import io
import logging
import os
from pathlib import PurePosixPath

import discord
from discord.ext import commands
from dotenv import load_dotenv

from censor import (
    CensorMode,
    censor_image,
    censor_video,
    is_image,
    is_video,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("censorbot")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOKEN: str = os.environ["DISCORD_TOKEN"]

_raw_channels = os.getenv("MONITORED_CHANNELS", "").strip()
MONITORED_CHANNELS: set[int] = (
    {int(c.strip()) for c in _raw_channels.split(",") if c.strip()}
    if _raw_channels
    else set()
)

CENSOR_MODE: CensorMode = os.getenv("CENSOR_MODE", "pixelate").lower()  # type: ignore[assignment]
if CENSOR_MODE not in ("pixelate", "blur"):
    raise ValueError(f"Invalid CENSOR_MODE: {CENSOR_MODE!r}. Must be 'pixelate' or 'blur'.")

PIXEL_BLOCK_SIZE: int = int(os.getenv("PIXEL_BLOCK_SIZE", "20"))
BLUR_RADIUS: int = int(os.getenv("BLUR_RADIUS", "20"))

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


@bot.event
async def on_ready() -> None:
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    if MONITORED_CHANNELS:
        log.info("Monitoring channels: %s", MONITORED_CHANNELS)
    else:
        log.info("Monitoring all channels.")


@bot.event
async def on_message(message: discord.Message) -> None:
    # Ignore messages from bots (including ourselves) to prevent loops.
    if message.author.bot:
        return

    # Honour channel allowlist when configured.
    if MONITORED_CHANNELS and message.channel.id not in MONITORED_CHANNELS:
        await bot.process_commands(message)
        return

    media_attachments = [
        att
        for att in message.attachments
        if att.content_type and (is_image(att.content_type) or is_video(att.content_type))
    ]

    if not media_attachments:
        await bot.process_commands(message)
        return

    censored_files: list[discord.File] = []

    for attachment in media_attachments:
        try:
            raw_data = await attachment.read()
            content_type = attachment.content_type or ""
            filename = attachment.filename

            if is_image(content_type):
                censored_data = censor_image(
                    raw_data,
                    mode=CENSOR_MODE,
                    block_size=PIXEL_BLOCK_SIZE,
                    blur_radius=BLUR_RADIUS,
                )
                # Always output as PNG.
                stem = PurePosixPath(filename).stem
                out_name = f"{stem}_censored.png"
            else:
                ext = PurePosixPath(filename).suffix or ".mp4"
                censored_data = censor_video(
                    raw_data,
                    input_extension=ext,
                    mode=CENSOR_MODE,
                    block_size=PIXEL_BLOCK_SIZE,
                    blur_radius=BLUR_RADIUS,
                )
                stem = PurePosixPath(filename).stem
                out_name = f"{stem}_censored.mp4"

            censored_files.append(discord.File(io.BytesIO(censored_data), filename=out_name))
            log.info(
                "Censored %s from %s in channel %s",
                filename,
                message.author,
                message.channel,
            )
        except Exception:
            log.exception("Failed to censor attachment %s", attachment.filename)

    if not censored_files:
        # Nothing was successfully censored – leave the original message alone.
        await bot.process_commands(message)
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
    send_content = f"{notice}\n{original_text}".strip()

    try:
        await message.channel.send(content=send_content, files=censored_files)
    except discord.HTTPException:
        log.exception(
            "Failed to send censored message in channel %s", message.channel
        )
    finally:
        for f in censored_files:
            f.fp.close()

    await bot.process_commands(message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot.run(TOKEN)
