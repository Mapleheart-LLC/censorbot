# censorbot
A Discord bot that censors images and videos.

CensorBot monitors Discord channels for messages that contain image or video
attachments.  When explicit content is detected the affected regions are
**pixelated** before the original message is deleted and a replacement message
(with the censored media) is posted on behalf of the original author.

## How it works

1. A user posts a message with one or more image / video attachments.
2. CensorBot downloads each attachment and runs it through
   [NudeNet](https://github.com/notAI-tech/NudeNet) – a lightweight
   ONNX-based nudity-detection model.
3. Every bounding box that NudeNet marks as explicit is **pixelated** using
   nearest-neighbour downscaling / upscaling (Pillow for images, OpenCV for
   video frames).
4. The original message is deleted.
5. A new message is posted in the same channel with the censored media
   and a mention of the original author.
6. Optionally, the original author receives a DM explaining the removal.
7. Optionally, a moderation audit embed is posted to a configured log channel.

If NudeNet finds nothing to censor the media is forwarded as-is (still
replacing the original message, so the flow is consistent).

## Requirements

* Python 3.11+
* The packages listed in `requirements.txt` (NudeNet pulls in
  `onnxruntime` and `opencv-python-headless` automatically)

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Mapleheart-LLC/censorbot.git
cd censorbot

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure the bot
cp .env.example .env
# Edit .env and set your DISCORD_TOKEN
```

### Discord bot permissions

The bot requires the following Gateway Intents to be enabled in the
[Discord Developer Portal](https://discord.com/developers/applications):

* **Message Content Intent** – to read attachment metadata
* **Server Members Intent** – to mention the original author

It also needs the following OAuth2 **Bot Permissions**:

| Permission | Reason |
|---|---|
| Read Messages / View Channels | Receive messages |
| Send Messages | Post the censored replacement |
| Attach Files | Re-attach the censored media |
| Manage Messages | Delete the original message |

## Running

```bash
python bot.py
```

## Docker

```bash
# Build and start (reads .env automatically)
docker compose up -d

# Or build the image manually
docker build -t censorbot .
docker run --env-file .env censorbot
```

The `Dockerfile` pre-warms the NudeNet ONNX model at build time so the first
runtime request has no cold-start delay.

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DISCORD_TOKEN` | *(required)* | Your Discord bot token |
| `MONITORED_CHANNELS` | *(empty – all)* | Comma-separated channel IDs to monitor |
| `PIXEL_BLOCK_SIZE` | `20` | Pixel block size used for the pixelation effect |
| `MIN_CONFIDENCE` | `0.0` | Minimum NudeNet detection score (0–1) to censor a region |
| `CENSOR_CLASSES` | *(built-in set)* | Comma-separated NudeNet labels to censor; leave empty for defaults |
| `VIDEO_FRAME_SAMPLE_RATE` | `1` | Run NudeNet every N frames for video; higher = faster, less accurate |
| `MAX_FILE_MB` | `8.0` | Drop censored files larger than this (MB) and warn in the channel |
| `DM_ON_CENSOR` | `true` | Send the original author a DM explaining the removal |
| `LOG_CHANNEL_ID` | *(none)* | Channel ID for a moderator audit log |
| `HEALTH_PORT` | `8080` | Port for the HTTP health-check endpoint (`GET /health`); `0` to disable |

### Built-in censored classes

The default `CENSOR_CLASSES` set targets:

* `BUTTOCKS_EXPOSED`
* `FEMALE_BREAST_EXPOSED`
* `FEMALE_GENITALIA_EXPOSED`
* `MALE_GENITALIA_EXPOSED`
* `ANUS_EXPOSED`
* `FEMALE_GENITALIA_COVERED`
* `FEMALE_BREAST_COVERED`

Override with `CENSOR_CLASSES=LABEL1,LABEL2` to change this list.

## Running the tests

```bash
pytest tests/
```
