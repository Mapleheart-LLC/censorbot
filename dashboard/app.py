"""
CensorBot Dashboard – FastAPI application.

Provides a web UI for Discord server owners to configure per-guild censoring
settings and review the censor audit log.

Routes
------
GET  /                  → redirect to /guilds or /login
GET  /login             → redirect to Discord OAuth2
GET  /callback          → OAuth2 code exchange; sets session
GET  /logout            → clears session; redirect to /login
GET  /guilds            → list guilds the logged-in user manages
GET  /guilds/{id}       → settings form for a guild
POST /guilds/{id}       → save settings for a guild
GET  /guilds/{id}/log   → last 100 censor events for a guild

Environment variables
---------------------
DISCORD_CLIENT_ID       OAuth2 application client ID.
DISCORD_CLIENT_SECRET   OAuth2 application client secret.
DISCORD_TOKEN           Bot token (used to fetch guild channel lists).
DASHBOARD_SECRET_KEY    Secret used to sign session cookies.
DASHBOARD_URL           Public base URL of this dashboard, e.g.
                        https://dashboard.example.com  (no trailing slash).
                        Defaults to http://localhost:5000.
DATABASE_URL            SQLAlchemy database URL (default: sqlite:///censorbot.db).
"""

import asyncio
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# Allow importing db / settings from the repo root when running the dashboard
# as a standalone process (e.g. `uvicorn dashboard.app:app`).
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from db import CensorEvent, GuildSettings, SessionFactory, init_db  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REQUIRED = object()


def _env(name: str, default=_REQUIRED) -> str:
    val = os.environ.get(name, "")
    if not val:
        if default is _REQUIRED:
            raise RuntimeError(f"Required environment variable {name!r} is not set.")
        return str(default)
    return val


DISCORD_CLIENT_ID: str = _env("DISCORD_CLIENT_ID")
DISCORD_CLIENT_SECRET: str = _env("DISCORD_CLIENT_SECRET")
DISCORD_BOT_TOKEN: str = _env("DISCORD_TOKEN")
DASHBOARD_SECRET_KEY: str = _env("DASHBOARD_SECRET_KEY")
DASHBOARD_URL: str = _env("DASHBOARD_URL", "http://localhost:5000").rstrip("/")

DISCORD_API = "https://discord.com/api/v10"
OAUTH2_REDIRECT_URI = f"{DASHBOARD_URL}/callback"

# Discord permission bit for MANAGE_GUILD
_MANAGE_GUILD_BIT = 0x00000020

# All NudeNet class labels (for the censor-classes checkbox list).
NUDENET_CLASSES: list[str] = sorted([
    "ANUS_COVERED",
    "ANUS_EXPOSED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "BELLY_COVERED",
    "BELLY_EXPOSED",
    "BUTTOCKS_COVERED",
    "BUTTOCKS_EXPOSED",
    "FACE_FEMALE",
    "FACE_MALE",
    "FEET_COVERED",
    "FEET_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
])

# The default subset that matches censor.EXPLICIT_CLASSES.
DEFAULT_CENSOR_CLASSES: frozenset[str] = frozenset({
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
})

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="CensorBot Dashboard", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=DASHBOARD_SECRET_KEY,
    max_age=60 * 60 * 24 * 7,  # 7 days
    https_only=DASHBOARD_URL.startswith("https"),
)

_BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(_BASE / "templates"))

import json as _json_module  # noqa: E402

templates.env.filters["from_json"] = _json_module.loads


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _user_manages_guild(guilds: list[dict], guild_id: int) -> bool:
    """Return True if any guild in *guilds* matches *guild_id* with MANAGE_GUILD."""
    for g in guilds:
        if int(g["id"]) == guild_id and (int(g.get("permissions", 0)) & _MANAGE_GUILD_BIT):
            return True
    return False


def _guild_name(guilds: list[dict], guild_id: int) -> str:
    for g in guilds:
        if int(g["id"]) == guild_id:
            return g.get("name", str(guild_id))
    return str(guild_id)


async def _fresh_managed_guilds(access_token: str) -> list[dict]:
    """Re-fetch the user's guilds from Discord and filter to MANAGE_GUILD."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{DISCORD_API}/users/@me/guilds",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
    return [
        g for g in resp.json()
        if int(g.get("permissions", 0)) & _MANAGE_GUILD_BIT
    ]


# ---------------------------------------------------------------------------
# Routes – auth
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def index(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/guilds")
    return RedirectResponse("/login")


@app.get("/login", include_in_schema=False)
async def login(request: Request):
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    params = urlencode({
        "client_id": DISCORD_CLIENT_ID,
        "redirect_uri": OAUTH2_REDIRECT_URI,
        "response_type": "code",
        "scope": "identify guilds",
        "state": state,
        "prompt": "none",
    })
    return RedirectResponse(f"https://discord.com/oauth2/authorize?{params}")


@app.get("/callback", include_in_schema=False)
async def callback(request: Request, code: str = "", state: str = ""):
    saved_state = request.session.pop("oauth_state", None)
    if not state or state != saved_state:
        return HTMLResponse("Invalid or missing OAuth2 state – please try logging in again.", status_code=400)
    if not code:
        return HTMLResponse("Missing authorization code.", status_code=400)

    async with httpx.AsyncClient(timeout=10) as client:
        # Exchange code for access token.
        token_resp = await client.post(
            f"{DISCORD_API}/oauth2/token",
            data={
                "client_id": DISCORD_CLIENT_ID,
                "client_secret": DISCORD_CLIENT_SECRET,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": OAUTH2_REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if token_resp.status_code != 200:
            return HTMLResponse("Failed to exchange OAuth2 code. Please try again.", status_code=502)
        token_data = token_resp.json()
        access_token: str = token_data.get("access_token", "")
        if not access_token:
            return HTMLResponse("Discord did not return an access token.", status_code=502)

        headers = {"Authorization": f"Bearer {access_token}"}

        # Fetch user identity and guild list in parallel.
        user_resp, guilds_resp = await asyncio.gather(
            client.get(f"{DISCORD_API}/users/@me", headers=headers),
            client.get(f"{DISCORD_API}/users/@me/guilds", headers=headers),
        )

    if user_resp.status_code != 200 or guilds_resp.status_code != 200:
        return HTMLResponse("Failed to fetch user data from Discord.", status_code=502)

    request.session["user"] = user_resp.json()
    request.session["access_token"] = access_token
    request.session["guilds"] = guilds_resp.json()
    return RedirectResponse("/guilds", status_code=303)


@app.get("/logout", include_in_schema=False)
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")


# ---------------------------------------------------------------------------
# Routes – guilds list
# ---------------------------------------------------------------------------


@app.get("/guilds", include_in_schema=False)
async def list_guilds(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login")

    guilds: list[dict] = request.session.get("guilds", [])
    managed = [g for g in guilds if int(g.get("permissions", 0)) & _MANAGE_GUILD_BIT]

    return templates.TemplateResponse("guilds.html", {
        "request": request,
        "user": user,
        "guilds": managed,
    })


# ---------------------------------------------------------------------------
# Routes – per-guild settings
# ---------------------------------------------------------------------------


@app.get("/guilds/{guild_id}", include_in_schema=False)
async def guild_settings_page(request: Request, guild_id: int):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login")

    guilds: list[dict] = request.session.get("guilds", [])
    if not _user_manages_guild(guilds, guild_id):
        raise HTTPException(status_code=403, detail="You do not have MANAGE_GUILD permission in this server.")

    # Load settings row (may be None if the guild has never been configured).
    with SessionFactory() as session:
        row: Optional[GuildSettings] = session.get(GuildSettings, guild_id)

    # Determine which censor classes are active for display.
    if row is not None and row.censor_classes.strip():
        active_classes = frozenset(c.strip() for c in row.censor_classes.split(",") if c.strip())
    else:
        active_classes = DEFAULT_CENSOR_CLASSES

    # Determine monitored channels for display.
    if row is not None and row.monitored_channels.strip():
        active_channels = [
            int(c.strip()) for c in row.monitored_channels.split(",") if c.strip()
        ]
    else:
        active_channels = []

    # Fetch text channels for this guild using the bot token.
    text_channels: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            ch_resp = await client.get(
                f"{DISCORD_API}/guilds/{guild_id}/channels",
                headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"},
            )
        if ch_resp.status_code == 200:
            text_channels = sorted(
                [c for c in ch_resp.json() if isinstance(c, dict) and c.get("type") == 0],
                key=lambda c: c.get("position", 0),
            )
    except Exception:
        pass  # channels just won't be shown by name

    saved = request.query_params.get("saved") == "1"

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "user": user,
        "guild_id": guild_id,
        "guild_name": _guild_name(guilds, guild_id),
        "row": row,
        "active_classes": active_classes,
        "active_channels": active_channels,
        "nudenet_classes": NUDENET_CLASSES,
        "default_censor_classes": DEFAULT_CENSOR_CLASSES,
        "text_channels": text_channels,
        "saved": saved,
    })


@app.post("/guilds/{guild_id}", include_in_schema=False)
async def save_guild_settings(
    request: Request,
    guild_id: int,
    pixel_block_size: Annotated[int, Form()] = 20,
    min_confidence: Annotated[float, Form()] = 0.0,
    video_frame_sample_rate: Annotated[int, Form()] = 1,
    max_file_mb: Annotated[float, Form()] = 8.0,
    dm_on_censor: Annotated[Optional[str], Form()] = None,
    log_channel_id: Annotated[Optional[str], Form()] = None,
):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=303)

    access_token: str = request.session.get("access_token", "")
    if not access_token:
        return RedirectResponse("/login", status_code=303)

    # Re-verify the user still has MANAGE_GUILD (fresh Discord API call).
    try:
        fresh_guilds = await _fresh_managed_guilds(access_token)
    except Exception:
        raise HTTPException(status_code=502, detail="Could not verify Discord permissions.")

    if not _user_manages_guild(fresh_guilds, guild_id):
        raise HTTPException(status_code=403, detail="You do not have MANAGE_GUILD permission.")

    # Parse multi-value form fields that FastAPI doesn't handle via Form() alone.
    form_data = await request.form()
    monitored_channels_raw: list[str] = form_data.getlist("monitored_channels")
    censor_classes_raw: list[str] = form_data.getlist("censor_classes")

    monitored_channels_str = ",".join(c for c in monitored_channels_raw if c.strip())
    censor_classes_str = ",".join(c for c in censor_classes_raw if c.strip())

    with SessionFactory() as session:
        with session.begin():
            row: Optional[GuildSettings] = session.get(GuildSettings, guild_id)
            if row is None:
                row = GuildSettings(guild_id=guild_id)
                session.add(row)
            row.monitored_channels = monitored_channels_str
            row.pixel_block_size = max(1, int(pixel_block_size))
            row.min_confidence = max(0.0, min(1.0, float(min_confidence)))
            row.censor_classes = censor_classes_str
            row.video_frame_sample_rate = max(1, int(video_frame_sample_rate))
            row.max_file_mb = max(0.1, float(max_file_mb))
            row.dm_on_censor = dm_on_censor is not None
            row.log_channel_id = (
                int(log_channel_id) if log_channel_id and log_channel_id.strip().isdigit() else None
            )

    return RedirectResponse(f"/guilds/{guild_id}?saved=1", status_code=303)


# ---------------------------------------------------------------------------
# Routes – censor event log
# ---------------------------------------------------------------------------


@app.get("/guilds/{guild_id}/log", include_in_schema=False)
async def guild_log(request: Request, guild_id: int):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login")

    guilds: list[dict] = request.session.get("guilds", [])
    if not _user_manages_guild(guilds, guild_id):
        raise HTTPException(status_code=403, detail="You do not have MANAGE_GUILD permission.")

    with SessionFactory() as session:
        events: list[CensorEvent] = (
            session.query(CensorEvent)
            .filter_by(guild_id=guild_id)
            .order_by(CensorEvent.ts.desc())
            .limit(100)
            .all()
        )

    return templates.TemplateResponse("events.html", {
        "request": request,
        "user": user,
        "guild_id": guild_id,
        "guild_name": _guild_name(guilds, guild_id),
        "events": events,
    })
