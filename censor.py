"""
Image and video censoring utilities.

Images are censored using Pillow (pixelation or Gaussian blur).
Videos are censored by processing each frame with the same technique
via ffmpeg (subprocess), which must be installed on the host system.
"""

import io
import os
import subprocess
import tempfile
from typing import Literal

from PIL import Image, ImageFilter


CensorMode = Literal["pixelate", "blur"]

# Supported MIME type prefixes
IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/gif", "image/webp")
VIDEO_MIME_PREFIXES = ("video/mp4", "video/webm", "video/quicktime", "video/x-matroska")


def is_image(content_type: str) -> bool:
    """Return True if the content type corresponds to a supported image."""
    return content_type.startswith(IMAGE_MIME_PREFIXES)


def is_video(content_type: str) -> bool:
    """Return True if the content type corresponds to a supported video."""
    return content_type.startswith(VIDEO_MIME_PREFIXES)


def censor_image(
    data: bytes,
    mode: CensorMode = "pixelate",
    block_size: int = 20,
    blur_radius: int = 20,
) -> bytes:
    """
    Apply a censoring effect to image data and return the result as PNG bytes.

    Parameters
    ----------
    data:
        Raw image bytes.
    mode:
        ``"pixelate"`` reduces the image to a tiny thumbnail and scales it
        back up so individual pixels become large blocks.  ``"blur"`` applies
        a Gaussian blur instead.
    block_size:
        Size of each pixel block when *mode* is ``"pixelate"``.
    blur_radius:
        Gaussian blur radius when *mode* is ``"blur"``.
    """
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGBA")
        censored = _apply_censor(img, mode=mode, block_size=block_size, blur_radius=blur_radius)
        buf = io.BytesIO()
        censored.save(buf, format="PNG")
        return buf.getvalue()


def censor_video(
    data: bytes,
    input_extension: str = ".mp4",
    mode: CensorMode = "pixelate",
    block_size: int = 20,
    blur_radius: int = 20,
) -> bytes:
    """
    Apply a censoring effect to video data and return the result as MP4 bytes.

    Requires ``ffmpeg`` to be installed and available on ``$PATH``.

    Parameters
    ----------
    data:
        Raw video bytes.
    input_extension:
        File extension (including the dot) of the source video, e.g. ``".mp4"``.
    mode:
        Censoring mode – ``"pixelate"`` or ``"blur"``.
    block_size:
        Pixel block size used when *mode* is ``"pixelate"``.
    blur_radius:
        Gaussian blur radius used when *mode* is ``"blur"``.
    """
    with (
        tempfile.NamedTemporaryFile(suffix=input_extension, delete=False) as in_f,
        tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out_f,
    ):
        in_path = in_f.name
        out_path = out_f.name

    try:
        with open(in_path, "wb") as f:
            f.write(data)

        vf_filter = _build_ffmpeg_vf(mode=mode, block_size=block_size, blur_radius=blur_radius)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", in_path,
            "-vf", vf_filter,
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            out_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {result.returncode}: "
                f"{result.stderr.decode(errors='replace')}"
            )

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_censor(
    img: Image.Image,
    mode: CensorMode,
    block_size: int,
    blur_radius: int,
) -> Image.Image:
    """Apply the chosen censoring effect to a Pillow *Image* and return the result."""
    if mode == "pixelate":
        return _pixelate(img, block_size)
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def _pixelate(img: Image.Image, block_size: int) -> Image.Image:
    """Pixelate *img* by downscaling and then upscaling with nearest-neighbour."""
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    original_size = img.size
    small_w = max(1, original_size[0] // block_size)
    small_h = max(1, original_size[1] // block_size)
    small = img.resize((small_w, small_h), Image.NEAREST)
    return small.resize(original_size, Image.NEAREST)


def _build_ffmpeg_vf(mode: CensorMode, block_size: int, blur_radius: int) -> str:
    """Return a ``-vf`` filter string for ffmpeg that censors each video frame."""
    if mode == "pixelate":
        # Scale down, then scale up with nearest-neighbour for a pixelation effect.
        return (
            f"scale=iw/{block_size}:ih/{block_size}:flags=neighbor,"
            f"scale=iw*{block_size}:ih*{block_size}:flags=neighbor"
        )
    # Gaussian blur using the boxblur filter (no extra libs needed).
    luma_power = max(1, blur_radius // 2)
    return f"boxblur={blur_radius}:{luma_power}"
