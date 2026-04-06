"""
Image and video censoring utilities backed by NudeNet.

NudeNet detects explicit body regions in images and video frames; the
detected bounding boxes are then pixelated using Pillow (images) or
OpenCV (video frames).  If no explicit content is detected the media is
returned unchanged.
"""

import io
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
from nudenet import NudeDetector
from PIL import Image


# ---------------------------------------------------------------------------
# Supported MIME types
# ---------------------------------------------------------------------------

IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/gif", "image/webp")
VIDEO_MIME_PREFIXES = ("video/mp4", "video/webm", "video/quicktime", "video/x-matroska")


def is_image(content_type: str) -> bool:
    """Return True if *content_type* corresponds to a supported image format."""
    return content_type.startswith(IMAGE_MIME_PREFIXES)


def is_video(content_type: str) -> bool:
    """Return True if *content_type* corresponds to a supported video format."""
    return content_type.startswith(VIDEO_MIME_PREFIXES)


# ---------------------------------------------------------------------------
# NudeNet classes that should be censored
# ---------------------------------------------------------------------------

EXPLICIT_CLASSES: frozenset[str] = frozenset(
    {
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "FEMALE_GENITALIA_COVERED",
        "FEMALE_BREAST_COVERED",
    }
)

# ---------------------------------------------------------------------------
# Singleton detector – loaded once per process
# ---------------------------------------------------------------------------

_detector: Optional[NudeDetector] = None


def _get_detector() -> NudeDetector:
    global _detector
    if _detector is None:
        _detector = NudeDetector()
    return _detector


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def censor_image(
    data: bytes,
    block_size: int = 20,
    min_confidence: float = 0.0,
    censor_classes: Optional[frozenset[str]] = None,
) -> bytes:
    """
    Detect explicit regions in *data* with NudeNet and pixelate them.

    Parameters
    ----------
    data:
        Raw image bytes (JPEG, PNG, GIF, WEBP, …).
    block_size:
        Size of each pixel block used for pixelation.
    min_confidence:
        Minimum NudeNet detection score (0–1) required before a region is
        pixelated.  Detections with a score below this threshold are ignored.
        Defaults to ``0.0`` (accept all detections).
    censor_classes:
        Set of NudeNet class labels to censor.  Defaults to ``None``, which
        uses the built-in :data:`EXPLICIT_CLASSES` set.  Pass an empty
        ``frozenset()`` to disable censoring entirely.

    Returns
    -------
    PNG bytes with explicit regions pixelated.  If nothing is detected
    the image is returned (as PNG) without any modifications.
    """
    effective_classes = EXPLICIT_CLASSES if censor_classes is None else censor_classes
    detector = _get_detector()
    detections = detector.detect(data)

    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGBA")
        for det in detections:
            if det["class"] not in effective_classes:
                continue
            if det.get("score", 1.0) < min_confidence:
                continue
            x, y, w, h = det["box"]
            # Guard against zero-size boxes
            if w <= 0 or h <= 0:
                continue
            region = img.crop((x, y, x + w, y + h))
            img.paste(_pixelate_pil(region, block_size), (x, y))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def censor_video(
    data: bytes,
    input_extension: str = ".mp4",
    block_size: int = 20,
    min_confidence: float = 0.0,
    censor_classes: Optional[frozenset[str]] = None,
    frame_sample_rate: int = 1,
) -> bytes:
    """
    Detect explicit regions per frame with NudeNet and pixelate them.

    Parameters
    ----------
    data:
        Raw video bytes.
    input_extension:
        File extension (including the dot) of the source video, e.g. ``".mp4"``.
    block_size:
        Pixel block size used for pixelation.
    min_confidence:
        Minimum NudeNet detection score (0–1) required before a region is
        pixelated.  Defaults to ``0.0`` (accept all detections).
    censor_classes:
        Set of NudeNet class labels to censor.  Defaults to ``None`` which
        uses the built-in :data:`EXPLICIT_CLASSES` set.
    frame_sample_rate:
        Run NudeNet detection on every *Nth* frame only; bounding boxes from
        the last sampled frame are reused for intermediate frames.  Higher
        values trade accuracy for speed on long videos.  ``1`` (the default)
        means every frame is analysed.

    Returns
    -------
    MP4 bytes with explicit regions pixelated on every frame.
    """
    if frame_sample_rate < 1:
        raise ValueError("frame_sample_rate must be at least 1")

    effective_classes = EXPLICIT_CLASSES if censor_classes is None else censor_classes
    detector = _get_detector()

    in_f = tempfile.NamedTemporaryFile(suffix=input_extension, delete=False)
    out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    in_path = in_f.name
    out_path = out_f.name
    in_f.close()
    out_f.close()

    try:
        with open(in_path, "wb") as f:
            f.write(data)

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_idx = 0
        # Carries the boxes (and metadata) from the most-recently sampled frame
        # forward to intermediate frames so they still get censored.
        last_active_dets: list[dict] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_sample_rate == 0:
                # NudeNet accepts numpy arrays (BGR is fine – it converts internally)
                raw_dets = detector.detect(frame)
                last_active_dets = [
                    det
                    for det in raw_dets
                    if det["class"] in effective_classes
                    and det.get("score", 1.0) >= min_confidence
                ]

            for det in last_active_dets:
                x, y, w, h = det["box"]
                if w <= 0 or h <= 0:
                    continue
                region = frame[y : y + h, x : x + w]
                if region.size > 0:
                    frame[y : y + h, x : x + w] = _pixelate_cv2(region, block_size)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pixelate_pil(img: Image.Image, block_size: int) -> Image.Image:
    """Pixelate a Pillow *Image* by downscaling then upscaling with nearest-neighbour."""
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    orig_w, orig_h = img.size
    small_w = max(1, orig_w // block_size)
    small_h = max(1, orig_h // block_size)
    small = img.resize((small_w, small_h), Image.Resampling.NEAREST)
    return small.resize((orig_w, orig_h), Image.Resampling.NEAREST)


def _pixelate_cv2(region: np.ndarray, block_size: int) -> np.ndarray:
    """Pixelate an OpenCV frame region (numpy array) using nearest-neighbour scaling."""
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    h, w = region.shape[:2]
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
