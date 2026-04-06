"""
Unit tests for censor.py.

These tests exercise the public API and internal helpers of the censor
module without requiring a real Discord connection.  NudeNet is exercised
on synthetic (solid-colour) images that contain no explicit content, so
no detections are expected and the functions should return the image
unchanged (but re-encoded as PNG).
"""

import io

import numpy as np
import pytest
from PIL import Image

from censor import (
    EXPLICIT_CLASSES,
    _pixelate_cv2,
    _pixelate_pil,
    censor_image,
    is_image,
    is_video,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg(width: int = 64, height: int = 64, color=(200, 100, 50)) -> bytes:
    """Return JPEG bytes of a solid-colour image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png(width: int = 64, height: int = 64, color=(50, 150, 250, 255)) -> bytes:
    """Return PNG bytes of a solid-colour RGBA image."""
    img = Image.new("RGBA", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# MIME-type helpers
# ---------------------------------------------------------------------------


class TestIsImage:
    def test_jpeg(self):
        assert is_image("image/jpeg")

    def test_png(self):
        assert is_image("image/png")

    def test_gif(self):
        assert is_image("image/gif")

    def test_webp(self):
        assert is_image("image/webp")

    def test_mp4_is_not_image(self):
        assert not is_image("video/mp4")

    def test_empty_is_not_image(self):
        assert not is_image("")


class TestIsVideo:
    def test_mp4(self):
        assert is_video("video/mp4")

    def test_webm(self):
        assert is_video("video/webm")

    def test_quicktime(self):
        assert is_video("video/quicktime")

    def test_mkv(self):
        assert is_video("video/x-matroska")

    def test_png_is_not_video(self):
        assert not is_video("image/png")

    def test_empty_is_not_video(self):
        assert not is_video("")


# ---------------------------------------------------------------------------
# _pixelate_pil
# ---------------------------------------------------------------------------


class TestPixelatePil:
    def test_output_size_matches_input(self):
        img = Image.new("RGBA", (80, 60), color=(255, 0, 0, 255))
        result = _pixelate_pil(img, block_size=10)
        assert result.size == (80, 60)

    def test_block_size_1_is_identity(self):
        """block_size=1 should leave the image pixel-for-pixel identical."""
        img = Image.new("RGBA", (40, 40), color=(10, 20, 30, 255))
        result = _pixelate_pil(img, block_size=1)
        assert result.tobytes() == img.tobytes()

    def test_block_size_larger_than_image(self):
        """When block_size > image dimension the image should still be returned."""
        img = Image.new("RGBA", (5, 5), color=(100, 200, 50, 255))
        result = _pixelate_pil(img, block_size=100)
        assert result.size == (5, 5)

    def test_invalid_block_size_raises(self):
        img = Image.new("RGBA", (20, 20))
        with pytest.raises(ValueError):
            _pixelate_pil(img, block_size=0)

    def test_returns_pillow_image(self):
        img = Image.new("RGBA", (30, 30))
        result = _pixelate_pil(img, block_size=5)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# _pixelate_cv2
# ---------------------------------------------------------------------------


class TestPixelateCv2:
    def test_output_shape_matches_input(self):
        region = np.zeros((60, 80, 3), dtype=np.uint8)
        result = _pixelate_cv2(region, block_size=10)
        assert result.shape == (60, 80, 3)

    def test_block_size_1_preserves_values(self):
        region = np.full((20, 20, 3), fill_value=128, dtype=np.uint8)
        result = _pixelate_cv2(region, block_size=1)
        np.testing.assert_array_equal(result, region)

    def test_block_size_larger_than_region(self):
        region = np.zeros((4, 4, 3), dtype=np.uint8)
        result = _pixelate_cv2(region, block_size=100)
        assert result.shape == (4, 4, 3)

    def test_invalid_block_size_raises(self):
        region = np.zeros((20, 20, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            _pixelate_cv2(region, block_size=0)

    def test_returns_numpy_array(self):
        region = np.zeros((30, 30, 3), dtype=np.uint8)
        result = _pixelate_cv2(region, block_size=5)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# censor_image
# ---------------------------------------------------------------------------


class TestCensorImage:
    """
    These tests use synthetic images with no nudity so NudeNet detects
    nothing and the image is returned as PNG without any region modifications.
    """

    def test_returns_bytes(self):
        result = censor_image(_make_jpeg())
        assert isinstance(result, bytes)

    def test_returns_valid_png(self):
        result = censor_image(_make_jpeg())
        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"

    def test_output_dimensions_match_input(self):
        result = censor_image(_make_jpeg(width=100, height=80))
        img = Image.open(io.BytesIO(result))
        assert img.size == (100, 80)

    def test_accepts_png_input(self):
        result = censor_image(_make_png())
        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"

    def test_custom_block_size(self):
        """A different block_size should still return a valid PNG."""
        result = censor_image(_make_jpeg(), block_size=10)
        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"

    def test_no_modifications_on_clean_image(self):
        """Clean images (no detections) should come back with the same pixels.
        PNG input is used here because JPEG is lossy; re-encoding JPEG through
        Pillow would alter pixel values even without any region modifications.
        """
        png_data = _make_png(color=(42, 84, 168, 255))
        result = censor_image(png_data, block_size=10)

        original = Image.open(io.BytesIO(png_data)).convert("RGBA")
        censored = Image.open(io.BytesIO(result)).convert("RGBA")

        assert original.size == censored.size
        assert original.tobytes() == censored.tobytes()


# ---------------------------------------------------------------------------
# EXPLICIT_CLASSES constant
# ---------------------------------------------------------------------------


class TestExplicitClasses:
    def test_is_frozenset(self):
        assert isinstance(EXPLICIT_CLASSES, frozenset)

    def test_contains_expected_labels(self):
        for label in (
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "ANUS_EXPOSED",
        ):
            assert label in EXPLICIT_CLASSES

    def test_covered_labels_included(self):
        assert "FEMALE_BREAST_COVERED" in EXPLICIT_CLASSES
        assert "FEMALE_GENITALIA_COVERED" in EXPLICIT_CLASSES
