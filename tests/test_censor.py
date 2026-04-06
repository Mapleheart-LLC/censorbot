"""
Unit tests for censor.py.

These tests exercise the public API and internal helpers of the censor
module without requiring a real Discord connection.  NudeNet is exercised
on synthetic (solid-colour) images that contain no explicit content, so
no detections are expected and the functions should return the image
unchanged (but re-encoded as PNG).
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import censor as censor_mod
from censor import (
    EXPLICIT_CLASSES,
    _pixelate_cv2,
    _pixelate_pil,
    censor_image,
    censor_video,
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


def _make_gradient_png(width: int = 64, height: int = 64) -> bytes:
    """Return PNG bytes of a colour-gradient image.

    Each pixel differs from its neighbours so pixelating the image visibly
    changes it, allowing tests to assert that censor_image actually modified
    pixels when a detection is applied.
    """
    img = Image.new("RGBA", (width, height))
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (x * 4 % 256, y * 4 % 256, 100, 255)
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
# censor_image – min_confidence / censor_classes (with mocked detector)
# ---------------------------------------------------------------------------


class TestCensorImageMocked:
    """Tests that mock the NudeNet detector to exercise filtering logic."""

    @patch("censor._get_detector")
    def test_min_confidence_filters_out_low_score(self, mock_get_detector):
        """Detections below min_confidence should not be pixelated."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.3, "box": [0, 0, 64, 64]},
        ]
        mock_get_detector.return_value = mock_detector

        png_data = _make_png(width=64, height=64, color=(42, 84, 168, 255))
        result = censor_image(png_data, min_confidence=0.5)

        original = Image.open(io.BytesIO(png_data)).convert("RGBA")
        censored = Image.open(io.BytesIO(result)).convert("RGBA")
        # Detection was below threshold – image must be unchanged.
        assert original.tobytes() == censored.tobytes()

    @patch("censor._get_detector")
    def test_min_confidence_applies_high_score(self, mock_get_detector):
        """Detections at or above min_confidence should be pixelated."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9, "box": [0, 0, 64, 64]},
        ]
        mock_get_detector.return_value = mock_detector

        png_data = _make_gradient_png(width=64, height=64)
        result = censor_image(png_data, min_confidence=0.5)

        original = Image.open(io.BytesIO(png_data)).convert("RGBA")
        censored = Image.open(io.BytesIO(result)).convert("RGBA")
        # Detection was applied – pixels must differ.
        assert original.tobytes() != censored.tobytes()

    @patch("censor._get_detector")
    def test_empty_censor_classes_skips_all(self, mock_get_detector):
        """An empty censor_classes set should suppress all pixelation."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9, "box": [0, 0, 64, 64]},
        ]
        mock_get_detector.return_value = mock_detector

        png_data = _make_png(width=64, height=64, color=(42, 84, 168, 255))
        result = censor_image(png_data, censor_classes=frozenset())

        original = Image.open(io.BytesIO(png_data)).convert("RGBA")
        censored = Image.open(io.BytesIO(result)).convert("RGBA")
        assert original.tobytes() == censored.tobytes()

    @patch("censor._get_detector")
    def test_custom_censor_classes_applied(self, mock_get_detector):
        """Only classes present in censor_classes should be pixelated."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            # This class is NOT in the custom set – must not be pixelated.
            {"class": "BUTTOCKS_EXPOSED", "score": 0.9, "box": [0, 0, 32, 32]},
            # This class IS in the custom set – must be pixelated.
            {"class": "MALE_GENITALIA_EXPOSED", "score": 0.9, "box": [32, 32, 32, 32]},
        ]
        mock_get_detector.return_value = mock_detector

        png_data = _make_gradient_png(width=64, height=64)
        result = censor_image(
            png_data,
            censor_classes=frozenset({"MALE_GENITALIA_EXPOSED"}),
        )
        censored = Image.open(io.BytesIO(result)).convert("RGBA")
        # At least one detection was applied, so pixels differ somewhere.
        original = Image.open(io.BytesIO(png_data)).convert("RGBA")
        assert original.tobytes() != censored.tobytes()

    @patch("censor._get_detector")
    def test_default_censor_classes_uses_explicit_classes(self, mock_get_detector):
        """When censor_classes=None the default EXPLICIT_CLASSES is used."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9, "box": [0, 0, 64, 64]},
        ]
        mock_get_detector.return_value = mock_detector

        png_data = _make_gradient_png(width=64, height=64)
        result = censor_image(png_data)  # censor_classes defaults to None

        original = Image.open(io.BytesIO(png_data)).convert("RGBA")
        censored = Image.open(io.BytesIO(result)).convert("RGBA")
        assert original.tobytes() != censored.tobytes()


# ---------------------------------------------------------------------------
# censor_video – frame_sample_rate validation
# ---------------------------------------------------------------------------


class TestCensorVideoFrameSampleRate:
    def test_invalid_frame_sample_rate_raises(self):
        """frame_sample_rate < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="frame_sample_rate"):
            censor_video(b"", frame_sample_rate=0)

    def test_negative_frame_sample_rate_raises(self):
        with pytest.raises(ValueError, match="frame_sample_rate"):
            censor_video(b"", frame_sample_rate=-5)


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

