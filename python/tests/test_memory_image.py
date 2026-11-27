import ctypes
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from rapidocr.utils.load_image import LoadImage, MemoryImage


def load_from_memory(loader, encoded):
    buffer = ctypes.create_string_buffer(encoded)
    return loader(MemoryImage(ctypes.addressof(buffer), len(encoded)))


@pytest.mark.parametrize("mode", ["RGBA", "P"])
def test_memory_image_preserves_transparent_background(mode):
    image = Image.new("RGBA", (3, 3), (0, 0, 0, 0))
    image.putpixel((1, 1), (10, 20, 30, 255))
    if mode == "P":
        image = image.convert("P", palette=Image.Palette.ADAPTIVE)
    encoded = BytesIO()
    image.save(encoded, format="PNG")

    actual = load_from_memory(LoadImage(), encoded.getvalue())

    expected = np.full((3, 3, 3), 255, dtype=np.uint8)
    expected[1, 1] = [30, 20, 10]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "alpha, expected_bgr",
    [(0, [255, 255, 255]), (128, [142, 137, 132]), (255, [30, 20, 10])],
)
def test_memory_image_composites_alpha(alpha, expected_bgr):
    image = Image.new("RGBA", (2, 3), (10, 20, 30, alpha))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    loader = LoadImage()

    actual = load_from_memory(loader, encoded.getvalue())

    np.testing.assert_array_equal(
        actual, np.full((3, 2, 3), expected_bgr, dtype=np.uint8)
    )
    np.testing.assert_array_equal(actual, loader(encoded.getvalue()))


@pytest.mark.parametrize(
    "mode, color, expected_bgr",
    [("RGB", (10, 20, 30), [30, 20, 10]), ("L", 40, [40, 40, 40])],
)
def test_memory_image_preserves_opaque_pixels(mode, color, expected_bgr):
    image = Image.new(mode, (2, 3), color)
    encoded = BytesIO()
    image.save(encoded, format="PNG")

    actual = load_from_memory(LoadImage(), encoded.getvalue())

    np.testing.assert_array_equal(
        actual, np.full((3, 2, 3), expected_bgr, dtype=np.uint8)
    )


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_memory_image_matches_oriented_encoded_bytes(orientation):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = orientation
    encoded = BytesIO()
    image.save(encoded, format="PNG", exif=exif)
    loader = LoadImage()

    actual = load_from_memory(loader, encoded.getvalue())

    rotations = {1: 0, 3: 2, 6: 3, 8: 1}
    expected = np.rot90(pixels, rotations[orientation])[..., ::-1]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, loader(encoded.getvalue()))
