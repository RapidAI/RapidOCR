from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from rapidocr.utils.load_image import LoadImage


@pytest.mark.parametrize("input_type", ["bytes", "pil"])
@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_exif_input_types_match_path(tmp_path, input_type, orientation):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = orientation
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    loader = LoadImage()
    rotations = {1: 0, 3: 2, 6: 3, 8: 1}
    expected = np.rot90(pixels, rotations[orientation])[..., ::-1]
    np.testing.assert_array_equal(loader(path), expected)

    if input_type == "bytes":
        actual = loader(path.read_bytes())
    else:
        with Image.open(path) as pil_image:
            original_pixels = np.array(pil_image)
            actual = loader(pil_image)
            assert pil_image.getexif()[274] == orientation
            np.testing.assert_array_equal(np.array(pil_image), original_pixels)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("input_type", ["bytes", "pil"])
def test_exif_existing_jpeg_matches_path(input_type):
    path = Path(__file__).parent / "test_files" / "img_exif_orientation.jpg"
    loader = LoadImage()
    expected = loader(path)
    if input_type == "bytes":
        actual = loader(path.read_bytes())
    else:
        with Image.open(path) as image:
            actual = loader(image)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("input_type", ["bytes", "pil"])
def test_input_without_exif_preserves_pixels(tmp_path, input_type):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    image = Image.fromarray(pixels)
    path = tmp_path / "unoriented.png"
    image.save(path)
    loader = LoadImage()
    if input_type == "bytes":
        actual = loader(path.read_bytes())
    else:
        actual = loader(image)
    np.testing.assert_array_equal(actual, pixels[..., ::-1])
