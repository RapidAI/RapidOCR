import numpy as np
from PIL import Image
from rapidocr.utils.load_image import LoadImage


def test_grayscale_alpha_blends_with_white():
    img = Image.new("LA", (5, 1))
    img.putdata([(64, 0), (64, 128), (128, 128), (64, 255), (64, 1)])

    result = LoadImage()(img)

    white = Image.new("RGBA", img.size, "white")
    expected = Image.alpha_composite(white, img.convert("RGBA")).convert("RGB")
    np.testing.assert_array_equal(result, np.asarray(expected))
