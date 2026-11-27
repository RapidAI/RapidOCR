import numpy as np
import pytest

from rapidocr.ch_ppocr_rec.main import TextRecognizer


class _Stub:
    """Minimal object exposing only the chunking helpers."""

    def __init__(self, chunk=240, above=800):
        self.split_wide_crops = True
        self.split_above = above
        self.split_chunk = chunk
        self.split_blocks = False
        self.split_block_min_height = 60

    _chunk_crops = TextRecognizer._chunk_crops
    _row_bands = staticmethod(TextRecognizer._row_bands)
    _split_wide_crops = TextRecognizer._split_wide_crops
    _join_split = staticmethod(TextRecognizer._join_split)


def _line(width: int, height: int = 32) -> np.ndarray:
    """White strip with a few dark glyph blocks, so ink columns exist."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    for x in range(10, width - 10, 40):
        img[8:-8, x : x + 20] = 0
    return img


def test_narrow_crop_is_not_split():
    stub = _Stub()
    img = _line(500)
    expanded, owners, gaps = stub._split_wide_crops([img])
    assert len(expanded) == 1
    assert owners == [0]
    assert gaps == [0]


def test_wide_crop_is_split_into_readable_pieces():
    stub = _Stub(chunk=240, above=800)
    img = _line(1200)
    expanded, owners, gaps = stub._split_wide_crops([img])
    assert len(expanded) > 1
    assert all(piece.shape[1] <= 240 * 1.2 for piece in expanded)
    assert owners == [0] * len(expanded)
    assert len(gaps) == len(expanded)


def test_split_covers_full_width():
    stub = _Stub(chunk=240, above=800)
    img = _line(1000)
    pieces = list(stub._chunk_crops(img))
    assert sum(p.shape[1] for p, _ in pieces) == img.shape[1]


def test_join_inserts_space_on_word_gap():
    owners = [0, 0, 1]
    gaps = [0, 12, 0]
    results = [("hola", 0.9), ("mundo", 0.8), ("otra", 0.7)]
    texts, scores = _Stub._join_split(results, owners, gaps, 2)
    assert texts == ("hola mundo", "otra")
    assert scores[0] == pytest.approx(0.85)
    assert scores[1] == pytest.approx(0.7)


def test_join_without_space_on_tight_cut():
    # Machine-readable zones and identifiers have no word gaps.
    owners = [0, 0]
    gaps = [0, 1]
    results = [("I<PER123", 0.9), ("4567<8", 0.9)]
    texts, _ = _Stub._join_split(results, owners, gaps, 1)
    assert texts == ("I<PER1234567<8",)


def test_disabled_split_keeps_original_list():
    stub = _Stub()
    stub.split_wide_crops = False
    img = _line(1200)
    expanded, owners, gaps = stub._split_wide_crops([img])
    assert len(expanded) == 1
    assert owners == [0]
    assert gaps == [0]


def test_row_bands_splits_two_lines():
    img = np.full((90, 200, 3), 255, dtype=np.uint8)
    img[10:30, 10:190] = 0
    img[60:80, 10:190] = 0
    bands = _Stub._row_bands(img)
    assert len(bands) == 2
    assert bands[0][0] < 30 and bands[1][0] > 50


def test_multi_line_block_is_split_when_enabled():
    stub = _Stub(chunk=240, above=800)
    stub.split_blocks = True
    img = np.full((90, 300, 3), 255, dtype=np.uint8)
    img[10:30, 10:290] = 0
    img[60:80, 10:290] = 0
    expanded, owners, gaps = stub._split_wide_crops([img])
    assert len(expanded) == 2
    assert owners == [0, 0]
