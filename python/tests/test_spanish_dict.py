# -*- encoding: utf-8 -*-
"""Tests for the Spanish reduced dictionary and head-surgery helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from tools.head_surgery import (  # noqa: E402
    build_index_mapping,
    compute_target_indices,
    read_char_list,
    slice_linear_weights,
)

DICT_PATH = root_dir / "rapidocr" / "models" / "ppocr_keys_es.txt"


def test_spanish_dict_exists_and_nonempty():
    assert DICT_PATH.exists()
    chars = read_char_list(DICT_PATH)
    assert len(chars) > 0
    assert len(chars) < 200


def test_spanish_dict_no_duplicates():
    chars = read_char_list(DICT_PATH)
    assert len(chars) == len(set(chars))


def test_spanish_dict_contains_core_chars():
    chars = set(read_char_list(DICT_PATH))
    for c in "abcdefghijklmnopqrstuvwxyz0123456789":
        assert c in chars
    for c in "áéíóúñÁÉÍÓÚÑ":
        assert c in chars


def test_head_surgery_mapping():
    source = ["a", "b", "c", "d"]
    target = ["b", "d"]
    mapping = build_index_mapping(source, target)
    assert mapping == {"b": 1, "d": 3}
    assert compute_target_indices(source, target) == [1, 3]


def test_head_surgery_mapping_missing_raises():
    with pytest.raises(ValueError):
        build_index_mapping(["a", "b"], ["z"])


def test_slice_linear_weights():
    w = np.arange(12, dtype=np.float32).reshape(4, 3)
    b = np.array([10, 20, 30, 40], dtype=np.float32)
    w2, b2 = slice_linear_weights(w, b, [1, 3])
    assert w2.shape == (2, 3)
    assert b2.shape == (2,)
    np.testing.assert_array_equal(w2[0], w[1])
    np.testing.assert_array_equal(w2[1], w[3])
    np.testing.assert_array_equal(b2, [20, 40])


def test_slice_linear_weights_invalid():
    w = np.arange(12, dtype=np.float32).reshape(4, 3)
    b = np.array([10, 20, 30, 40], dtype=np.float32)
    with pytest.raises(ValueError):
        slice_linear_weights(w, b, [10])
    with pytest.raises(ValueError):
        slice_linear_weights(w.reshape(2, 2, 3), b, [0])

V1_PATH = root_dir / "rapidocr" / "models" / "ppocr_keys_es_v1.txt"


def test_v1_dict_matches_legacy_vocab_size():
    chars = read_char_list(V1_PATH)
    assert len(chars) == 107
    # CTC vocabulary = chars + blank + space
    assert len(chars) + 2 == 109


def test_v2_dict_is_v1_plus_symbols():
    v1 = set(read_char_list(V1_PATH))
    v2 = set(read_char_list(DICT_PATH))
    added = v2 - v1
    assert added == {"<", ">", "_", "Ø", "|", "·", "□", "Φ", "≠"}
    assert v1.issubset(v2)
    assert len(v2) + 2 == 118
