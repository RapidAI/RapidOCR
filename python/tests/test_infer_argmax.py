# -*- encoding: utf-8 -*-
# Unit tests for the rec-logits reduction introduced in PR #757.
# These tests deliberately avoid downloading models or requiring a GPU.
import sys
from pathlib import Path

import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from rapidocr.inference_engine.base import InferSession
from rapidocr.utils.utils import import_package


class _ArrayReturningSession(InferSession):
    """Minimal InferSession whose ``__call__`` returns a fixed 3D array."""

    def __init__(self):
        # bypass the abstract base __init__
        class _Cfg:
            pass

        super().__init__(_Cfg())
        self._fake = None

    def setup(self, arr: np.ndarray) -> None:
        self._fake = arr

    def __call__(self, input_content):
        return self._fake

    def have_key(self, key: str = "character") -> bool:
        return False

    def get_character_list(self, key: str = "character"):
        return []


def test_infer_argmax_reduces_over_last_axis():
    arr = np.random.RandomState(0).rand(2, 5, 10).astype(np.float32)
    sess = _ArrayReturningSession()
    sess.setup(arr)

    idx, prob = sess.infer_argmax(np.zeros((1, 1), dtype=np.float32))

    assert np.array_equal(idx, arr.argmax(axis=-1))
    assert np.array_equal(prob, arr.max(axis=-1))


def test_infer_argmax_rejects_under_two_dims():
    arr = np.random.RandomState(1).rand(10).astype(np.float32)
    sess = _ArrayReturningSession()
    sess.setup(arr)

    with pytest.raises(ValueError):
        sess.infer_argmax(np.zeros((1,), dtype=np.float32))


def test_ctc_label_decode_accepts_tuple_and_array():
    character = list("abcdefghij")
    decode = CTCLabelDecode(character=character)
    logits = np.random.RandomState(2).rand(2, 6, len(character)).astype(np.float32)

    out_from_array = decode(logits)
    out_from_tuple = decode((logits.argmax(axis=2), logits.max(axis=2)))

    assert out_from_array == out_from_tuple


@pytest.mark.skipif(not import_package("torch"), reason="torch is not installed")
def test_torch_infer_argmax_matches_base_reduction():
    from rapidocr.inference_engine.pytorch.main import TorchInferSession

    character = list("abcdefghij")

    class _FakeTorchSession(TorchInferSession):
        def __init__(self):
            import torch

            self.device = torch.device("cpu")

            def predictor(x):
                # Deterministic given x so __call__ and infer_argmax agree.
                return torch.full(
                    (x.shape[0], 25, len(character)), float(x.mean().item())
                )

            self.predictor = predictor

    sess = _FakeTorchSession()
    x = np.random.RandomState(3).rand(2, 3, 32, 100).astype(np.float32)

    full = sess(x)  # reduce=False -> (B, T, vocab) ndarray
    idx, prob = sess.infer_argmax(x)  # reduce=True -> (B, T) tuple

    assert np.array_equal(idx, full.argmax(axis=-1))
    assert np.array_equal(prob, full.max(axis=-1))
