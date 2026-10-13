"""Metal versus CPU text on macOS, and the CPU fallback when Metal is absent."""

import os
import subprocess
import sys

import pytest

from rapidocr import RapidOCR
from rapidocr.output import RapidOCROutput


def _models():
    det = os.environ.get("PPOCR_TEST_DET")
    rec = os.environ.get("PPOCR_TEST_REC")
    dictionary = os.environ.get("PPOCR_TEST_DICT")
    image = os.environ.get("PPOCR_TEST_IMAGE")
    if not all((det, rec, dictionary, image, os.environ.get("PPOCR_LIBRARY"))):
        pytest.skip("set PPOCR_LIBRARY and PPOCR_TEST_* to run native OCR")
    return det, rec, dictionary, image


def _iou(left: str, right: str) -> float:
    def bounds(text):
        values = [int(v) for v in text.split(",")]
        xs, ys = values[0::2], values[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    a, b = bounds(left), bounds(right)
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 1.0


def _run(backend: str) -> subprocess.CompletedProcess:
    # Each backend gets its own process: the GPU driver is chosen once per
    # process.
    script = r"""
import os, sys
from rapidocr import RapidOCR
engine = RapidOCR(params={
    "Det.model_path": os.environ["PPOCR_TEST_DET"],
    "Rec.model_path": os.environ["PPOCR_TEST_REC"],
    "Rec.rec_keys_path": os.environ["PPOCR_TEST_DICT"],
    "Global.use_cls": False,
    "Global.text_score": 0.0,
    "EngineConfig.ppocr_cpp.backend": sys.argv[1],
})
result = engine(os.environ["PPOCR_TEST_IMAGE"])
for text, box in zip(result.txts or (), result.boxes if result.boxes is not None else ()):
    print(text + "\t" + ",".join(str(int(round(float(v)))) for v in box.reshape(-1)))
"""
    return subprocess.run(
        [sys.executable, "-c", script, backend],
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )


def test_metal_option_is_accepted_everywhere():
    _models()
    from rapidocr import _native

    completed = _run("metal")
    assert completed.returncode == 0, completed.stderr
    cpu = _run("cpu")
    assert cpu.returncode == 0, cpu.stderr
    if not _native.metal_available():
        # Linux, or macOS without Metal: CPU fallback, identical output.
        assert completed.stdout == cpu.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal runs on macOS only")
def test_metal_matches_cpu_text():
    _models()
    from rapidocr import _native

    if not _native.metal_available():
        pytest.skip("no Metal device")
    metal = _run("metal")
    cpu = _run("cpu")
    assert metal.returncode == 0, metal.stderr
    assert cpu.returncode == 0, cpu.stderr
    metal_lines = [line.split("\t") for line in metal.stdout.splitlines()]
    cpu_lines = [line.split("\t") for line in cpu.stdout.splitlines()]
    assert [t for t, _ in metal_lines] == [t for t, _ in cpu_lines]
    for (_, left), (_, right) in zip(cpu_lines, metal_lines):
        assert _iou(left, right) >= 0.98, (left, right)


def test_metal_output_type():
    det, rec, dictionary, image = _models()
    engine = RapidOCR(
        params={
            "Det.model_path": det,
            "Rec.model_path": rec,
            "Rec.rec_keys_path": dictionary,
            "Global.use_cls": False,
            "EngineConfig.ppocr_cpp.use_metal": True,
        }
    )
    assert isinstance(engine(image), RapidOCROutput)
