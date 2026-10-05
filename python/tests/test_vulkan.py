"""Vulkan versus CPU text, skipped when this process has no compute device."""

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


def _params(det, rec, dictionary, **extra):
    params = {
        "Det.model_path": det,
        "Rec.model_path": rec,
        "Rec.rec_keys_path": dictionary,
        "Global.use_cls": False,
        "Global.text_score": 0.0,
    }
    params.update(extra)
    return params


def test_vulkan_matches_cpu_text():
    det, rec, dictionary, image = _models()
    index = os.environ.get("PPOCR_VULKAN_DEVICE_INDEX")
    if index is None:
        pytest.skip("set PPOCR_VULKAN_DEVICE_INDEX to pin a compute device")
    from rapidocr import backend_info, request_vulkan_device

    request_vulkan_device(int(index))
    info = backend_info()
    if not info["vulkan_compute_available"]:
        pytest.skip("no Vulkan compute device")
    common = _params(det, rec, dictionary)
    cpu = RapidOCR(params={**common, "EngineConfig.ppocr_cpp.backend": "cpu"})
    vulkan = RapidOCR(
        params={
            **common,
            "EngineConfig.ppocr_cpp.use_vulkan": True,
            "EngineConfig.ppocr_cpp.device_index": int(index),
        }
    )
    cpu_result = cpu(image)
    vulkan_result = vulkan(image)
    assert isinstance(cpu_result, RapidOCROutput)
    assert isinstance(vulkan_result, RapidOCROutput)
    assert vulkan_result.txts == cpu_result.txts
    assert vulkan_result.scores is not None and cpu_result.scores is not None
    for left, right in zip(cpu_result.scores, vulkan_result.scores):
        assert abs(float(left) - float(right)) <= 0.05


def test_vulkan_request_falls_back_without_device():
    det, rec, dictionary, image = _models()
    env = os.environ.copy()
    env["VK_ICD_FILENAMES"] = "/dev/null"
    env["VK_DRIVER_FILES"] = "/dev/null"
    env.pop("PPOCR_VULKAN_DEVICE_INDEX", None)
    env.pop("PPOCR_VULKAN_DEVICE_NAME", None)
    script = r"""
import os
from rapidocr import RapidOCR
engine = RapidOCR(params={
    "Det.model_path": os.environ["PPOCR_TEST_DET"],
    "Rec.model_path": os.environ["PPOCR_TEST_REC"],
    "Rec.rec_keys_path": os.environ["PPOCR_TEST_DICT"],
    "Global.use_cls": False,
    "Global.text_score": 0.0,
    "EngineConfig.ppocr_cpp.use_vulkan": True,
    "EngineConfig.ppocr_cpp.device_index": -1,
})
result = engine(os.environ["PPOCR_TEST_IMAGE"])
print("\n".join(result.txts or ()))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    cpu = RapidOCR(params=_params(det, rec, dictionary, **{"EngineConfig.ppocr_cpp.backend": "cpu"}))
    assert completed.stdout.splitlines() == list(cpu(image).txts or ())
