#!/bin/bash
# macOS arm64 wheel check used by cibuildwheel: the shared wheel check, then
# deployment target (macOS 14), Metal compiled in, and hello OCR on Metal.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
bash "$ROOT/python/tests/ci_wheel_check.sh"

python - << 'PY'
import os
import subprocess
from pathlib import Path

import rapidocr
from rapidocr import _native

root = Path(rapidocr.__file__).resolve().parent
libs = sorted(p for p in root.glob("libppocr*.dylib") if p.is_file() and not p.is_symlink())
assert libs, list(root.iterdir())
for lib in libs:
    out = subprocess.run(["vtool", "-show-build", str(lib)], check=True,
                         capture_output=True, text=True).stdout
    print(lib.name, out)
    assert "minos 14.0" in out, out
    arch = subprocess.run(["lipo", "-archs", str(lib)], check=True,
                          capture_output=True, text=True).stdout.split()
    assert arch == ["arm64"], arch

# Metal must be compiled in: linked against Metal.framework and carrying the
# embedded MSL kernels. GitHub's macOS VMs expose no Metal device, so a missing
# device only downgrades the runtime check (backend=metal must then fall back
# to CPU and still read the text).
for lib in libs:
    links = subprocess.run(["otool", "-L", str(lib)], check=True, capture_output=True, text=True).stdout
    assert "Metal.framework" in links, links
    syms = subprocess.run(["nm", "-g", str(lib)], check=True, capture_output=True, text=True).stdout
    assert "kMetalBinaryMsl" in syms and "_ppocr_metal_device_available" in syms, lib
print("metal_built_ok")
has_device = _native.metal_available()
print("metal_device_available", has_device)

from rapidocr import RapidOCR

models = Path(os.environ.get("PPOCR_CI_MODELS", Path.cwd() / "ci-models"))
engine = RapidOCR(params={
    "Det.model_path": str(models / "PP-OCRv6_det_tiny.onnx"),
    "Rec.model_path": str(models / "PP-OCRv6_rec_tiny.onnx"),
    "Rec.rec_keys_path": str(models / "ppocrv6_tiny_dict.txt"),
    "Det.model_type": "tiny",
    "Rec.model_type": "tiny",
    "Global.use_cls": False,
    "EngineConfig.ppocr_cpp.use_metal": True,
})
result = engine("python/tests/data/hello.png")
print("metal hello", result.txts, result.scores)
assert result.txts == ("Hello RapidOCR 123",), result.txts
print("metal_hello_ocr_ok" if has_device else "metal_fallback_hello_ocr_ok")
PY
echo "ci_wheel_check_macos_ok"
