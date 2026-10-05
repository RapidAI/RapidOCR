#!/bin/bash
# Wheel install check used by cibuildwheel: import, ISA dispatch, hello OCR, pytest.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

python - << 'PY'
import platform
import rapidocr

info = rapidocr.cpu_info()
print(rapidocr.__version__, info)
isa = info["active_isa"]
assert isa in {"scalar", "avx2", "avx512", "neon"}, info
machine = platform.machine().lower()
if machine in {"x86_64", "amd64"}:
    assert info["avx2_compiled"] and info["avx512_compiled"], info
    assert isa in {"avx2", "avx512"}, info
    if info["avx512"]:
        assert info["avx2"]
elif machine in {"aarch64", "arm64"}:
    assert info["neon_compiled"] and info["neon"] and isa == "neon", info
else:
    raise SystemExit(f"unexpected machine {machine}")
print("isa_dispatch_ok", isa)
PY

if [[ ! -f "$ROOT/ci-models/PP-OCRv6_det_tiny.onnx" ]]; then
  python "$ROOT/python/tests/fetch_ci_models.py"
fi

LIB=$(python - << 'PY'
import rapidocr
from pathlib import Path
root = Path(rapidocr.__file__).resolve().parent
for name in ("libppocr.so.5.0.0", "libppocr.so.5", "libppocr.so"):
    path = root / name
    if path.is_file():
        print(path)
        break
else:
    raise SystemExit("installed libppocr not found")
PY
)
export PPOCR_LIBRARY="$LIB"
export PPOCR_TEST_IMAGE="$ROOT/python/tests/data/hello.png"
export PPOCR_TEST_DET="$ROOT/ci-models/PP-OCRv6_det_tiny.onnx"
export PPOCR_TEST_REC="$ROOT/ci-models/PP-OCRv6_rec_tiny.onnx"
export PPOCR_TEST_DICT="$ROOT/ci-models/ppocrv6_tiny_dict.txt"

python - << 'PY'
import os
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Det.model_path": os.environ["PPOCR_TEST_DET"],
    "Rec.model_path": os.environ["PPOCR_TEST_REC"],
    "Rec.rec_keys_path": os.environ["PPOCR_TEST_DICT"],
    "Det.model_type": "tiny",
    "Rec.model_type": "tiny",
    "Global.use_cls": False,
})
result = engine(os.environ["PPOCR_TEST_IMAGE"])
print("hello", result.txts, result.scores)
assert result.txts == ("Hello RapidOCR 123",), result.txts
assert result.scores and float(result.scores[0]) > 0.9
print("hello_ocr_ok")
PY

rapidocr "$PPOCR_TEST_IMAGE" --model-type tiny \
  --det "$PPOCR_TEST_DET" --rec "$PPOCR_TEST_REC" --dict "$PPOCR_TEST_DICT" \
  | tee /tmp/rapidocr-cli.txt
grep -q "Hello RapidOCR 123" /tmp/rapidocr-cli.txt

python -m rapidocr "$PPOCR_TEST_IMAGE" --model-type tiny \
  --det "$PPOCR_TEST_DET" --rec "$PPOCR_TEST_REC" --dict "$PPOCR_TEST_DICT" \
  | tee /tmp/rapidocr-module.txt
grep -q "Hello RapidOCR 123" /tmp/rapidocr-module.txt

python -m pytest -q python/tests
echo "ci_wheel_check_ok"
