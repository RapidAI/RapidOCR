#!/usr/bin/env bash
# Metal checks on macOS: CPU vs Metal text/boxes, multithreaded determinism,
# zero Metal driver errors under Metal API validation, the CPU fallback when
# Metal is hidden, and the Python tests against this build's libppocr.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-"$ROOT/build-metal"}"
DET="${PPOCR_TEST_DET:-"$ROOT/ci-models/PP-OCRv6_det_tiny.onnx"}"
REC="${PPOCR_TEST_REC:-"$ROOT/ci-models/PP-OCRv6_rec_tiny.onnx"}"
DICT="${PPOCR_TEST_DICT:-"$ROOT/ci-models/ppocrv6_tiny_dict.txt"}"
HELLO_PPM="$BUILD/hello.ppm"

python3 - "$ROOT/python/tests/data/hello.png" "$HELLO_PPM" <<'PY'
import sys
from pathlib import Path
from PIL import Image
source, dest = sys.argv[1:]
image = Image.open(source).convert("RGB")
Path(dest).parent.mkdir(parents=True, exist_ok=True)
with open(dest, "wb") as handle:
    handle.write(f"P6\n{image.size[0]} {image.size[1]}\n255\n".encode())
    handle.write(image.tobytes())
PY

IMAGES=("$HELLO_PPM")
if [[ -n "${PPOCR_PAGE_PPM:-}" && -f "${PPOCR_PAGE_PPM}" ]]; then IMAGES+=("$PPOCR_PAGE_PPM"); fi

echo "+ Metal vs CPU (Metal API validation on)"
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert \
  "$BUILD/ppocr_metal_check" "$DET" "$REC" "$DICT" "${IMAGES[@]}" | tee "$BUILD/metal-check.log"

echo "+ Metal hidden: backend=metal must give CPU output"
PPOCR_DISABLE_METAL=1 "$BUILD/ppocr_metal_check" "$DET" "$REC" "$DICT" "$HELLO_PPM" | tee "$BUILD/metal-fallback.log"
grep -q "metal_available=0" "$BUILD/metal-fallback.log"

echo "+ ppocr_bench backend=metal"
PPOCR_BACKEND=metal "$BUILD/ppocr_bench" "$DET" "$REC" "$DICT" "$HELLO_PPM" 1 3

LIB="$BUILD/libppocr.dylib"
if [[ -f "$LIB" ]]; then
  export PPOCR_LIBRARY="$LIB"
  export PPOCR_TEST_DET="$DET" PPOCR_TEST_REC="$REC" PPOCR_TEST_DICT="$DICT"
  export PPOCR_TEST_IMAGE="$ROOT/python/tests/data/hello.png"
  export PYTHONPATH="$ROOT/python/src${PYTHONPATH:+:$PYTHONPATH}"
  python3 -m pytest -q "$ROOT/python/tests"
fi
echo "metal macos checks passed"
