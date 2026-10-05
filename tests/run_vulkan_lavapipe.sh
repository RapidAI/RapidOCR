#!/usr/bin/env bash
# Run the Vulkan checks against lavapipe (or another device already selected
# by PPOCR_VULKAN_DEVICE_INDEX). Exit 0 only when text matches the CPU path
# and the validation layer reports no errors. Lavapipe timings are software
# rendering, not GPU performance.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-"$ROOT/build-vk"}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg}"
mkdir -p "$XDG_RUNTIME_DIR"
unset DISPLAY || true

if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  for icd in /usr/share/vulkan/icd.d/lvp_icd.json /usr/share/vulkan/icd.d/lvp_icd.*.json; do
    if [[ -f "$icd" ]]; then
      export VK_ICD_FILENAMES="$icd"
      break
    fi
  done
fi
export PPOCR_VULKAN_DEVICE_INDEX="${PPOCR_VULKAN_DEVICE_INDEX:-0}"
export PPOCR_VULKAN_VALIDATION="${PPOCR_VULKAN_VALIDATION:-1}"

DET="${PPOCR_TEST_DET:-"$ROOT/ci-models/PP-OCRv6_det_tiny.onnx"}"
REC="${PPOCR_TEST_REC:-"$ROOT/ci-models/PP-OCRv6_rec_tiny.onnx"}"
DICT="${PPOCR_TEST_DICT:-"$ROOT/ci-models/ppocrv6_tiny_dict.txt"}"
HELLO_PPM="${PPOCR_TEST_PPM:-"$BUILD/hello.ppm"}"

python3 - "$ROOT/python/tests/data/hello.png" "$HELLO_PPM" <<'PY'
import sys
from pathlib import Path
from PIL import Image
source, dest = sys.argv[1:]
image = Image.open(source).convert("RGB")
width, height = image.size
payload = image.tobytes()
Path(dest).parent.mkdir(parents=True, exist_ok=True)
with open(dest, "wb") as handle:
    handle.write(f"P6\n{width} {height}\n255\n".encode())
    handle.write(payload)
print(dest)
PY

LOG="$BUILD/vulkan-lavapipe.log"
mkdir -p "$BUILD"
run() {
  echo "+ $*"
  "$@"
}

{
  run "$BUILD/ppocr_determinism" pool 50
  run "$BUILD/ppocr_determinism" ocr "$DET" "$REC" "$DICT" "$HELLO_PPM" 20
  serial_hello="$(PPOCR_BENCH_THREADS=1 "$BUILD/ppocr_determinism" fingerprint "$DET" "$REC" "$DICT" "$HELLO_PPM" 1)"
  default_hello="$("$BUILD/ppocr_determinism" fingerprint "$DET" "$REC" "$DICT" "$HELLO_PPM" 5)"
  if [[ "$serial_hello" != "$default_hello" ]]; then
    echo "hello fingerprint differs between 1 thread and the default pool" >&2
    exit 1
  fi
  if [[ -n "${PPOCR_PAGE_PPM:-}" && -f "${PPOCR_PAGE_PPM}" ]]; then
    run "$BUILD/ppocr_determinism" ocr "$DET" "$REC" "$DICT" "$PPOCR_PAGE_PPM" 20
    serial_page="$(PPOCR_BENCH_THREADS=1 "$BUILD/ppocr_determinism" fingerprint "$DET" "$REC" "$DICT" "$PPOCR_PAGE_PPM" 1)"
    default_page="$("$BUILD/ppocr_determinism" fingerprint "$DET" "$REC" "$DICT" "$PPOCR_PAGE_PPM" 5)"
    if [[ "$serial_page" != "$default_page" ]]; then
      echo "page fingerprint differs between 1 thread and the default pool" >&2
      exit 1
    fi
  fi
  run "$BUILD/ppocr_vulkan_smoke"
  run "$BUILD/ppocr_gpu_ocr_smoke" "$DET" "$REC" "$DICT" "$HELLO_PPM"
  if [[ -n "${PPOCR_PAGE_PPM:-}" && -f "${PPOCR_PAGE_PPM}" ]]; then
    run "$BUILD/ppocr_gpu_ocr_smoke" "$DET" "$REC" "$DICT" "$HELLO_PPM" "$PPOCR_PAGE_PPM"
  fi
  run "$BUILD/ppocr_vulkan_robust" lifecycle "$DET" "$REC" "$DICT" "$HELLO_PPM"
  run "$BUILD/ppocr_vulkan_robust" threads "$DET" "$REC" "$DICT" "$HELLO_PPM"
} 2>&1 | tee "$LOG"

if grep -q "Validation Error:" "$LOG"; then
  echo "Vulkan validation reported errors" >&2
  exit 1
fi

# A process with no ICD, and a nonsense device index, must stay on CPU.
env -u PPOCR_VULKAN_DEVICE_INDEX -u PPOCR_VULKAN_DEVICE_NAME \
  VK_ICD_FILENAMES=/dev/null VK_DRIVER_FILES=/dev/null \
  "$BUILD/ppocr_vulkan_robust" missing "$DET" "$REC" "$DICT"
"$BUILD/ppocr_vulkan_robust" bad-index "$DET" "$REC" "$DICT"

if [[ -n "${PPOCR_LIBRARY:-}" || -f "$BUILD/libppocr.so" ]]; then
  export PPOCR_LIBRARY="${PPOCR_LIBRARY:-$BUILD/libppocr.so}"
  export PPOCR_TEST_DET="$DET" PPOCR_TEST_REC="$REC" PPOCR_TEST_DICT="$DICT"
  export PPOCR_TEST_IMAGE="$ROOT/python/tests/data/hello.png"
  export PYTHONPATH="$ROOT/python/src${PYTHONPATH:+:$PYTHONPATH}"
  python3 -m pytest -q "$ROOT/python/tests/test_vulkan.py" "$ROOT/python/tests/test_main_compat.py"
fi

echo "vulkan lavapipe checks passed"
