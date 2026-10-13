"""Portable wheel check (used on Windows): ISA dispatch, OCR on the sample
images against the Linux CPU reference texts, CLI, GPU fallback, pytest."""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
DATA = ROOT / "python" / "tests" / "data"
MODELS = ROOT / "ci-models"

import rapidocr  # noqa: E402
from rapidocr import _native  # noqa: E402

info = rapidocr.cpu_info()
print(rapidocr.__version__, platform.platform(), info, flush=True)
isa = info["active_isa"]
if platform.machine().lower() in {"x86_64", "amd64"}:
    assert info["avx2_compiled"] and info["avx512_compiled"], info
    assert isa in {"avx2", "avx512"}, info
print("isa_dispatch_ok", isa)

if not (MODELS / "PP-OCRv6_det_tiny.onnx").is_file():
    subprocess.run([sys.executable, str(ROOT / "python/tests/fetch_ci_models.py")], check=True)

root = Path(rapidocr.__file__).resolve().parent
lib = next(p for p in (root / "ppocr.dll", root / "libppocr.so", root / "libppocr.dylib") if p.is_file())
os.environ.update(
    PPOCR_LIBRARY=str(lib),
    PPOCR_TEST_IMAGE=str(DATA / "hello.png"),
    PPOCR_TEST_DET=str(MODELS / "PP-OCRv6_det_tiny.onnx"),
    PPOCR_TEST_REC=str(MODELS / "PP-OCRv6_rec_tiny.onnx"),
    PPOCR_TEST_DICT=str(MODELS / "ppocrv6_tiny_dict.txt"),
)

try:
    print("backend_info", _native.backend_info())
except Exception as exc:  # GPU probing must never break the CPU path
    raise SystemExit(f"backend probe raised: {exc!r}")

from rapidocr import RapidOCR  # noqa: E402

engine = RapidOCR(params={
    "Det.model_path": os.environ["PPOCR_TEST_DET"],
    "Rec.model_path": os.environ["PPOCR_TEST_REC"],
    "Rec.rec_keys_path": os.environ["PPOCR_TEST_DICT"],
    "Det.model_type": "tiny",
    "Rec.model_type": "tiny",
    "Global.use_cls": False,
})
expected = json.loads((DATA / "expected_cpu_texts.json").read_text(encoding="utf-8"))
for name, texts in expected.items():
    result = engine(str(DATA / name))
    got = list(result.txts or ())
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        engine(str(DATA / name))
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"timing {name} isa={isa} median_ms={times[1]:.1f} lines={len(got)}", flush=True)
    if got != texts:
        for i, (a, b) in enumerate(zip(got, texts)):
            if a != b:
                print("diff", i, repr(a), "!=", repr(b))
        raise SystemExit(f"{name}: {len(got)} lines, expected {len(texts)}; text differs from the Linux CPU reference")
    print("match", name, len(got))
print("reference_texts_ok")

env = dict(os.environ, PYTHONIOENCODING="utf-8")
args = [str(DATA / "hello.png"), "--model-type", "tiny", "--det", os.environ["PPOCR_TEST_DET"],
        "--rec", os.environ["PPOCR_TEST_REC"], "--dict", os.environ["PPOCR_TEST_DICT"]]
out = subprocess.run([sys.executable, "-m", "rapidocr", *args], check=True, capture_output=True,
                     text=True, encoding="utf-8", env=env).stdout
print(out)
assert "Hello RapidOCR 123" in out
print("cli_ok")

sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", str(ROOT / "python/tests")]))
