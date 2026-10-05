#!/usr/bin/env python3
"""Download the tiny PP-OCRv6 files used by the wheel test command.

The files land in ``ci-models/`` at the repository root. GitHub Actions
caches that directory. The script is stdlib-only so it can run before the
package is installed.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "ci-models"

# URLs and digests match python/src/rapidocr/default_models.yaml (3.9.2 catalog).
FILES = (
    (
        "PP-OCRv6_det_tiny.onnx",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_tiny.onnx",
        "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5",
    ),
    (
        "PP-OCRv6_rec_tiny.onnx",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx",
        "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b",
    ),
    (
        "ppocrv6_tiny_dict.txt",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/paddle/PP-OCRv6/rec/PP-OCRv6_rec_tiny/ppocrv6_tiny_dict.txt",
        "c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd",
    ),
    (
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url, expected in FILES:
        path = DEST / name
        if path.is_file() and _sha256(path) == expected:
            print(f"cached {path}")
            continue
        print(f"download {url}")
        tmp = path.with_suffix(path.suffix + ".partial")
        urllib.request.urlretrieve(url, tmp)
        actual = _sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"{name} sha256 {actual} != {expected}")
        tmp.replace(path)
        print(f"saved {path}")


if __name__ == "__main__":
    try:
        fetch()
    except Exception as exc:
        print(f"model fetch failed: {exc}", file=sys.stderr)
        raise
