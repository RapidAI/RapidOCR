"""Download the official PP-OCRv6 ONNX bundles used by RapidOCR."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

# URLs and ONNX hashes match python/rapidocr/default_models.yaml on main
# (RapidOCR v3.9.2). Dictionaries are the ones published next to the MNN
# PP-OCRv6 recognizers.
_MODELS = {
    "tiny": {
        "det": (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_tiny.onnx",
            "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5",
        ),
        "rec": (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx",
            "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b",
        ),
        "dict": (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/paddle/PP-OCRv6/rec/PP-OCRv6_rec_tiny/ppocrv6_tiny_dict.txt",
            None,
        ),
    },
    "small": {
        "det": (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
            "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
        ),
        "rec": (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
            "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
        ),
        "dict": (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt",
            None,
        ),
    },
    "medium": {
        "det": (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_medium.onnx",
            "92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2",
        ),
        "rec": (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx",
            "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b",
        ),
        "dict": (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/paddle/PP-OCRv6/rec/PP-OCRv6_rec_medium/ppocrv6_dict.txt",
            None,
        ),
    },
}


def download_models(model_type: str = "small", dest: Optional[str] = None) -> dict:
    """Download detector, recognizer, and dictionary. Returns their paths."""

    key = str(model_type).lower()
    if key not in _MODELS:
        raise ValueError(f"unsupported PP-OCRv6 model_type {model_type!r}")
    root = Path(dest) if dest else Path.home() / ".cache" / "rapidocr" / "ppocrv6" / key
    root.mkdir(parents=True, exist_ok=True)
    names = {"det": "det.onnx", "rec": "rec.onnx", "dict": "dict.txt"}
    out = {}
    for role, filename in names.items():
        url, digest = _MODELS[key][role]
        path = root / filename
        if not _matches(path, digest):
            urllib.request.urlretrieve(url, path)
            if digest and not _matches(path, digest):
                raise RuntimeError(f"checksum mismatch for {path}")
        out[role] = str(path)
    return out


def _matches(path: Path, digest: Optional[str]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if not digest:
        return True
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == digest
