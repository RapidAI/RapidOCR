"""Document-family detection for Spanish OCR model routing.

The fine-tuned checkpoints specialize: the CAD-tuned model wins on technical
drawings and tables, while the baseline is better on clean CV/CUL pages. This
module decides the family from cheap image statistics (no second model):

- CAD drawings have many long, straight edges (frame, dimensions, grid lines);
- clean administrative pages are mostly text over white with few long lines.

The decision is intentionally conservative: anything that looks like a drawing
goes to the CAD model; everything else stays on the clean model.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

CAD = "cad"
CLEAN = "clean"

# A dense technical drawing produces dozens of long segments after Hough;
# a scanned form or a CV page produces a handful at most.
DEFAULT_MIN_LINES = 100
DEFAULT_MIN_LINE_LEN = 250


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def count_long_lines(
    img: np.ndarray,
    min_line_len: int = DEFAULT_MIN_LINE_LEN,
) -> int:
    """Count long straight segments via Canny + probabilistic Hough."""
    gray = _to_gray(img)
    # Downscale very large sheets: the heuristic is about structure, not pixels.
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > 2000:
        scale = 2000.0 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(20, int(min_line_len * scale)),
        maxLineGap=4,
    )
    return 0 if lines is None else len(lines)


def detect_doc_family(
    img: np.ndarray,
    min_lines: int = DEFAULT_MIN_LINES,
    min_line_len: int = DEFAULT_MIN_LINE_LEN,
) -> Tuple[str, int]:
    """Return (family, long_line_count)."""
    n = count_long_lines(img, min_line_len=min_line_len)
    return (CAD if n >= min_lines else CLEAN), n


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img