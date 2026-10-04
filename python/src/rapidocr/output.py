from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RapidOCROutput:
    """Same fields as ``rapidocr.utils.output.RapidOCROutput`` on ``main``.

    ``word_results`` stays empty: the native engine does not return per-word
    boxes. ``elapse`` is the summed runtime in seconds, matching ``main``.
    """

    img: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    txts: Optional[Tuple[str, ...]] = None
    scores: Optional[Tuple[float, ...]] = None
    word_results: Tuple[Tuple[Any, ...], ...] = ()
    elapse_list: List[Optional[float]] = field(default_factory=list)
    elapse: float = field(init=False)

    def __post_init__(self) -> None:
        self.elapse = sum(v for v in self.elapse_list if isinstance(v, float))

    def __len__(self) -> int:
        if self.txts is None:
            return 0
        return len(self.txts)

    def to_json(self) -> Optional[List[Dict[str, Any]]]:
        if self.boxes is None or self.txts is None or self.scores is None:
            return None
        return [
            {"box": box.tolist(), "txt": txt, "score": float(score)}
            for box, txt, score in zip(self.boxes, self.txts, self.scores)
        ]

    def to_markdown(self) -> str:
        if not self.txts:
            return ""
        return "\n".join(self.txts)

    def vis(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        if self.img is None or self.boxes is None:
            return None
        from PIL import Image, ImageDraw

        image = Image.fromarray(self.img.copy())
        draw = ImageDraw.Draw(image)
        for box, text in zip(self.boxes, self.txts or ()):
            polygon = [(float(p[0]), float(p[1])) for p in box]
            draw.line(polygon + [polygon[0]], fill=(0, 180, 0), width=2)
            if text:
                draw.text(polygon[0], text, fill=(200, 0, 0))
        out = np.asarray(image)
        if save_path is not None:
            image.save(save_path)
        return out


def filter_by_score(
    boxes: Sequence[Sequence[Sequence[float]]],
    txts: Sequence[str],
    scores: Sequence[float],
    text_score: float,
) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[float, ...]]:
    kept_boxes = []
    kept_txts = []
    kept_scores = []
    for box, txt, score in zip(boxes, txts, scores):
        if not str(txt).strip() or float(score) < text_score:
            continue
        kept_boxes.append(box)
        kept_txts.append(txt)
        kept_scores.append(float(score))
    if not kept_boxes:
        return np.zeros((0, 4, 2), dtype=np.float32), tuple(), tuple()
    return (
        np.asarray(kept_boxes, dtype=np.float32),
        tuple(kept_txts),
        tuple(kept_scores),
    )
