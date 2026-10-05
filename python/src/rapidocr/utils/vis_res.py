"""``VisRes`` entry point. Drawing is Pillow RGB rather than OpenCV BGR."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class VisRes:
    def __init__(self, text_score: float = 0.5, font_path: Optional[str] = None, lang_type=None):
        self.text_score = text_score
        self.font_path = font_path
        self.lang_type = lang_type

    def __call__(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        txts: Optional[Sequence[str]] = None,
        scores: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        from PIL import Image, ImageDraw

        image = Image.fromarray(np.ascontiguousarray(img))
        if image.mode != "RGB":
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        labels = list(txts or ())
        for index, box in enumerate(boxes):
            polygon = [(float(point[0]), float(point[1])) for point in box]
            if len(polygon) >= 2:
                draw.line(polygon + [polygon[0]], fill=(0, 180, 0), width=2)
            if index < len(labels) and labels[index]:
                draw.text(polygon[0], str(labels[index]), fill=(200, 0, 0))
        return np.asarray(image)
