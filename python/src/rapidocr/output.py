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
        return _markdown(self.boxes, self.txts)

    def vis(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        if self.img is None or self.boxes is None:
            return None
        from .utils.vis_res import VisRes

        image = VisRes()(self.img, self.boxes, self.txts, self.scores)
        if save_path is not None:
            from PIL import Image

            Image.fromarray(image).save(save_path)
        return image


def _markdown(boxes: Optional[np.ndarray], txts: Optional[Tuple[str, ...]]) -> str:
    """Line grouping from ``rapidocr.utils.to_markdown`` on ``main``."""

    if boxes is None or txts is None:
        return "没有检测到任何文本。"
    items = []
    for box, text in zip(boxes, txts):
        array = np.asarray(box, dtype=np.float32)
        top = float(np.min(array[:, 1]))
        bottom = float(np.max(array[:, 1]))
        left = float(np.min(array[:, 0]))
        right = float(np.max(array[:, 0]))
        items.append(
            {
                "text": text,
                "props": {
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                    "height": bottom - top,
                    "width": right - left,
                    "center_y": top + (bottom - top) / 2,
                },
            }
        )
    if not items:
        return ""
    items.sort(key=lambda item: (item["props"]["center_y"], item["props"]["left"]))
    lines = []
    for item in items:
        matched = None
        for line in lines:
            if _same_line(item["props"], line["props"]):
                matched = line
                break
        if matched is None:
            lines.append({"items": [item], "props": dict(item["props"])})
            continue
        matched["items"].append(item)
        matched["props"] = _merge_props(matched["props"], item["props"])
    lines.sort(key=lambda line: (line["props"]["top"], line["props"]["left"]))
    output_lines = []
    previous = None
    for line in lines:
        line["items"].sort(key=lambda item: item["props"]["left"])
        parts = [line["items"][0]["text"]]
        prev_props = line["items"][0]["props"]
        for item in line["items"][1:]:
            gap = item["props"]["left"] - prev_props["right"]
            parts.append(_gap_text(gap, prev_props, item["props"]))
            parts.append(item["text"])
            prev_props = item["props"]
        if previous is not None:
            vertical = line["props"]["top"] - previous["bottom"]
            if vertical > max(previous["height"], line["props"]["height"]) * 0.8:
                output_lines.append("")
        output_lines.append("".join(parts))
        previous = line["props"]
    return "\n".join(output_lines)


def _same_line(current: dict, line: dict) -> bool:
    overlap = max(0.0, min(line["bottom"], current["bottom"]) - max(line["top"], current["top"]))
    min_height = max(1.0, min(current["height"], line["height"]))
    center_diff = abs(current["center_y"] - line["center_y"])
    return overlap / min_height > 0.5 or center_diff < min_height * 0.35


def _merge_props(left: dict, right: dict) -> dict:
    top = min(left["top"], right["top"])
    bottom = max(left["bottom"], right["bottom"])
    edge_left = min(left["left"], right["left"])
    edge_right = max(left["right"], right["right"])
    return {
        "top": top,
        "bottom": bottom,
        "left": edge_left,
        "right": edge_right,
        "height": bottom - top,
        "width": edge_right - edge_left,
        "center_y": top + (bottom - top) / 2,
    }


def _gap_text(gap: float, previous: dict, current: dict) -> str:
    if gap <= 1:
        return ""
    ref = max(1.0, min(previous["width"], current["width"]))
    spaces = max(1, int(round(gap / max(1.0, ref * 0.5))))
    return " " * min(spaces, 12)


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
