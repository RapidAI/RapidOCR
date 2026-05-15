# -*- encoding: utf-8 -*-
import numpy as np


class ToMarkdown:
    @classmethod
    def to(cls, boxes, txts) -> str:
        """
        根据 OCR 结果的坐标信息，将文本还原为近似原始排版的 Markdown。

        Args:
            result (RapidOCROutput): RapidOCR 的输出结果对象。

        Returns:
            str: 模拟原始排版的 Markdown 字符串。
        """
        if boxes is None or txts is None:
            return "没有检测到任何文本。"

        items = [
            {"text": text, "props": cls.get_box_properties(box)}
            for box, text in zip(boxes, txts)
        ]
        if not items:
            return ""

        items.sort(key=lambda item: (item["props"]["center_y"], item["props"]["left"]))

        lines = []
        for item in items:
            matched_line = None
            for line in lines:
                if cls.is_same_line(item["props"], line["props"]):
                    matched_line = line
                    break

            if matched_line is None:
                lines.append({"items": [item], "props": dict(item["props"])})
                continue

            matched_line["items"].append(item)
            matched_line["props"] = cls.merge_props(matched_line["props"], item["props"])

        lines.sort(key=lambda line: (line["props"]["top"], line["props"]["left"]))

        output_lines = []
        prev_line_props = None
        for line in lines:
            line["items"].sort(key=lambda item: item["props"]["left"])

            current_line_parts = [line["items"][0]["text"]]
            prev_item_props = line["items"][0]["props"]

            for item in line["items"][1:]:
                current_props = item["props"]
                gap = current_props["left"] - prev_item_props["right"]
                current_line_parts.append(cls.get_gap_text(gap, prev_item_props, current_props))
                current_line_parts.append(item["text"])
                prev_item_props = current_props

            if prev_line_props is not None:
                vertical_gap = line["props"]["top"] - prev_line_props["bottom"]
                if vertical_gap > max(prev_line_props["height"], line["props"]["height"]) * 0.8:
                    output_lines.append("")

            output_lines.append("".join(current_line_parts))
            prev_line_props = line["props"]

        return "\n".join(output_lines)

    @staticmethod
    def is_same_line(current_props: dict, line_props: dict) -> bool:
        overlap_top = max(line_props["top"], current_props["top"])
        overlap_bottom = min(line_props["bottom"], current_props["bottom"])
        overlap = max(0.0, overlap_bottom - overlap_top)

        min_height = max(1.0, min(current_props["height"], line_props["height"]))
        overlap_ratio = overlap / min_height
        center_diff = abs(current_props["center_y"] - line_props["center_y"])

        return overlap_ratio > 0.5 or center_diff < min_height * 0.35

    @staticmethod
    def merge_props(a: dict, b: dict) -> dict:
        top = min(a["top"], b["top"])
        bottom = max(a["bottom"], b["bottom"])
        left = min(a["left"], b["left"])
        right = max(a["right"], b["right"])

        return {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "height": bottom - top,
            "width": right - left,
            "center_y": top + (bottom - top) / 2,
        }

    @staticmethod
    def get_gap_text(gap: float, prev_props: dict, current_props: dict) -> str:
        if gap <= 1:
            return ""

        ref_width = max(1.0, min(prev_props["width"], current_props["width"]))
        spaces = max(1, int(round(gap / max(1.0, ref_width * 0.5))))
        return " " * min(spaces, 12)

    @staticmethod
    def get_box_properties(box: np.ndarray) -> dict:
        """从坐标数组中计算框的几何属性"""
        # box shape is (4, 2) -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        ys = box[:, 1]
        xs = box[:, 0]

        top = np.min(ys)
        bottom = np.max(ys)
        left = np.min(xs)
        right = np.max(xs)

        return {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "height": bottom - top,
            "width": right - left,
            "center_y": top + (bottom - top) / 2,
        }
