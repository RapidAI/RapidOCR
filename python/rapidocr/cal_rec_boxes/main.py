# -*- encoding: utf-8 -*-
# @Author: SWHL / Joker1212
# @Contact: liekkaskono@163.com
import copy
import math
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np

from ..ch_ppocr_rec.typings import TextRecOutput, WordInfo, WordType
from ..utils.utils import quads_to_rect_bbox


class Direction(Enum):
    HORIZONTAL = "horizontal_direct"  # 水平
    VERTICAL = "vertical_direct"  # 垂直


class CalRecBoxes:
    """计算识别文字的汉字单字和英文单词的坐标框。
    代码借鉴自PaddlePaddle/PaddleOCR和fanqie03/char-detection"""

    def __call__(
        self,
        imgs: List[np.ndarray],
        dt_boxes: List[np.ndarray],
        rec_res: TextRecOutput,
        return_single_char_box: bool = False,
    ) -> TextRecOutput:
        word_results = []
        for idx, (img, box) in enumerate(zip(imgs, dt_boxes)):
            if rec_res.txts is None:
                continue

            h, w = img.shape[:2]
            img_box = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            word_box_content_list, word_box_list, conf_list = self.cal_ocr_word_box(
                rec_res.txts[idx],
                img_box,
                rec_res.word_results[idx],
                return_single_char_box,
            )
            word_box_list = self.adjust_box_overlap(copy.deepcopy(word_box_list))
            direction = self.get_box_direction(box)
            word_box_list = self.reverse_rotate_crop_image(
                copy.deepcopy(box), word_box_list, direction
            )
            word_results.append(
                list(zip(word_box_content_list, conf_list, word_box_list))
            )

        rec_res.word_results = tuple(word_results)
        return rec_res

    @staticmethod
    def get_box_direction(box: np.ndarray) -> Direction:
        edge_lengths = [
            float(np.linalg.norm(box[0] - box[1])),  # 上边
            float(np.linalg.norm(box[1] - box[2])),  # 右边
            float(np.linalg.norm(box[2] - box[3])),  # 下边
            float(np.linalg.norm(box[3] - box[0])),  # 左边
        ]

        # 宽和高取对边的最大距离
        width = max(edge_lengths[0], edge_lengths[2])
        height = max(edge_lengths[1], edge_lengths[3])

        if width < 1e-6:
            return Direction.VERTICAL

        aspect_ratio = round(height / width, 2)
        return Direction.VERTICAL if aspect_ratio >= 1.5 else Direction.HORIZONTAL

    def cal_ocr_word_box(
        self,
        rec_txt: str,
        bbox: np.ndarray,
        word_info: WordInfo,
        return_single_char_box: bool = False,
    ) -> Tuple[List[str], List[List[List[float]]], List[float]]:
        """Calculate the detection frame for each word based on the results of recognition and detection of ocr
        汉字坐标是单字的
        英语坐标是单词级别的
        三种情况：
        1. 全是汉字
        2. 全是英文
        3. 中英混合
        """
        if not rec_txt or word_info.line_txt_len == 0:
            return [], [], []

        bbox_points = quads_to_rect_bbox(bbox[None, ...])
        avg_col_width = (bbox_points[2] - bbox_points[0]) / word_info.line_txt_len

        is_all_en_num = all(v is WordType.EN_NUM for v in word_info.word_types)

        line_cols, char_widths, word_contents = [], [], []
        for word, word_col in zip(word_info.words, word_info.word_cols):
            if is_all_en_num and not return_single_char_box:
                line_cols.append(word_col)
                word_contents.append("".join(word))
            else:
                line_cols.extend(word_col)
                word_contents.extend(word)

            if len(word_col) == 1:
                continue

            avg_width = self.calc_avg_char_width(word_col, avg_col_width)
            char_widths.append(avg_width)

        avg_char_width = self.calc_all_char_avg_width(
            char_widths, bbox_points[0], bbox_points[2], len(rec_txt)
        )

        if is_all_en_num and not return_single_char_box:
            word_boxes = self.calc_en_num_box(
                line_cols, avg_char_width, avg_col_width, bbox_points
            )
        else:
            word_boxes = self.calc_box(
                line_cols, avg_char_width, avg_col_width, bbox_points
            )
        return word_contents, word_boxes, word_info.confs

    def calc_en_num_box(
        self,
        line_cols: List[List[int]],
        avg_char_width: float,
        avg_col_width: float,
        bbox_points: Tuple[float, float, float, float],
    ) -> List[List[List[float]]]:
        results = []
        for one_col in line_cols:
            cur_word_cell = self.calc_box(
                one_col, avg_char_width, avg_col_width, bbox_points
            )
            x0, y0, x1, y1 = quads_to_rect_bbox(np.array(cur_word_cell))
            results.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return results

    @staticmethod
    def calc_box(
        line_cols: List[int],
        avg_char_width: float,
        avg_col_width: float,
        bbox_points: Tuple[float, float, float, float],
    ) -> List[List[List[float]]]:
        x0, y0, x1, y1 = bbox_points

        results = []
        for col_idx in line_cols:
            # 将中心点定位在列的中间位置
            center_x = (col_idx + 0.5) * avg_col_width

            # 计算字符单元格的左右边界
            char_x0 = max(int(center_x - avg_char_width / 2), 0) + x0
            char_x1 = min(int(center_x + avg_char_width / 2), x1 - x0) + x0
            cell = [
                [char_x0, y0],
                [char_x1, y0],
                [char_x1, y1],
                [char_x0, y1],
            ]
            results.append(cell)
        return sorted(results, key=lambda x: x[0][0])

    @staticmethod
    def calc_avg_char_width(word_col: List[int], each_col_width: float) -> float:
        char_total_length = (word_col[-1] - word_col[0]) * each_col_width
        return char_total_length / (len(word_col) - 1)

    @staticmethod
    def calc_all_char_avg_width(
        width_list: List[float], bbox_x0: float, bbox_x1: float, txt_len: int
    ) -> float:
        if txt_len == 0:
            return 0.0

        if len(width_list) > 0:
            return sum(width_list) / len(width_list)

        return (bbox_x1 - bbox_x0) / txt_len

    @staticmethod
    def adjust_box_overlap(
        word_box_list: List[List[List[float]]],
    ) -> List[List[List[float]]]:
        # 调整bbox有重叠的地方
        for i in range(len(word_box_list) - 1):
            cur, nxt = word_box_list[i], word_box_list[i + 1]
            if cur[1][0] > nxt[0][0]:  # 有交集
                distance = abs(cur[1][0] - nxt[0][0])
                cur[1][0] -= distance / 2
                cur[2][0] -= distance / 2
                nxt[0][0] += distance - distance / 2
                nxt[3][0] += distance - distance / 2
        return word_box_list

    def reverse_rotate_crop_image(
        self,
        bbox_points: np.ndarray,
        word_points_list: List[List[List[float]]],
        direction: Direction,
    ) -> List[List[List[int]]]:
        """
        get_rotate_crop_image的逆操作
        img为原图
        part_img为crop后的图
        bbox_points为part_img中对应在原图的bbox, 四个点，左上，右上，右下，左下
        part_points为在part_img中的点[(x, y), (x, y)]
        """
        left = int(np.min(bbox_points[:, 0]))
        top = int(np.min(bbox_points[:, 1]))
        bbox_points[:, 0] = bbox_points[:, 0] - left
        bbox_points[:, 1] = bbox_points[:, 1] - top

        img_crop_width = int(np.linalg.norm(bbox_points[0] - bbox_points[1]))
        img_crop_height = int(np.linalg.norm(bbox_points[0] - bbox_points[3]))

        pts_std = np.array(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        ).astype(np.float32)
        M = cv2.getPerspectiveTransform(bbox_points, pts_std)
        _, IM = cv2.invert(M)

        new_word_points_list = []
        for word_points in word_points_list:
            new_word_points = []
            for point in word_points:
                new_point = point
                if direction == Direction.VERTICAL:
                    new_point = self.s_rotate(
                        math.radians(-90), new_point[0], new_point[1], 0, 0
                    )
                    new_point[0] = new_point[0] + img_crop_width

                p = np.array(new_point + [1])
                x, y, z = np.dot(IM, p)
                new_point = [x / z, y / z]

                new_point = [int(new_point[0] + left), int(new_point[1] + top)]
                new_word_points.append(new_point)
            new_word_points = self.order_points(new_word_points)
            new_word_points_list.append(new_word_points)
        return new_word_points_list

    @staticmethod
    def s_rotate(angle, valuex, valuey, pointx, pointy):
        """绕pointx,pointy顺时针旋转
        https://blog.csdn.net/qq_38826019/article/details/84233397
        """
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        sRotatex = (
            (valuex - pointx) * math.cos(angle)
            + (valuey - pointy) * math.sin(angle)
            + pointx
        )
        sRotatey = (
            (valuey - pointy) * math.cos(angle)
            - (valuex - pointx) * math.sin(angle)
            + pointy
        )
        return [sRotatex, sRotatey]

    @staticmethod
    def order_points(ori_box: List[List[int]]) -> List[List[int]]:
        """矩形框顺序排列"""

        def convert_to_1x2(p):
            if p.shape == (2,):
                return p.reshape((1, 2))

            if p.shape == (1, 2):
                return p
            return p[:1, :]

        box = np.array(ori_box).reshape((-1, 2))
        center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
        if np.any(box[:, 0] == center_x) and np.any(
            box[:, 1] == center_y
        ):  # 有两点横坐标相等，有两点纵坐标相等，菱形
            p1 = box[np.where(box[:, 0] == np.min(box[:, 0]))]
            p2 = box[np.where(box[:, 1] == np.min(box[:, 1]))]
            p3 = box[np.where(box[:, 0] == np.max(box[:, 0]))]
            p4 = box[np.where(box[:, 1] == np.max(box[:, 1]))]
        elif np.all(box[:, 0] == center_x):  # 四个点的横坐标都相同
            y_sort = np.argsort(box[:, 1])
            p1 = box[y_sort[0]]
            p2 = box[y_sort[1]]
            p3 = box[y_sort[2]]
            p4 = box[y_sort[3]]
        elif np.any(box[:, 0] == center_x) and np.all(
            box[:, 1] != center_y
        ):  # 只有两点横坐标相等，先上下再左右
            p12, p34 = (
                box[np.where(box[:, 1] < center_y)],
                box[np.where(box[:, 1] > center_y)],
            )
            p1, p2 = (
                p12[np.where(p12[:, 0] == np.min(p12[:, 0]))],
                p12[np.where(p12[:, 0] == np.max(p12[:, 0]))],
            )
            p3, p4 = (
                p34[np.where(p34[:, 0] == np.max(p34[:, 0]))],
                p34[np.where(p34[:, 0] == np.min(p34[:, 0]))],
            )
        else:  # 只有两点纵坐标相等，或者是没有相等的，先左右再上下
            p14, p23 = (
                box[np.where(box[:, 0] < center_x)],
                box[np.where(box[:, 0] > center_x)],
            )
            p1, p4 = (
                p14[np.where(p14[:, 1] == np.min(p14[:, 1]))],
                p14[np.where(p14[:, 1] == np.max(p14[:, 1]))],
            )
            p2, p3 = (
                p23[np.where(p23[:, 1] == np.min(p23[:, 1]))],
                p23[np.where(p23[:, 1] == np.max(p23[:, 1]))],
            )

        # 解决单字切割后横坐标完全相同的shape错误
        p1 = convert_to_1x2(p1)
        p2 = convert_to_1x2(p2)
        p3 = convert_to_1x2(p3)
        p4 = convert_to_1x2(p4)
        return np.array([p1, p2, p3, p4]).reshape((-1, 2)).tolist()
