# -*- encoding: utf-8 -*-
# @Author: SWHL / Joker1212
# @Contact: liekkaskono@163.com
import copy
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..ch_ppocr_rec.utils import TextRecOutput


class CalRecBoxes:
    """计算识别文字的汉字单字和英文单词的坐标框。
    代码借鉴自PaddlePaddle/PaddleOCR和fanqie03/char-detection"""

    def __init__(self):
        pass

    def __call__(
        self,
        imgs: Optional[List[np.ndarray]],
        dt_boxes: Optional[List[np.ndarray]],
        rec_res: TextRecOutput,
    ) -> TextRecOutput:
        word_results = []
        for idx, (img, box) in enumerate(zip(imgs, dt_boxes)):
            direction = self.get_box_direction(box)

            rec_txt = rec_res.txts[idx]
            rec_word_info = rec_res.word_results[idx]

            h, w = img.shape[:2]
            img_box = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            word_box_content_list, word_box_list, conf_list = self.cal_ocr_word_box(
                rec_txt, img_box, rec_word_info
            )
            word_box_list = self.adjust_box_overlap(copy.deepcopy(word_box_list))
            word_box_list = self.reverse_rotate_crop_image(
                copy.deepcopy(box), word_box_list, direction
            )
            word_results.extend(
                list(zip(word_box_content_list, conf_list, word_box_list))
            )

        rec_res.word_results = tuple(word_results)
        return rec_res

    @staticmethod
    def get_box_direction(box: np.ndarray) -> str:
        direction = "w"
        img_crop_width = int(
            max(
                np.linalg.norm(box[0] - box[1]),
                np.linalg.norm(box[2] - box[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(box[0] - box[3]),
                np.linalg.norm(box[1] - box[2]),
            )
        )
        if img_crop_height * 1.0 / img_crop_width >= 1.5:
            direction = "h"
        return direction

    @staticmethod
    def cal_ocr_word_box(
        rec_txt: str, box: np.ndarray, rec_word_info: List[Tuple[str, List[int]]]
    ) -> Tuple[List[str], List[List[int]], List[float]]:
        """Calculate the detection frame for each word based on the results of recognition and detection of ocr
        汉字坐标是单字的
        英语坐标是单词级别的
        """

        col_num, word_list, word_col_list, state_list, conf_list = rec_word_info
        box = box.tolist()
        bbox_x_start = box[0][0]
        bbox_x_end = box[1][0]
        bbox_y_start = box[0][1]
        bbox_y_end = box[2][1]

        cell_width = (bbox_x_end - bbox_x_start) / col_num
        word_box_list = []
        word_box_content_list = []
        cn_width_list = []
        en_width_list = []
        cn_col_list = []
        en_col_list = []

        def cal_char_width(width_list, word_col_):
            if len(word_col_) == 1:
                return
            char_total_length = (word_col_[-1] - word_col_[0]) * cell_width
            char_width = char_total_length / (len(word_col_) - 1)
            width_list.append(char_width)

        def cal_box(col_list, width_list, word_box_list_):
            if len(col_list) == 0:
                return
            if len(width_list) != 0:
                avg_char_width = np.mean(width_list)
            else:
                avg_char_width = (bbox_x_end - bbox_x_start) / len(rec_txt)

            for center_idx in col_list:
                center_x = (center_idx + 0.5) * cell_width
                cell_x_start = max(int(center_x - avg_char_width / 2), 0) + bbox_x_start
                cell_x_end = (
                    min(int(center_x + avg_char_width / 2), bbox_x_end - bbox_x_start)
                    + bbox_x_start
                )
                cell = [
                    [cell_x_start, bbox_y_start],
                    [cell_x_end, bbox_y_start],
                    [cell_x_end, bbox_y_end],
                    [cell_x_start, bbox_y_end],
                ]
                word_box_list_.append(cell)

        for word, word_col, state in zip(word_list, word_col_list, state_list):
            if state == "cn":
                cal_char_width(cn_width_list, word_col)
                cn_col_list += word_col
                word_box_content_list += word
            else:
                cal_char_width(en_width_list, word_col)
                en_col_list += word_col
                word_box_content_list += word

        cal_box(cn_col_list, cn_width_list, word_box_list)
        cal_box(en_col_list, en_width_list, word_box_list)
        sorted_word_box_list = sorted(word_box_list, key=lambda box: box[0][0])
        return word_box_content_list, sorted_word_box_list, conf_list

    @staticmethod
    def adjust_box_overlap(
        word_box_list: List[List[List[int]]],
    ) -> List[List[List[int]]]:
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
        word_points_list: List[List[List[int]]],
        direction: str = "w",
    ) -> List[List[List[int]]]:
        """
        get_rotate_crop_image的逆操作
        img为原图
        part_img为crop后的图
        bbox_points为part_img中对应在原图的bbox, 四个点，左上，右上，右下，左下
        part_points为在part_img中的点[(x, y), (x, y)]
        """
        bbox_points = np.float32(bbox_points)

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
                if direction == "h":
                    new_point = self.s_rotate(
                        math.radians(-90), new_point[0], new_point[1], 0, 0
                    )
                    new_point[0] = new_point[0] + img_crop_width

                p = np.float32(new_point + [1])
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
    def order_points(box: List[List[int]]) -> List[List[int]]:
        """矩形框顺序排列"""

        def convert_to_1x2(p):
            if p.shape == (2,):
                return p.reshape((1, 2))
            elif p.shape == (1, 2):
                return p
            else:
                return p[:1, :]

        box = np.array(box).reshape((-1, 2))
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
