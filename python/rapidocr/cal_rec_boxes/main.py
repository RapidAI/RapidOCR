# -*- encoding: utf-8 -*-
# @Author: SWHL / Joker1212
# @Contact: liekkaskono@163.com
import copy
import math
from typing import List

import cv2
import numpy as np

from ..ch_ppocr_rec.typings import TextRecOutput, WordInfo, WordType
from ..utils.utils import quads_to_rect_bbox


class CalRecBoxes:
    """计算识别文字的汉字单字和英文单词的坐标框。
    代码借鉴自PaddlePaddle/PaddleOCR和fanqie03/char-detection"""

    def __init__(self):
        pass

    def __call__(
        self, imgs: List[np.ndarray], dt_boxes: List[np.ndarray], rec_res: TextRecOutput
    ) -> TextRecOutput:
        word_results = []
        for idx, (img, box) in enumerate(zip(imgs, dt_boxes)):
            direction = self.get_box_direction(box)

            if rec_res.txts is None:
                continue

            h, w = img.shape[:2]
            img_box = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            word_box_content_list, word_box_list, conf_list = self.cal_ocr_word_box(
                rec_res.txts[idx], img_box, rec_res.word_results[idx]
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
            max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))
        )
        img_crop_height = int(
            max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))
        )
        if img_crop_height * 1.0 / img_crop_width >= 1.5:
            direction = "h"
        return direction

    def cal_ocr_word_box(
        self,
        rec_txt: str,
        bbox: np.ndarray,
        word_info: List[WordInfo],
    ):
        """Calculate the detection frame for each word based on the results of recognition and detection of ocr
        汉字坐标是单字的
        英语坐标是单词级别的
        三种情况：
        1. 全是汉字
        2. 全是英文
        3. 中英混合
        """
        word_len = word_info.word_len
        words = word_info.words
        word_cols = word_info.word_cols
        word_types = word_info.word_types
        confs = word_info.confs

        bbox_x_start, bbox_y_start, bbox_x_end, bbox_y_end = quads_to_rect_bbox(
            bbox[None, ...]
        )
        each_col_width = (bbox_x_end - bbox_x_start) / word_len

        if all(v is WordType.EN_NUM for v in word_types):
            return self.all_en_num_process(
                words,
                word_cols,
                each_col_width,
                bbox_x_start,
                bbox_y_start,
                bbox_x_end,
                bbox_y_end,
                rec_txt,
                confs,
            )

        if all(v is WordType.CN for v in word_types):
            return self.all_cn_process(
                words,
                word_cols,
                each_col_width,
                bbox_x_start,
                bbox_y_start,
                bbox_x_end,
                bbox_y_end,
                rec_txt,
                confs,
            )

        if all(v in [WordType.CN, WordType.EN_NUM] for v in word_types):
            return self.all_cn_process(
                words,
                word_cols,
                each_col_width,
                bbox_x_start,
                bbox_y_start,
                bbox_x_end,
                bbox_y_end,
                rec_txt,
                confs,
            )

    def all_cn_process(
        self,
        word_list,
        word_col_list,
        each_col_width,
        bbox_x_start,
        bbox_y_start,
        bbox_x_end,
        bbox_y_end,
        rec_txt,
        conf_list,
    ):
        cn_width_list, cn_col_list, cn_word_box_content_list = [], [], []
        for word, word_col in zip(word_list, word_col_list):
            cn_col_list.extend(word_col)
            cn_word_box_content_list.extend(word)

            if len(word_col) == 1:
                continue

            char_avg_width = self.calc_char_avg_width(word_col, each_col_width)
            cn_width_list.append(char_avg_width)

        cn_cell_avg_width = self.calc_cell_avg_width(
            cn_width_list, bbox_x_start, bbox_x_end, len(rec_txt)
        )
        cn_word_box_list = self.calc_box(
            cn_col_list,
            cn_cell_avg_width,
            each_col_width,
            bbox_x_start,
            bbox_y_start,
            bbox_x_end,
            bbox_y_end,
        )
        sorted_word_box_list = sorted(cn_word_box_list, key=lambda box: box[0][0])
        return cn_word_box_content_list, sorted_word_box_list, conf_list

    def all_en_num_process(
        self,
        word_list,
        word_col_list,
        each_col_width,
        bbox_x_start,
        bbox_y_start,
        bbox_x_end,
        bbox_y_end,
        rec_txt,
        conf_list,
    ):
        en_width_list, en_col_list, en_word_box_content_list = [], [], []
        for word, word_col in zip(word_list, word_col_list):
            en_col_list.append(word_col)
            en_word_box_content_list.append("".join(word))

            if len(word_col) == 1:
                continue

            char_avg_width = self.calc_char_avg_width(word_col, each_col_width)
            en_width_list.append(char_avg_width)

        en_cell_avg_width = self.calc_cell_avg_width(
            en_width_list, bbox_x_start, bbox_x_end, len(rec_txt)
        )
        en_word_box_list = self.calc_en_box(
            en_col_list,
            en_cell_avg_width,
            each_col_width,
            bbox_x_start,
            bbox_y_start,
            bbox_x_end,
            bbox_y_end,
        )
        sorted_word_box_list = sorted(en_word_box_list, key=lambda box: box[0][0])
        return en_word_box_content_list, sorted_word_box_list, conf_list

    def calc_en_box(
        self,
        col_list,
        cell_avg_width: float,
        each_col_width: float,
        bbox_x0: float,
        bbox_y0: float,
        bbox_x1: float,
        bbox_y1: float,
    ):
        results = []
        for one_col in col_list:
            cur_word_cell = self.calc_box(
                one_col,
                cell_avg_width,
                each_col_width,
                bbox_x0,
                bbox_y0,
                bbox_x1,
                bbox_y1,
            )
            x0, y0, x1, y1 = quads_to_rect_bbox(np.array(cur_word_cell))
            results.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return results

    @staticmethod
    def calc_box(
        col_list,
        cell_avg_width: float,
        each_col_width: float,
        bbox_x0: float,
        bbox_y0: float,
        bbox_x1: float,
        bbox_y1: float,
    ) -> List[List[float]]:
        results = []
        for col_idx in col_list:
            center_x = (col_idx + 0.5) * each_col_width
            cell_x_start = max(int(center_x - cell_avg_width / 2), 0) + bbox_x0
            cell_x_end = (
                min(int(center_x + cell_avg_width / 2), bbox_x1 - bbox_x0) + bbox_x0
            )
            cell = [
                [cell_x_start, bbox_y0],
                [cell_x_end, bbox_y0],
                [cell_x_end, bbox_y1],
                [cell_x_start, bbox_y1],
            ]
            results.append(cell)
        return results

    @staticmethod
    def calc_char_avg_width(word_col: List[int], each_col_width: float) -> float:
        char_total_length = (word_col[-1] - word_col[0]) * each_col_width
        return char_total_length / (len(word_col) - 1)

    @staticmethod
    def calc_cell_avg_width(
        width_list: List[float],
        bbox_x0: float,
        bbox_x1: float,
        txt_len: int,
    ) -> float:
        if len(width_list) > 0:
            return sum(width_list) / len(width_list)
        return (bbox_x1 - bbox_x0) / txt_len

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
