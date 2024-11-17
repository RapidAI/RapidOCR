import copy
import math
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np


def cal_rec_boxes(
    dt_boxes: Optional[List[np.ndarray]],
    crop_imgs: Optional[List[np.ndarray]],
    rec_res: Optional[List[Any]],
):
    res = []
    for i, (box, rec_res) in enumerate(zip(dt_boxes, rec_res)):
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

        rec_str, rec_conf, rec_word_info = rec_res[0], rec_res[1], rec_res[2]
        crop_img = crop_imgs[i]
        h, w = crop_img.shape[:2]
        crop_img_box = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        word_box_content_list, word_box_list = cal_ocr_word_box(
            rec_str, crop_img_box, rec_word_info
        )
        # fix word box overlap
        adjust_box_overlap(word_box_list)
        word_box_list = reverse_rotate_crop_image(
            copy.deepcopy(box), word_box_list, direction
        )

        res.append([rec_res[0], rec_res[1], word_box_content_list, word_box_list])
    return res


def adjust_box_overlap(word_box_list):
    # 调整bbox有重叠的地方
    for i in range(len(word_box_list) - 1):
        cur, nxt = word_box_list[i], word_box_list[i + 1]
        if cur[1][0] > nxt[0][0]:  # 有交集
            distance = abs(cur[1][0] - nxt[0][0])
            cur[1][0] -= distance / 2
            cur[2][0] -= distance / 2
            nxt[0][0] += distance / 2
            nxt[3][0] += distance / 2


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


def reverse_rotate_crop_image(bbox_points, word_points_list, direction="w"):
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
                new_point = s_rotate(
                    math.radians(-90), new_point[0], new_point[1], 0, 0
                )
                new_point[0] = new_point[0] + img_crop_width

            p = np.float32(new_point + [1])
            x, y, z = np.dot(IM, p)
            new_point = [x / z, y / z]

            new_point = [int(new_point[0] + left), int(new_point[1] + top)]
            new_word_points.append(new_point)
        new_word_points = order_points(new_word_points)
        new_word_points_list.append(new_word_points)
    return new_word_points_list


def cal_ocr_word_box(
    rec_str: str, box: np.ndarray, rec_word_info: List[Tuple[str, List[int]]]
):
    """Calculate the detection frame for each word based on the results of recognition and detection of ocr"""

    col_num, word_list, word_col_list, state_list = rec_word_info
    box = box.tolist()
    bbox_x_start = box[0][0]
    bbox_x_end = box[1][0]
    bbox_y_start = box[0][1]
    bbox_y_end = box[2][1]

    cell_width = (bbox_x_end - bbox_x_start) / col_num
    word_box_list = []
    word_box_content_list = []
    cn_width_list = []
    cn_col_list = []
    for word, word_col, state in zip(word_list, word_col_list, state_list):
        if state == "cn":
            if len(word_col) != 1:
                char_seq_length = (word_col[-1] - word_col[0] + 1) * cell_width
                char_width = char_seq_length / (len(word_col) - 1)
                cn_width_list.append(char_width)
            cn_col_list += word_col
            word_box_content_list += word
        else:
            cell_x_start = bbox_x_start + int(word_col[0] * cell_width)
            cell_x_end = bbox_x_start + int((word_col[-1] + 1) * cell_width)
            cell = [
                [cell_x_start, bbox_y_start],
                [cell_x_end, bbox_y_start],
                [cell_x_end, bbox_y_end],
                [cell_x_start, bbox_y_end],
            ]
            word_box_list.append(cell)
            word_box_content_list.append("".join(word))

    if len(cn_col_list) != 0:
        if len(cn_width_list) != 0:
            avg_char_width = np.mean(cn_width_list)
        else:
            avg_char_width = (bbox_x_end - bbox_x_start) / len(rec_str)
        for center_idx in cn_col_list:
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
            word_box_list.append(cell)
    sorted_word_box_list = sorted(word_box_list, key=lambda box: box[0][0])
    return word_box_content_list, sorted_word_box_list


def order_points(box):
    """矩形框顺序排列
    :param box: numpy.array, shape=(4, 2)
    :return:
    """
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

    return np.array([p1, p2, p3, p4]).reshape((-1, 2)).tolist()
