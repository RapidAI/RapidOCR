# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: db_process.py
# @Time: 2021/01/16 12:00:18
# @Author: Max
import sys
from copy import deepcopy

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


class DBPreProcess(object):
    def __init__(self):
        super(DBPreProcess, self).__init__()
        self.max_side_len = 2400

    def resize_image(self, im):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        max_side_len = self.max_side_len
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            if resize_h > resize_w:
                ratio = float(max_side_len) / resize_h
            else:
                ratio = float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        if resize_h % 32 == 0:
            resize_h = resize_h
        elif resize_h // 32 <= 1:
            resize_h = 32
        else:
            resize_h = (resize_h // 32) * 32
        if resize_w % 32 == 0:
            resize_w = resize_w
        elif resize_w // 32 <= 1:
            resize_w = 32
        else:
            resize_w = (resize_w // 32) * 32
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            im = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(im.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def normalize(self, im):
        """
        Normalize image
        :param im: input image
        :return: Normalized image
        """
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im[:, :, 0] -= img_mean[0]
        im[:, :, 1] -= img_mean[1]
        im[:, :, 2] -= img_mean[2]
        im[:, :, 0] /= img_std[0]
        im[:, :, 1] /= img_std[1]
        im[:, :, 2] /= img_std[2]
        channel_swap = (2, 0, 1)
        im = im.transpose(channel_swap)
        return im

    def __call__(self, im):
        im, (ratio_h, ratio_w) = self.resize_image(im)
        im = self.normalize(im)
        im = im[np.newaxis, :]
        return [im, (ratio_h, ratio_w)]


class DBPostProcess(object):
    def __init__(self):
        self.thresh = 0.3
        self.box_thresh = 0.5
        self.max_candidates = 1000
        self.unclip_ratio = 1.6
        self.min_size = 3
        self.dilation_kernel = np.array([[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, mask):
        """
        Get boxes from the binarized image predicted by DB.
        :param pred: the binarized image predicted by DB.
        :param mask: new 'pred' after threshold filtering.
        :return: (boxes, the score of each boxes)
        """
        dest_height, dest_width = pred.shape[-2:]
        bitmap = deepcopy(mask)
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours, ), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box):
        """
        Shrink or expand the boxaccording to 'unclip_ratio'
        :param box: The predicted box.
        :return: uncliped box
        """
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        """
        Get boxes from the contour or box.
        :param contour: The predicted contour.
        :return: The predicted box.
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        Calculate the score of box.
        :param bitmap: The binarized image predicted by DB.
        :param _box: The predicted box
        :return: score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    @staticmethod
    def order_points_clockwise(pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        x_sorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most

        right_most = right_most[np.argsort(right_most[:, 1]), :]
        (tr, br) = right_most

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    @staticmethod
    def clip_det_res(points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        return dt_boxes

    def __call__(self, pred, ratio_list, ori_im_shape):
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch = []
        for batch_index in range(pred.shape[0]):

            mask = cv2.dilate(
                np.array(segmentation[batch_index]).astype(np.uint8),
                self.dilation_kernel)
            tmp_boxes, tmp_scores = self.boxes_from_bitmap(pred[batch_index],
                                                           mask)
            boxes = []
            for k in range(len(tmp_boxes)):
                if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
            if len(boxes) > 0:
                boxes = np.array(boxes)
                ratio_h, ratio_w = ratio_list[batch_index]
                boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
                boxes[:, :, 1] = boxes[:, :, 1] / ratio_h
            boxes_batch.append(boxes)
        dt_boxes = self.filter_tag_det_res(boxes_batch[0], ori_im_shape)
        return dt_boxes
