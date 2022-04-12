# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import copy
import math
import time

import cv2
import numpy as np
import onnxruntime as ort

try:
    from .utils import ClsPostProcess, check_and_read_gif, get_image_file_list
except:
    from utils import ClsPostProcess, check_and_read_gif, get_image_file_list


class TextClassifier(object):
    def __init__(self, cls_model_path):
        self.cls_image_shape = [3, 48, 192]
        self.cls_batch_num = 6
        self.cls_thresh = 0.9
        self.postprocess_op = ClsPostProcess(label_list=['0', '180'])

        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(cls_model_path, sess_opt)

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def load_image(self, image_dir):
        image_file_list = get_image_file_list(image_dir)
        valid_image_file_list = []
        img_list = []
        for image_file in image_file_list:
            img, flag = check_and_read_gif(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                print(f"error in loading image:{image_file}")
                continue
            valid_image_file_list.append(image_file)
            img_list.append(img)
        return img_list

    def __call__(self, image_dir):
        if isinstance(image_dir[0], (list, str)):
            img_list = self.load_image(image_dir)
        else:
            img_list = image_dir

        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)

            starttime = time.time()
            onnx_inputs = {self.session.get_inputs()[0].name: norm_img_batch}
            prob_out = self.session.run(None, onnx_inputs)[0]

            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - starttime

            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if '180' in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1)
        return img_list, cls_res, elapse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='image_dir|image_path')
    parser.add_argument('--model_path', type=str, help='model_path')
    args = parser.parse_args()

    text_classifier = TextClassifier(args.model_path)

    img_list, cls_res, predict_time = text_classifier(args.image_path)
    for ino in range(len(img_list)):
        print(f"分类结果:{cls_res[ino]}")
