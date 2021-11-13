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
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

try:
    from .utils import CTCLabelDecode, check_and_read_gif, get_image_file_list
except:
    from utils import CTCLabelDecode, check_and_read_gif, get_image_file_list


root_path = Path(__file__).parent


class TextRecognizer(object):
    def __init__(self, rec_model_path, keys_path=None):
        self.rec_image_shape = [3, 32, 320]
        self.rec_batch_num = 6

        if keys_path is not None:
            self.character_dict_path = keys_path
        else:
            self.character_dict_path = str(root_path / 'ppocr_keys_v1.txt')
        self.postprocess_op = CTCLabelDecode(self.character_dict_path)

        sess_opt = onnxruntime.SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        self.session = onnxruntime.InferenceSession(rec_model_path, sess_opt)

    def resize_norm_img(self, img, max_wh_ratio):
        img_channel, img_height, img_width = self.rec_image_shape
        assert img_channel == img.shape[2]

        img_width = int((32 * max_wh_ratio))
        max_wh_ratio = 1

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(img_height * ratio))
        resized_image = cv2.resize(img, (resized_w, img_height))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((img_channel, img_height, img_width),
                              dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    @staticmethod
    def load_image(image_dir):
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
        if not isinstance(image_dir, list):
            img_list = self.load_image(image_dir)
        else:
            img_list = image_dir
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            starttime = time.time()
            ort_inputs = {self.session.get_inputs()[0].name: norm_img_batch}
            preds = self.session.run(None, ort_inputs)[0]

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            elapse += time.time() - starttime
        return rec_res, elapse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='image_dir|image_path')
    parser.add_argument('--model_path', type=str, help='rec_model_path')
    args = parser.parse_args()

    text_recognizer = TextRecognizer(args.model_path)
    rec_res, predict_time = text_recognizer(args.image_path)
    print(f'识别结果: {rec_res}\t cost: {predict_time}s')
