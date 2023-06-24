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
import json
import math
from typing import List

import cv2
import numpy as np

from utils.utils import OrtInferSession


class CTCLabelDecode:
    """Convert between text-label and text-index"""

    def __init__(self, characters: List[str]):
        super(CTCLabelDecode, self).__init__()

        self.characters = characters
        self.characters.append(" ")

        dict_character = self.add_special_char(self.characters)
        self.character = dict_character

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i

    def __call__(self, preds, label=None):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""

        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            # avoid `Mean of empty slice.` warning
            score = np.mean(conf_list) if conf_list else 0
            text = "".join(char_list)
            result_list.append((text, score))
        return result_list


class TextRecognizer:
    def __init__(self, path, config):
        self.rec_batch_num = config.get("rec_batch_num", 6)

        session_instance = OrtInferSession(path)
        self.session = session_instance.session

        metamap = session_instance.session.get_modelmeta().custom_metadata_map
        chars = metamap["dictionary"].splitlines()
        self.postprocess_op = CTCLabelDecode(chars)
        self.rec_image_shape = json.loads(metamap["shape"])
        self.input_name = session_instance.get_input_name()

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
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((img_channel, img_height, img_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list: List[np.ndarray]):
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]

        # Calculate the aspect ratio of all text bars
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        rec_res = [["", 0.0]] * img_num

        batch_num = self.rec_batch_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

            onnx_inputs = {self.input_name: norm_img_batch}
            preds = self.session.run(None, onnx_inputs)[0]
            rec_result = self.postprocess_op(preds)

            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res
