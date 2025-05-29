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
import math
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from rapidocr.inference_engine.base import FileInfo, get_engine

from ..utils import Logger
from ..utils.download_file import DownloadFile, DownloadFileInput
from .typings import TextRecInput, TextRecOutput
from .utils import CTCLabelDecode

DEFAULT_DICT_PATH = Path(__file__).parent.parent / "models" / "ppocr_keys_v1.txt"
DEFAULT_DICT_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v2.0.7/paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt"
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models"


class TextRecognizer:
    def __init__(self, cfg: Dict[str, Any]):
        self.session = get_engine(cfg.engine_type)(cfg)
        self.logger = Logger(logger_name=__name__).get_log()

        # onnx has inner character, other engine get or download character_dict_path
        character, character_dict_path = self.get_character_dict(cfg)

        self.postprocess_op = CTCLabelDecode(
            character=character, character_path=character_dict_path
        )

        self.rec_batch_num = cfg["rec_batch_num"]
        self.rec_image_shape = cfg["rec_img_shape"]

    def get_character_dict(self, cfg):
        character = None
        dict_path = cfg.get("rec_keys_path", None)
        if self.session.have_key():
            character = self.session.get_character_list()
            return character, dict_path

        # onnx has character, other engine need dict_path
        if (not dict_path and not character) or (not Path(dict_path).exists()):
            dict_download_url = self.session.get_dict_key_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            dict_download_url = (
                dict_download_url if dict_download_url is not None else DEFAULT_DICT_URL
            )
            dict_path = DEFAULT_MODEL_PATH / Path(dict_download_url).name
            if not Path(dict_path).exists():
                DownloadFile.run(
                    DownloadFileInput(
                        file_url=dict_download_url,
                        sha256=None,
                        save_path=dict_path,
                        logger=self.logger,
                    )
                )

        return character, dict_path

    def __call__(self, args: TextRecInput) -> TextRecOutput:
        img_list = [args.img] if isinstance(args.img, np.ndarray) else args.img
        return_word_box = args.return_word_box

        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        rec_res = [("", 0.0)] * img_num

        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)

            # Parameter Alignment for PaddleOCR
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

            start_time = time.perf_counter()
            preds = self.session(norm_img_batch)
            line_results, word_results = self.postprocess_op(
                preds,
                return_word_box,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )

            for rno, one_res in enumerate(line_results):
                if return_word_box:
                    rec_res[indices[beg_img_no + rno]] = (one_res, word_results[rno])
                    continue

                rec_res[indices[beg_img_no + rno]] = (one_res, None)
            elapse += time.perf_counter() - start_time

        all_line_results, all_word_results = list(zip(*rec_res))
        txts, scores = list(zip(*all_line_results))
        return TextRecOutput(img_list, txts, scores, all_word_results, elapse)

    def resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        img_channel, img_height, img_width = self.rec_image_shape
        assert img_channel == img.shape[2]

        img_width = int(img_height * max_wh_ratio)

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
