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
import time
from pathlib import Path
import numpy as np

from .utils import (OrtInferSession, PicoDetPostProcess, create_operators,
                    read_yaml, transform)

root_dir = Path(__file__).resolve().parent


class RapidLayout():
    def __init__(self,
                 model_path: str = None):
        config_path = str(root_dir / 'config.yaml')
        config = read_yaml(config_path)
        if model_path is None:
            model_path = str(root_dir / 'models' / 'layout_cdla.onnx')
        config['model_path'] = model_path

        session_instance = OrtInferSession(config)
        self.predictor = session_instance.session
        self.input_name = session_instance.get_input_name()
        labels = session_instance.get_metadata()['character'].splitlines()

        self.preprocess_op = create_operators(config['pre_process'])
        self.postprocess_op = PicoDetPostProcess(labels,
                                                 **config['post_process'])

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]

        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        img = img.copy()

        preds, elapse = 0, 1
        starttime = time.time()

        input_dict = {self.input_name: img}
        preds_onnx = self.predictor.run(None, input_dict)

        np_score_list, np_boxes_list = [], []
        num_outs = int(len(preds_onnx) / 2)
        for out_idx in range(num_outs):
            np_score_list.append(preds_onnx[out_idx])
            np_boxes_list.append(preds_onnx[out_idx + num_outs])
        preds = dict(boxes=np_score_list, boxes_num=np_boxes_list)

        post_preds = self.postprocess_op(ori_im, img, preds)
        elapse = time.time() - starttime
        return post_preds, elapse


def main():
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--model_path', type=str,
                        default=str(root_dir / 'models' / 'layout_cdla.onnx'))
    args = parser.parse_args()

    layout_engine = RapidLayout(args.model_path)

    img = cv2.imread(args.img_path)

    layout_res, elapse = layout_engine(img)

    print(layout_res)


if __name__ == '__main__':
    main()
