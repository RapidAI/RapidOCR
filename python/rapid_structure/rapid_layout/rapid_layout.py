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
import time
from pathlib import Path

import cv2
import numpy as np

from .utils import (OrtInferSession, PicoDetPostProcess, create_operators,
                    read_yaml, transform, vis_layout)

root_dir = Path(__file__).resolve().parent


class RapidLayout():
    def __init__(self,
                 model_path: str = None):
        config_path = str(root_dir / 'config.yaml')
        config = read_yaml(config_path)
        if model_path is None:
            model_path = str(root_dir / 'models' / 'layout_cdla.onnx')
        config['model_path'] = model_path

        self.session = OrtInferSession(config)
        labels = self.session.get_metadata()['character'].splitlines()

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

        preds = self.session(img)

        score_list, boxes_list = [], []
        num_outs = int(len(preds) / 2)
        for out_idx in range(num_outs):
            score_list.append(preds[out_idx])
            boxes_list.append(preds[out_idx + num_outs])
        preds = dict(boxes=score_list, boxes_num=boxes_list)
        post_preds = self.postprocess_op(ori_im, img, preds)
        elapse = time.time() - starttime
        return post_preds, elapse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vis', action='store_true',
                        help='Wheter to visualize the layout results.')
    parser.add_argument('-img', '--img_path', type=str, required=True,
                        help='Path to image for layout.')
    parser.add_argument('-m', '--model_path', type=str,
                        default=str(root_dir / 'models' / 'layout_cdla.onnx'),
                        help='The model path used for inference.')
    args = parser.parse_args()

    layout_engine = RapidLayout(args.model_path)

    img = cv2.imread(args.img_path)
    layout_res, elapse = layout_engine(img)
    print(layout_res)

    if args.vis:
        img_path = Path(args.img_path)
        save_path = img_path.resolve().parent / f'vis_{img_path.name}'
        vis_layout(img, layout_res, str(save_path))


if __name__ == '__main__':
    main()
