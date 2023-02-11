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
# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import time
from pathlib import Path
import cv2

import numpy as np
import yaml

from .utils import OrtInferSession, create_operators

root_dir = Path(__file__).resolve().parent


class RapidOrientation():
    def __init__(self, model_path: str = None):
        config_path = str(root_dir / 'config.yaml')
        config = self.read_yaml(config_path)
        if model_path is None:
            model_path = str(root_dir / 'models' / 'rapid_orientation.onnx')
        config['model_path'] = model_path

        self.session = OrtInferSession(config)
        self.labels = self.session.get_metadata()['character'].splitlines()

        self.preprocess_ops = create_operators(config["PreProcess"])

    def __call__(self, images):
        s = time.time()
        for ops in self.preprocess_ops:
            images = ops(images)
        image = np.array(images)[None, ...]

        pred_output = self.session(image)[0]

        pred_output = pred_output.squeeze()
        pred_idx = np.argmax(pred_output)
        pred_txt = self.labels[pred_idx]
        elapse = time.time() - s
        return pred_txt, elapse

    @staticmethod
    def read_yaml(yaml_path):
        with open(yaml_path, 'rb') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '--img_path', type=str, required=True,
                        help='Path to image for layout.')
    parser.add_argument('-m', '--model_path', type=str,
                        default=str(root_dir / 'models' /
                                    'rapid_orientation.onnx'),
                        help='The model path used for inference.')
    args = parser.parse_args()

    orientation_engine = RapidOrientation(args.model_path)

    img = cv2.imread(args.img_path)
    orientaion_result, _ = orientation_engine(img)
    print(orientaion_result)


if __name__ == '__main__':
    main()
