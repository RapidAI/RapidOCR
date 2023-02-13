# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import numpy as np
import yaml
from openvino.runtime import Core

root_dir = Path(__file__).resolve().parent


class OpenVINOInferSession():
    def __init__(self, config):
        ie = Core()

        config['model_path'] = str(root_dir / config['model_path'])
        self._verify_model(config['model_path'])
        model_onnx = ie.read_model(config['model_path'])
        compile_model = ie.compile_model(model=model_onnx, device_name='CPU')
        self.session = compile_model.create_infer_request()

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        self.session.infer(inputs=[input_content])
        return self.session.get_output_tensor().data

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f'{model_path} does not exists.')
        if not model_path.is_file():
            raise FileExistsError(f'{model_path} is not a file.')


def read_yaml(yaml_path):
    with open(yaml_path, 'rb') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data
