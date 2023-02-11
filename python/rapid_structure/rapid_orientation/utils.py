# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import importlib
from pathlib import Path
import warnings

import cv2
import numpy as np
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions, get_available_providers, get_device)


class OrtInferSession():
    def __init__(self, config):
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = 'CUDAExecutionProvider'
        cpu_ep = 'CPUExecutionProvider'
        cpu_provider_options = {"arena_extend_strategy": "kSameAsRequested", }

        EP_list = []
        if config['use_cuda'] and get_device() == 'GPU' \
                and cuda_ep in get_available_providers():
            EP_list = [(cuda_ep, config[cuda_ep])]
        EP_list.append((cpu_ep, cpu_provider_options))

        self._verify_model(config['model_path'])
        self.session = InferenceSession(config['model_path'],
                                        sess_options=sess_opt,
                                        providers=EP_list)

        has_cuda_ep = cuda_ep not in self.session.get_providers()
        if config['use_cuda'] and has_cuda_ep:
            warnings.warn(f'{cuda_ep} is not avaiable for current env,'
                          f'the inference part is automatically shifted to '
                          f'be executed under {cpu_ep}. '
                          f'Please ensure the installed onnxruntime-gpu '
                          f' version matches your cuda and cudnn version, '
                          f'you can check their relations from the offical web site: '
                          f'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html',
                          RuntimeWarning)

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            raise ONNXRuntimeError('ONNXRuntime inferece failed.') from e

    def get_input_names(self, ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self,):
        return [v.name for v in self.session.get_outputs()]

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f'{model_path} does not exists.')
        if not model_path.is_file():
            raise FileExistsError(f'{model_path} is not a file.')


class ONNXRuntimeError(Exception):
    pass


def create_operators(params):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    mod = importlib.import_module(__name__)
    ops = []
    for operator in params:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = getattr(mod, op_name)(**param)
        ops.append(op)
    return ops


class ResizeImage():
    def __init__(self, size=None, resize_short=None):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w, self.h = None, None
        elif size is not None:
            self.resize_short = None
            self.w = size if isinstance(size, int) else size[0]
            self.h = size if isinstance(size, int) else size[1]
        else:
            raise ValueError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

    def __call__(self, img: np.ndarray):
        img_h, img_w = img.shape[:2]

        if self.resize_short:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return cv2.resize(img, (w, h))


class NormalizeImage():
    def __init__(self,):
        self.scale = np.float32(1.0 / 255.0)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        shape = 1, 1, 3
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img = (img * self.scale - self.mean) / self.std
        return img.astype(np.float32)


class ToCHWImage():
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img)
        return img.transpose((2, 0, 1))


class CropImage():
    def __init__(self, size):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]

        if img_h < h or img_w < w:
            raise ValueError(
                f"The size({h}, {w}) of CropImage must be greater than "
                f"size({img_h}, {img_w}) of image."
            )

        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]
