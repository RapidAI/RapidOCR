# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import imghdr
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import yaml
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions, get_available_providers, get_device)

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path]


class OrtInferSession():
    def __init__(self, config):
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = 'CUDAExecutionProvider'
        cpu_ep = 'CPUExecutionProvider'
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if config['use_cuda'] and get_device() == 'GPU' \
                and cuda_ep in get_available_providers():
            EP_list = [(cuda_ep, config[cuda_ep])]
        EP_list.append((cpu_ep, cpu_provider_options))

        config['model_path'] = str(root_dir / config['model_path'])
        self._verify_model(config['model_path'])
        self.session = InferenceSession(config['model_path'],
                                        sess_options=sess_opt,
                                        providers=EP_list)

        if config['use_cuda'] and cuda_ep not in self.session.get_providers():
            warnings.warn(f'{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n'
                          'Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, '
                          'you can check their relations from the offical web site: '
                          'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html',
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

    def get_character_list(self, key: str = 'character'):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = 'character') -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f'{model_path} does not exists.')
        if not model_path.is_file():
            raise FileExistsError(f'{model_path} is not a file.')


class ONNXRuntimeError(Exception):
    pass


def read_yaml(yaml_path):
    with open(yaml_path, 'rb') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


class LoadImage():
    def __init__(self, ):
        self.valid_format = ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp']
        self.gif_format = ['gif']
        self.ndarray_format = ['ndarray']

    def __call__(self, img: InputType):
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f'The img type {type(img)} does not in [str, np.ndarray, bytes]')

        img = self.load_direct_img(img)
        img = self.convert_three_channels(img)
        return img

    def load_direct_img(self, img: InputType) -> np.ndarray:
        img_suffix = self._which_format(img)

        if img_suffix in self.gif_format:
            if isinstance(img, (str, Path)):
                return self._read_gif_str(img)

            if isinstance(img, bytes):
                return self._read_gif_bytes(img)
        elif img_suffix in self.valid_format:
            if isinstance(img, (str, Path)):
                return self._read_valid_img_str(img)

            if isinstance(img, bytes):
                return self._read_valid_img_bytes(img)
        elif img_suffix in self.ndarray_format:
            return img
        else:
            raise LoadImageError(f'{img_suffix} is not supported!')

    def convert_three_channels(self, img_content: np.ndarray) -> np.ndarray:
        img_shape_len = len(img_content.shape)
        if img_shape_len == 2:
            return cv2.cvtColor(img_content, cv2.COLOR_GRAY2BGR)

        if img_content.shape[2] == 4:
            # RGBA → RGB
            return self._read_transparent_png(img_content)

        return img_content

    def _which_format(self, img_content: InputType) -> Optional[str]:
        if isinstance(img_content, (str, Path)):
            with open(img_content, 'rb') as f:
                return imghdr.what(f).lower()

        if isinstance(img_content, bytes):
            with tempfile.NamedTemporaryFile(delete=True) as fp:
                fp.write(img_content)
                fp.seek(0)
                return imghdr.what(fp).lower()

        if isinstance(img_content, np.ndarray):
            return self.ndarray_format[0]

    def _read_gif_str(self, img: Union[str, Path]) -> np.ndarray:
        @contextmanager
        def video_capture(img):
            cap = cv2.VideoCapture(str(img))
            try:
                yield cap
            finally:
                cap.release()

        with video_capture(img) as cap:
            ret, frame = cap.read()
        if not ret:
            raise LoadImageError('OpenCV does not read the gif img')

        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        img = frame[:, :, ::-1]
        return img

    def _read_valid_img_str(self, img: Union[str, Path]) -> np.ndarray:
        with open(img, 'rb') as f:
            img = self._read_valid_img_bytes(f.read())
        return img

    def _read_gif_bytes(self, img: bytes) -> np.ndarray:
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            fp.write(img)
            fp.seek(0)
            img = self._read_gif_str(fp)
        return img

    def _read_valid_img_bytes(self, img: bytes) -> np.ndarray:
        np_arr = np.frombuffer(img, dtype=np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            from io import BytesIO
            from PIL import Image, UnidentifiedImageError

            bytes_img = BytesIO(img)
            try:
                cv_img = np.array(Image.open(bytes_img))
            except (UnidentifiedImageError, ValueError) as e:
                raise LoadImageError('read_valid_img_bytes has errors.') from e
        return cv_img

    @staticmethod
    def _read_transparent_png(image_4channel: np.ndarray) -> np.ndarray:
        if image_4channel.shape[-1] != 4:
            return image_4channel
        alpha_channel = image_4channel[:, :, 3]
        rgb_channels = image_4channel[:, :, :3]

        # White Background Image
        white_bg_img = np.ones_like(rgb_channels, dtype=np.uint8) * 255

        # Alpha factor
        alpha_channel = alpha_channel[:, :, np.newaxis].astype(np.float32)
        alpha_factor = alpha_channel / 255.0
        alpha_factor = np.concatenate(
            (alpha_factor, alpha_factor, alpha_factor), axis=2)

        # Transparent Image Rendered on White Background
        base = rgb_channels.astype(np.float32) * alpha_factor
        white = white_bg_img.astype(np.float32) * (1 - alpha_factor)
        final_image = base + white
        return final_image.astype(np.uint8)


class LoadImageError(Exception):
    pass
