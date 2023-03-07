# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import imghdr
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import yaml
from openvino.runtime import Core

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path]


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


def read_yaml(yaml_path):
    with open(yaml_path, 'rb') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


def concat_model_path(config):
    key = 'model_path'
    config['Det'][key] = str(root_dir / config['Det'][key])
    config['Rec'][key] = str(root_dir / config['Rec'][key])
    config['Cls'][key] = str(root_dir / config['Cls'][key])
    return config


class ParseArgs():
    def __init__(self, ):
        self.args = self.init_args()
        self.args_dict = vars(self.args)

    def init_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-img', '--img_path', type=str, default=None,
                            required=True)
        parser.add_argument('-p', '--print_cost',
                            action='store_true', default=False)

        global_group = parser.add_argument_group(title='Global')
        global_group.add_argument('--text_score', type=float, default=0.5)
        global_group.add_argument('--use_angle_cls', type=bool, default=True)
        global_group.add_argument('--use_text_det', type=bool, default=True)
        global_group.add_argument('--print_verbose', type=bool, default=False)
        global_group.add_argument('--min_height', type=int, default=30)
        global_group.add_argument('--width_height_ratio', type=int, default=8)

        det_group = parser.add_argument_group(title='Det')
        det_group.add_argument('--det_model_path', type=str, default=None)
        det_group.add_argument('--det_limit_side_len', type=float, default=736)
        det_group.add_argument('--det_limit_type', type=str, default='min',
                               choices=['max', 'min'])
        det_group.add_argument('--det_thresh', type=float, default=0.3)
        det_group.add_argument('--det_box_thresh', type=float, default=0.5)
        det_group.add_argument('--det_unclip_ratio', type=float, default=1.6)
        det_group.add_argument('--det_use_dilation', type=bool, default=True)
        det_group.add_argument('--det_score_mode', type=str, default='fast',
                               choices=['slow', 'fast'])

        cls_group = parser.add_argument_group(title='Cls')
        cls_group.add_argument('--cls_model_path', type=str, default=None)
        cls_group.add_argument('--cls_image_shape', type=list,
                               default=[3, 48, 192])
        cls_group.add_argument('--cls_label_list', type=list,
                               default=['0', '180'])
        cls_group.add_argument('--cls_batch_num', type=int, default=6)
        cls_group.add_argument('--cls_thresh', type=float, default=0.9)

        rec_group = parser.add_argument_group(title='Rec')
        rec_group.add_argument('--rec_model_path', type=str, default=None)
        rec_group.add_argument('--rec_image_shape', type=list,
                               default=[3, 48, 320])
        rec_group.add_argument('--rec_batch_num', type=int, default=6)

        args = parser.parse_args()
        return args

    def parse_kwargs(self, **kwargs):
        global_dict, det_dict, cls_dict, rec_dict = {}, {}, {}, {}
        for k, v in kwargs.items():
            if k.startswith('det'):
                det_dict[k] = v
            elif k.startswith('cls'):
                cls_dict[k] = v
            elif k.startswith('rec'):
                rec_dict[k] = v
            else:
                global_dict[k] = v
        return global_dict, det_dict, cls_dict, rec_dict

    def update_config(self, config, **kwargs):
        global_dict, det_dict, cls_dict, rec_dict = self.parse_kwargs(**kwargs)
        new_config = {
            'Global': self.update_global_params(config['Global'],
                                                global_dict),
            'Det': self.update_det_params(config['Det'], det_dict),
            'Cls': self.update_cls_params(config['Cls'], cls_dict),
            'Rec': self.update_rec_params(config['Rec'], rec_dict)
        }
        return new_config

    def update_global_params(self, config, global_dict):
        config.update(global_dict)
        return config

    def update_det_params(self, config, det_dict):
        det_dict = {k.split('det_')[1]: v for k, v in det_dict.items()}
        if not det_dict['model_path']:
            det_dict['model_path'] = str(root_dir / config['model_path'])
        config.update(det_dict)
        return config

    def update_cls_params(self, config, cls_dict):
        need_remove_prefix = ['cls_label_list', 'cls_model_path']
        new_cls_dict = {}
        for k, v in cls_dict.items():
            if k in need_remove_prefix:
                k = k.split('cls_')[1]
            new_cls_dict[k] = v

        if not new_cls_dict['model_path']:
            new_cls_dict['model_path'] = str(root_dir / config['model_path'])
        config.update(new_cls_dict)
        return config

    def update_rec_params(self, config, rec_dict):
        need_remove_prefix = ['rec_model_path']
        new_rec_dict = {}
        for k, v in rec_dict.items():
            if k in need_remove_prefix:
                k = k.split('rec_')[1]
            new_rec_dict[k] = v

        if not new_rec_dict['model_path']:
            new_rec_dict['model_path'] = str(root_dir / config['model_path'])
        config.update(new_rec_dict)
        return config
