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
from pathlib import Path

import cv2
import numpy as np
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)


class OrtInferSession():
    def __init__(self, onnx_path: str):
        self.__verify_model(onnx_path)

        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_ep = 'CPUExecutionProvider'
        cpu_provider_options = {"arena_extend_strategy": "kSameAsRequested",}
        EP_list = [(cpu_ep, cpu_provider_options)]

        self.session = InferenceSession(onnx_path,
                                        sess_options=sess_opt,
                                        providers=EP_list)

    def get_input_name(self, input_idx=0):
        return self.session.get_inputs()[input_idx].name

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name

    def get_metadata(self, key: str = 'character') -> list:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        content_list = meta_dict[key].splitlines()
        return content_list

    @staticmethod
    def __verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f'{model_path} does not exists.')
        if not model_path.is_file():
            raise FileExistsError(f'{model_path} is not a file.')


class TableLabelDecode():
    """  """
    def __init__(self,
                 dict_character,
                 merge_no_span_structure=True,
                 **kwargs):
        if merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        self.td_token = ['<td>', '<td', '<td></td>']

    def __call__(self, preds, batch=None):
        structure_probs = preds['structure_probs']
        bbox_preds = preds['loc_preds']
        shape_list = batch[-1]
        result = self.decode(structure_probs, bbox_preds, shape_list)
        if len(batch) == 1:  # only contains shape
            return result

        label_decode_result = self.decode_label(batch)
        return result, label_decode_result

    def decode(self, structure_probs, bbox_preds, shape_list):
        """convert text-label into text-index.
        """
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            score_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break

                if char_idx in ignored_tokens:
                    continue

                text = self.character[char_idx]
                if text in self.td_token:
                    bbox = bbox_preds[batch_idx, idx]
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_batch_list.append([structure_list, np.mean(score_list)])
            bbox_batch_list.append(np.array(bbox_list))
        result = {
            'bbox_batch_list': bbox_batch_list,
            'structure_batch_list': structure_batch_list,
        }
        return result

    def decode_label(self, batch):
        """convert text-label into text-index.
        """
        structure_idx = batch[1]
        gt_bbox_list = batch[2]
        shape_list = batch[-1]
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                structure_list.append(self.character[char_idx])

                bbox = gt_bbox_list[batch_idx][idx]
                if bbox.sum() != 0:
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
            structure_batch_list.append(structure_list)
            bbox_batch_list.append(bbox_list)
        result = {
            'bbox_batch_list': bbox_batch_list,
            'structure_batch_list': structure_batch_list,
        }
        return result

    def _bbox_decode(self, bbox, shape):
        h, w, ratio_h, ratio_w, pad_h, pad_w = shape
        bbox[0::2] *= w
        bbox[1::2] *= h
        return bbox

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character


class TablePreprocess():
    def __init__(self):
        self.table_max_len = 488
        self.build_pre_process_list()
        self.ops = self.create_operators()

    def __call__(self, data):
        """ transform """
        if self.ops is None:
            self.ops = []

        for op in self.ops:
            data = op(data)
            if data is None:
                return None
        return data

    def create_operators(self,):
        """
        create operators based on the config

        Args:
            params(list): a dict list, used to create some operators
        """
        assert isinstance(self.pre_process_list, list), ('operator config should be a list')
        ops = []
        for operator in self.pre_process_list:
            assert isinstance(operator,
                            dict) and len(operator) == 1, "yaml format error"
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]
            op = eval(op_name)(**param)
            ops.append(op)
        return ops

    def build_pre_process_list(self):
        resize_op = {'ResizeTableImage': {'max_len': self.table_max_len, }}
        pad_op = {
            'PaddingTableImage': {
                'size': [self.table_max_len, self.table_max_len]
            }
        }
        normalize_op = {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }
        to_chw_op = {'ToCHWImage': None}
        keep_keys_op = {'KeepKeys': {'keep_keys': ['image', 'shape']}}
        self.pre_process_list = [
            resize_op, normalize_op, pad_op, to_chw_op, keep_keys_op
        ]


class ResizeTableImage():
    def __init__(self, max_len, resize_bboxes=False, infer_mode=False,
                **kwargs):
        super(ResizeTableImage, self).__init__()
        self.max_len = max_len
        self.resize_bboxes = resize_bboxes
        self.infer_mode = infer_mode

    def __call__(self, data):
        img = data['image']
        height, width = img.shape[0:2]
        ratio = self.max_len / (max(height, width) * 1.0)
        resize_h = int(height * ratio)
        resize_w = int(width * ratio)
        resize_img = cv2.resize(img, (resize_w, resize_h))
        if self.resize_bboxes and not self.infer_mode:
            data['bboxes'] = data['bboxes'] * ratio
        data['image'] = resize_img
        data['src_img'] = img
        data['shape'] = np.array([height, width, ratio, ratio])
        data['max_len'] = self.max_len
        return data


class PaddingTableImage():
    def __init__(self, size, **kwargs):
        super(PaddingTableImage, self).__init__()
        self.size = size

    def __call__(self, data):
        img = data['image']
        pad_h, pad_w = self.size
        padding_img = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        height, width = img.shape[0:2]
        padding_img[0:height, 0:width, :] = img.copy()
        data['image'] = padding_img
        shape = data['shape'].tolist()
        shape.extend([pad_h, pad_w])
        data['shape'] = np.array(shape)
        return data


class NormalizeImage():
    """ normalize image such as substract mean, divide std
    """
    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = np.array(data['image'])
        assert isinstance(img,
                        np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage():
    """ convert hwc image to chw image
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = np.array(data['image'])
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys():
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list
