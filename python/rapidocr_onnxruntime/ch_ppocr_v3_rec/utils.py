# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import warnings

import numpy as np
import yaml
from onnxruntime import (get_available_providers, get_device,
                         SessionOptions, InferenceSession,
                         GraphOptimizationLevel)


class OrtInferSession(object):
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

        self.session = InferenceSession(config['model_path'],
                                        sess_options=sess_opt,
                                        providers=EP_list)

        if config['use_cuda'] and cuda_ep not in self.session.get_providers():
            warnings.warn(f'{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n'
                          'Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, '
                          'you can check their relations from the offical web site: '
                          'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html',
                          RuntimeWarning)

    def get_input_name(self, input_idx=0):
        return self.session.get_inputs()[input_idx].name

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict


def read_yaml(yaml_path):
    with open(yaml_path, 'rb') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


class CTCLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path):
        super(CTCLabelDecode, self).__init__()

        self.character_str = []
        assert character_dict_path is not None, "character_dict_path should not be None"

        if isinstance(character_dict_path, str):
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
        else:
            self.character_str = character_dict_path
        self.character_str.append(' ')

        dict_character = self.add_special_char(self.character_str)
        self.character = dict_character

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i

    def __call__(self, preds, label=None):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob,
                           is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """

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
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list  + [1e-10] )))
        return result_list
