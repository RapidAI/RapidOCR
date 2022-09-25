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
import warnings

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


def read_yaml(yaml_path):
    with open(yaml_path, 'rb') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


class ClsPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, label_list):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds, label=None):
        pred_idxs = preds.argmax(axis=1)
        decode_out = [(self.label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out

        label = [(self.label_list[idx], 1.0) for idx in label]
        return decode_out, label