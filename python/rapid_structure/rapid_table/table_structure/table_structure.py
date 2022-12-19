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
import numpy as np

from .utils import (OrtInferSession, TableLabelDecode, TablePreprocess)


class TableStructurer():
    def __init__(self, model_path: str):
        self.preprocess_op = TablePreprocess()

        session_instance = OrtInferSession(model_path)
        self.predictor = session_instance.session
        self.input_name = session_instance.get_input_name()

        self.character = session_instance.get_metadata()
        self.postprocess_op = TableLabelDecode(self.character)

    def __call__(self, img):
        starttime = time.time()
        data = {'image': img}
        data = self.preprocess_op(data)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()

        input_dict = {self.input_name: img}
        outputs = self.predictor.run(None, input_dict)

        preds = {
            'loc_preds': outputs[0],
            'structure_probs': outputs[1]
        }

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        bbox_list = post_result['bbox_batch_list'][0]

        structure_str_list = post_result['structure_batch_list'][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = [
            '<html>', '<body>', '<table>'
        ] + structure_str_list + ['</table>', '</body>', '</html>']
        elapse = time.time() - starttime
        return (structure_str_list, bbox_list), elapse
