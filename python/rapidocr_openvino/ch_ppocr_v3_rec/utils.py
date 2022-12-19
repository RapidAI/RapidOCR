# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
import numpy as np
import yaml
from openvino.runtime import Core


root_dir = Path(__file__).resolve().parent.parent


class OpenVINOInferSession():
    def __init__(self, config):
        ie = Core()

        config['model_path'] = str(root_dir / config['model_path'])
        self._verify_model(config['model_path'])
        model_onnx = ie.read_model(config['model_path'])
        compile_model = ie.compile_model(model=model_onnx, device_name='CPU')
        self.session = compile_model.create_infer_request()

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


class CTCLabelDecode():
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path):
        super(CTCLabelDecode, self).__init__()

        self.character_str = []
        assert character_dict_path is not None, "character_dict_path should not be None"
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
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
            result_list.append((text, np.mean(conf_list + [1e-50])))
        return result_list
