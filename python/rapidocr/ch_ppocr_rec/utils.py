# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np


@dataclass
class TextRecConfig:
    intra_op_num_threads: int = -1
    inter_op_num_threads: int = -1
    use_cuda: bool = False
    use_dml: bool = False
    model_path: Union[str, Path, None] = None

    rec_batch_num: int = 6
    rec_img_shape: Tuple[int, int, int] = (3, 48, 320)
    rec_keys_path: Union[str, Path, None] = None


@dataclass
class TextRecInput:
    img: Union[np.ndarray, List[np.ndarray], None] = None
    return_word_box: bool = False


@dataclass
class TextRecOutput:
    txts: Optional[Tuple[str]] = None
    scores: Tuple[float] = (1.0,)
    word_results: Tuple[Tuple[str, float, Optional[List[List[int]]]]] = (
        ("", 1.0, None),
    )
    elapse: Optional[float] = None

    def __len__(self):
        if self.txts is None:
            return 0
        return len(self.txts)


class CTCLabelDecode:
    def __init__(
        self,
        character: Optional[List[str]] = None,
        character_path: Union[str, Path, None] = None,
    ):
        self.character = self.get_character(character, character_path)
        self.dict = {char: i for i, char in enumerate(self.character)}

    def __call__(
        self, preds: np.ndarray, return_word_box: bool = False, **kwargs
    ) -> Tuple[List[Tuple[str, float]], List[Any]]:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        wh_ratio_list = kwargs.get("wh_ratio_list", (1.0,))
        max_wh_ratio = kwargs.get("max_wh_ratio", 1.0)

        line_results, word_results = self.decode(
            preds_idx,
            preds_prob,
            return_word_box,
            wh_ratio_list,
            max_wh_ratio,
            is_remove_duplicate=True,
        )
        return line_results, word_results

    def get_character(
        self,
        character: Optional[List[str]] = None,
        character_path: Union[str, Path, None] = None,
    ) -> List[str]:
        if character is None and character_path is None:
            raise ValueError("character must not be None")

        character_list = None
        if character:
            character_list = character

        if character is None and character_path is not None:
            character_list = self.read_character_file(character_path)

        if character_list is None:
            raise ValueError("character must not be None")

        character_list = self.insert_special_char(
            character_list, " ", len(character_list)
        )
        character_list = self.insert_special_char(character_list, "blank", 0)
        return character_list

    @staticmethod
    def read_character_file(character_path: Union[str, Path]) -> List[str]:
        character_list = []
        with open(character_path, "rb") as f:
            lines = f.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                character_list.append(line)
        return character_list

    @staticmethod
    def insert_special_char(
        character_list: List[str], special_char: str, loc: int = -1
    ) -> List[str]:
        character_list.insert(loc, special_char)
        return character_list

    def decode(
        self,
        text_index: np.ndarray,
        text_prob: Optional[np.ndarray] = None,
        return_word_box: bool = False,
        wh_ratio_list: Tuple[float] = (1.0,),
        max_wh_ratio: float = 1.0,
        is_remove_duplicate: bool = False,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[Any]]]:
        result_list, result_words_list = [], []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            if text_prob is not None:
                conf_list = np.array(text_prob[batch_idx][selection]).tolist()
                conf_list = [round(conf, 5) for conf in conf_list]
            else:
                conf_list = [1] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0]

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            text = "".join(char_list)

            result_list.append((text, np.mean(conf_list).round(5).tolist()))

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection
                )

                word_len = len(text_index[batch_idx])
                word_len *= wh_ratio_list[batch_idx] / max_wh_ratio

                result_words_list.append(
                    (word_len, word_list, word_col_list, state_list, conf_list)
                )
        return result_list, result_words_list

    @staticmethod
    def get_word_info(
        text: str, selection: np.ndarray
    ) -> Tuple[List[List[str]], List[List[int]], List[str]]:
        """
        Group the decoded characters and record the corresponding decoded positions.
        from https://github.com/PaddlePaddle/PaddleOCR/blob/fbba2178d7093f1dffca65a5b963ec277f1a6125/ppocr/postprocess/rec_postprocess.py#L70

        Args:
            text: the decoded text
            selection: the bool array that identifies which columns of features are decoded as non-separated characters
        Returns:
            word_list: list of the grouped words
            word_col_list: list of decoding positions corresponding to each character in the grouped word
            state_list: list of marker to identify the type of grouping words, including two types of grouping words:
                        - 'cn': continous chinese characters (e.g., 你好啊)
                        - 'en&num': continous english characters (e.g., hello), number (e.g., 123, 1.123), or mixed of them connected by '-' (e.g., VGG-16)
        """
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection)[0]
        col_width = np.zeros(valid_col.shape)
        if len(valid_col) > 0:
            col_width[1:] = valid_col[1:] - valid_col[:-1]
            col_width[0] = min(
                3 if "\u4e00" <= text[0] <= "\u9fff" else 2, int(valid_col[0])
            )

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            else:
                c_state = "en&num"

            if state is None:
                state = c_state

            if state != c_state or col_width[c_i] > 4:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            word_content.append(char)
            word_col_content.append(int(valid_col[c_i]))

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    @staticmethod
    def get_ignored_tokens() -> List[int]:
        return [0]  # for ctc blank
