# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


class CTCLabelDecode:
    def __init__(
        self,
        character: Optional[List[str]] = None,
        character_path: Union[str, Path, None] = None,
    ):
        self.character = self.get_character(character, character_path)
        self.dict = {char: i for i, char in enumerate(self.character)}

    def __call__(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        return text

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

        if character_path:
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
        is_remove_duplicate: bool = False,
    ) -> List[Tuple[str, float]]:
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list, conf_list = [], []
            cur_pred_ids = text_index[batch_idx]
            for idx, cur_idx in enumerate(cur_pred_ids):
                if cur_idx in ignored_tokens:
                    continue

                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and cur_pred_ids[idx - 1] == cur_idx:
                        continue

                char_list.append(self.character[int(cur_idx)])

                if text_prob is None:
                    conf_list.append(1)
                else:
                    conf_list.append(text_prob[batch_idx][idx])

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list if any(conf_list) else [0])))
        return result_list

    @staticmethod
    def get_ignored_tokens() -> List[int]:
        return [0]  # for ctc blank
