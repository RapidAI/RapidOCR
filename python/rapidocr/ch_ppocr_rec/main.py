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
import math
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from rapidocr.inference_engine.base import FileInfo, get_engine

from ..utils.download_file import DownloadFile, DownloadFileInput
from ..utils.log import logger
from ..utils.model_resolver import normalize_lang
from ..utils.typings import LangRec
from ..utils.utils import reorder_bidi_for_display, validate_rtl_dependency
from ..utils.vis_res import VisRes
from .typings import TextRecInput, TextRecOutput
from .utils import CTCLabelDecode
from ..utils.doc_classifier import classify as classify_doc_type
from ..utils.postprocess_es import mask_text as mask_text_es

DEFAULT_DICT_PATH = Path(__file__).parent.parent / "models" / "ppocr_keys_v1.txt"
DEFAULT_DICT_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v2.0.7/paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt"
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models"
DEFAULT_ES_MODEL = DEFAULT_MODEL_PATH / "spanish_rec_v18.onnx"
DEFAULT_ES_DICT = DEFAULT_MODEL_PATH / "ppocr_keys_es.txt"
RTL_LANGS = {LangRec.ARABIC.value}


class TextRecognizer:
    def __init__(self, cfg: Dict[str, Any]):
        if normalize_lang(cfg.lang_type) in RTL_LANGS:
            validate_rtl_dependency()

        # Spanish ships its own pruned model + reduced dictionary; use them when
        # the caller selects LangRec.ES without explicit paths.
        if normalize_lang(cfg.lang_type) == LangRec.ES.value:
            if not cfg.get("model_path") and DEFAULT_ES_MODEL.exists():
                cfg["model_path"] = str(DEFAULT_ES_MODEL)
            if not cfg.get("rec_keys_path") and DEFAULT_ES_DICT.exists():
                cfg["rec_keys_path"] = str(DEFAULT_ES_DICT)

        self.session = get_engine(cfg.engine_type)(cfg)

        # onnx has inner character, other engine get or download character_dict_path
        character, character_dict_path = self.get_character_dict(cfg)

        self.postprocess_op = CTCLabelDecode(
            character=character, character_path=character_dict_path
        )

        self.rec_batch_num = cfg["rec_batch_num"]
        self.rec_image_shape = cfg["rec_img_shape"]

        # Long lines are unreadable at the network's fixed input height, so wide
        # crops are cut into pieces before recognition and joined afterwards.
        self.split_wide_crops = cfg.get("split_wide_crops", True)
        self.split_above = cfg.get("split_above", 800)
        self.split_chunk = cfg.get("split_chunk", 240)
        # A cut is rejoined with a space unless the blank run is narrower than
        # this. Prose gains from always inserting the space (measured), while
        # identifier-dense documents (MRZ) prefer a tight join, so a caller can
        # raise it.
        self.split_word_gap = cfg.get("split_word_gap", 6)
        self.split_blocks = cfg.get("split_blocks", False)
        self.split_block_min_height = cfg.get("split_block_min_height", 60)

        self.cfg = cfg

    def get_character_dict(self, cfg):
        character = None
        dict_path = cfg.get("rec_keys_path", None)
        if self.session.have_key():
            character = self.session.get_character_list()
            return character, dict_path

        # onnx has character, other engine need dict_path
        if (not dict_path and not character) or (not Path(dict_path).exists()):
            dict_download_url = self.session.get_dict_key_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            dict_download_url = (
                dict_download_url if dict_download_url is not None else DEFAULT_DICT_URL
            )
            model_root_dir = cfg.get("model_root_dir", None)
            dict_dir = (
                DEFAULT_MODEL_PATH if model_root_dir is None else Path(model_root_dir)
            )
            dict_path = dict_dir / Path(dict_download_url).name
            if not Path(dict_path).exists():
                DownloadFile.run(
                    DownloadFileInput(
                        file_url=dict_download_url,
                        sha256=None,
                        save_path=dict_path,
                        logger=logger,
                    )
                )

        return character, dict_path

    def __call__(self, args: TextRecInput) -> TextRecOutput:
        start_time = time.perf_counter()

        orig_img_list = [args.img] if isinstance(args.img, np.ndarray) else args.img
        # Long lines are unreadable at the network's fixed input height: a line
        # carrying 300 characters ends up with ~3 px per glyph. Crops wider than
        # `split_above` are cut into `split_chunk`-wide pieces (at the whitest
        # column, so no word is sliced) and their transcripts joined afterwards.
        img_list, owners, gaps = self._split_wide_crops(orig_img_list)
        return_word_box = args.return_word_box and len(owners) == len(orig_img_list)

        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        rec_res = [("", 0.0)] * img_num

        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)

            # Parameter Alignment for PaddleOCR
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

            preds = self.session.infer_argmax(norm_img_batch)
            line_results, word_results = self.postprocess_op(
                preds,
                return_word_box,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )

            for rno, one_res in enumerate(line_results):
                if return_word_box:
                    rec_res[indices[beg_img_no + rno]] = (one_res, word_results[rno])
                    continue

                rec_res[indices[beg_img_no + rno]] = (one_res, None)

        all_line_results, all_word_results = list(zip(*rec_res))
        if len(owners) == len(orig_img_list):
            txts, scores = list(zip(*all_line_results))
        else:
            txts, scores = self._join_split(
                all_line_results, owners, gaps, len(orig_img_list), self.split_word_gap
            )
            all_word_results = None

        # Routing por tipo de documento + post-procesado (español multi-documento)
        try:
            doc_type = classify_doc_type("\n".join(txts)).get("type", "OTRO")
            txts = tuple(mask_text_es(t, doc_type) for t in txts)
        except Exception:
            pass

        if normalize_lang(self.cfg.lang_type) in RTL_LANGS:
            txts = reorder_bidi_for_display(txts)

        elapse = time.perf_counter() - start_time
        return TextRecOutput(
            img_list,
            txts,
            scores,
            all_word_results,
            elapse,
            viser=VisRes(lang_type=self.cfg.lang_type, font_path=self.cfg.font_path),
        )

    def _split_wide_crops(self, img_list):
        """Expand crops wider than the limit into readable pieces.

        Returns the pieces, the index of the crop each piece came from, and the
        width of the white gap left of each piece (0 for the first piece and for
        crops that were not split). The gap decides later whether a space must be
        inserted when the transcripts are joined: machine-readable zones (MRZ)
        and identifiers have no word gaps and must be rejoined tight.
        """
        if not self.split_wide_crops:
            return list(img_list), list(range(len(img_list))), [0] * len(img_list)
        expanded, owners, gaps_out = [], [], []
        for index, img in enumerate(img_list):
            if self.split_blocks and img.shape[0] >= self.split_block_min_height:
                bands = self._row_bands(img)
                if len(bands) >= 2:
                    for top, bottom in bands:
                        piece = img[max(0, top - 2) : min(img.shape[0], bottom + 2), :]
                        for sub, gap in self._chunk_crops(piece) if piece.shape[1] > self.split_above else [(piece, 0)]:
                            expanded.append(sub)
                            owners.append(index)
                            gaps_out.append(gap)
                    continue
            if img.shape[1] <= self.split_above:
                expanded.append(img)
                owners.append(index)
                gaps_out.append(0)
                continue
            pieces = list(self._chunk_crops(img))
            if len(pieces) < 2:
                expanded.append(img)
                owners.append(index)
                gaps_out.append(0)
                continue
            for position, (piece, gap) in enumerate(pieces):
                expanded.append(piece)
                owners.append(index)
                gaps_out.append(gap if position else 0)
        return expanded, owners, gaps_out

    @staticmethod
    def _row_bands(img: np.ndarray, min_gap: int = 3):
        """Row ranges holding ink, i.e. the text lines inside a merged block."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (gray < 160).sum(axis=1)
        bands, start, blank = [], None, 0
        for y, filled in enumerate(ink > 0):
            if filled:
                if start is None:
                    start = y
                blank = 0
            elif start is not None:
                blank += 1
                if blank >= min_gap:
                    bands.append((start, y - blank + 1))
                    start, blank = None, 0
        if start is not None:
            bands.append((start, len(ink)))
        return [(a, b) for a, b in bands if b - a >= 6]

    def _chunk_crops(self, img: np.ndarray):
        """Yield (piece, gap_width) cutting on the whitest nearby column."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (gray < 160).sum(axis=0)
        width = img.shape[1]
        start = 0
        chunk = max(32, int(self.split_chunk))
        while start + chunk < width:
            low = max(start + int(chunk * 0.7), start + 1)
            high = min(width - 1, start + chunk + int(chunk * 0.15))
            window = ink[low:high]
            cut = low + int(np.argmin(window)) if window.size else start + chunk
            if cut <= start:
                cut = start + chunk
            # width of the blank run around the cut, i.e. is this a word gap?
            gap = 0
            left = cut - 1
            while left >= 0 and ink[left] == 0:
                gap += 1
                left -= 1
            right = cut
            while right < width and ink[right] == 0:
                gap += 1
                right += 1
            yield img[:, start:cut], gap
            start = cut
        if width - start >= 8:
            yield img[:, start:width], 0

    @staticmethod
    def _join_split(line_results, owners, gaps, count, word_gap=6):
        """Join the transcripts of the pieces of one crop.

        A space is inserted only when the cut fell on a word-sized blank run;
        an identifier or a machine-readable line is rejoined with no separator.
        """
        texts, scores = [], []
        for index in range(count):
            positions = [pos for pos, owner in enumerate(owners) if owner == index]
            built = ""
            for order, pos in enumerate(positions):
                piece_text = line_results[pos][0].strip()
                if not piece_text:
                    continue
                if built and gaps[pos] >= word_gap:
                    built += " "
                built += piece_text
            score = float(np.mean([line_results[p][1] for p in positions])) if positions else 0.0
            texts.append(built)
            scores.append(score)
        return tuple(texts), tuple(scores)

    def resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        img_channel, img_height, img_width = self.rec_image_shape
        assert img_channel == img.shape[2]

        img_width = int(img_height * max_wh_ratio)

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(img_height * ratio))

        resized_image = cv2.resize(img, (resized_w, img_height))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((img_channel, img_height, img_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
