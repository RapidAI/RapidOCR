# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import traceback
from pathlib import Path

import MNN
import numpy as np
from omegaconf import DictConfig

from ...utils.download_file import DownloadFile, DownloadFileInput
from ...utils.log import logger
from ..base import FileInfo, InferSession


class MNNInferSession(InferSession):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        model_path = cfg.get("model_path", None)
        if model_path is None:
            model_info = self.get_model_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            model_path = self.DEFAULT_MODEL_PATH / Path(model_info["model_dir"]).name
            DownloadFile.run(
                DownloadFileInput(
                    file_url=model_info["model_dir"],
                    sha256=model_info["SHA256"],
                    save_path=model_path,
                    logger=logger,
                )
            )

        model_path = Path(model_path)
        self._verify_model(model_path)
        self.interpreter = MNN.Interpreter(str(model_path))
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        try:
            self.interpreter.resizeTensor(self.input_tensor, img.shape)
            self.interpreter.resizeSession(self.session)

            tmp = MNN.Tensor(
                img.shape,
                MNN.Halide_Type_Float,
                img,
                MNN.Tensor_DimensionType_Caffe,
            )
            self.input_tensor.copyFromHostTensor(tmp)
            self.interpreter.runSession(self.session)

            output = self.interpreter.getSessionOutput(self.session)
            out_shape = output.getShape()
            out_tensor = MNN.Tensor(
                out_shape, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe
            )
            output.copyToHostTensor(out_tensor)

            return np.array(out_tensor.getData()).reshape(out_shape)

        except Exception as e:
            error_info = traceback.format_exc()
            raise MNNError(f"MNN inference failed: {error_info}") from e

    def have_key(self, key: str = "character") -> bool:
        return False


class MNNError(Exception):
    pass
