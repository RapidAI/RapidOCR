"""Model URL lookup shared with rapidocr 3.x.

The native build does not ship the ONNX Runtime / OpenVINO / Paddle /
PyTorch / TensorRT / MNN sessions. ``InferSession.get_model_url`` is the
part those sessions and ``download_models`` both call.
"""

from .base import FileInfo, InferSession

__all__ = ["FileInfo", "InferSession"]
