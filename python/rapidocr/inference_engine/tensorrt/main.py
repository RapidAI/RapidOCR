# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
"""
TensorRT Inference Session for RapidOCR.

This module provides TensorRT-based inference for OCR models, offering
significant performance improvements over ONNX Runtime on NVIDIA GPUs.

Key Features:
- Automatic engine building and caching
- Dynamic shape support via optimization profiles
- Pre-allocated buffers for minimal inference overhead
- Optional pinned memory for faster data transfers

Performance Optimizations:
1. Pre-allocated buffers with max shape (avoids reallocation overhead)
2. Pinned memory for faster CPU-GPU transfers (~2x on discrete GPUs)
3. Persistent CUDA stream (no stream creation per inference)
4. Async memory copies overlapped with computation

Example:
    >>> from rapidocr.inference_engine.tensorrt import TRTInferSession
    >>> with TRTInferSession(config) as session:
    ...     output = session(input_array)
"""
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from cuda.bindings import runtime as cudart
import tensorrt as trt

from ...utils.download_file import DownloadFile, DownloadFileInput
from ...utils.log import logger
from ...utils.typings import EngineType
from ..base import FileInfo, InferSession
from .engine_builder import TRTEngineBuilder
from .memory_utils import allocate_buffers, free_buffers


class TRTInferSession(InferSession):
    """TensorRT Inference Session for RapidOCR.

    This class provides GPU-accelerated inference using NVIDIA TensorRT.
    It manages engine loading/building, memory allocation, and inference
    execution with optimizations for minimal latency.

    Supports context manager protocol for automatic resource cleanup:
        >>> with TRTInferSession(cfg) as session:
        ...     result = session(input_data)

    Attributes:
        cfg: Configuration dictionary.
        engine: TensorRT engine instance.
        context: TensorRT execution context.
        stream: CUDA stream for async operations.
        inputs: List of input buffer objects.
        outputs: List of output buffer objects.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize TensorRT inference session.

        Args:
            cfg: Configuration dictionary containing:
                - engine_cfg: TensorRT-specific settings (device_id, precision, etc.)
                - model_path: Optional path to custom ONNX model
                - task_type, lang_type, etc.: For default model selection

        Raises:
            AssertionError: If CUDA device setup fails.
            RuntimeError: If engine building fails.
        """
        self.cfg = cfg
        self.engine_cfg = cfg.get("engine_cfg", {})
        self._closed = False

        # Initialize CUDA device
        self.device_id = self._setup_cuda_device()

        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Get or build engine
        engine_path = self._get_engine_path(cfg)
        self.engine = self._load_or_build_engine(cfg, engine_path)

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Allocate memory buffers (pre-allocated with max shape)
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.engine, self.context
        )

        logger.info(f"TensorRT engine loaded: {engine_path}")

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    def __enter__(self) -> "TRTInferSession":
        """Enter context manager.

        Returns:
            self: The session instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and cleanup resources.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.close()

    def close(self) -> None:
        """Explicitly close and cleanup resources.

        This method releases all CUDA resources including:
        - GPU memory buffers
        - CUDA stream
        - TensorRT context and engine

        Safe to call multiple times.

        Example:
            >>> session = TRTInferSession(cfg)
            >>> try:
            ...     result = session(data)
            ... finally:
            ...     session.close()
        """
        if self._closed:
            return

        self._closed = True

        try:
            # Synchronize stream before cleanup
            if hasattr(self, "stream") and self.stream is not None:
                try:
                    cudart.cudaStreamSynchronize(self.stream)
                except Exception as e:
                    logger.debug(f"Stream sync error during close: {e}")

            # Free GPU memory buffers
            if hasattr(self, "inputs") and hasattr(self, "outputs"):
                try:
                    free_buffers(self.inputs, self.outputs, self.stream)
                except Exception as e:
                    logger.debug(f"Buffer free error during close: {e}")

            # Destroy CUDA stream
            if hasattr(self, "stream") and self.stream is not None:
                try:
                    cudart.cudaStreamDestroy(self.stream)
                except Exception as e:
                    logger.debug(f"Stream destroy error during close: {e}")

        except Exception as e:
            logger.debug(f"Error during session close: {e}")

    def __del__(self):
        """Destructor - cleanup resources if not already closed."""
        self.close()

    # =========================================================================
    # Inference Methods
    # =========================================================================

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        """Run inference on input data.

        This method executes the TensorRT engine on the provided input.
        It uses pre-allocated buffers to avoid memory allocation overhead.

        Args:
            input_content: Input numpy array with shape matching the model.
                          For detection: (N, C, H, W)
                          For recognition: (N, C, H, W)

        Returns:
            Output numpy array from the model.

        Raises:
            TensorRTError: If inference fails.

        Example:
            >>> input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            >>> output = session(input_data)
        """
        try:
            # Step 1: Set input shape for dynamic dimensions
            output_shape = self._set_input_shape(input_content)

            # Step 2: Copy input data to GPU
            self._copy_input_to_device(input_content)

            # Step 3: Execute inference
            self._execute_inference()

            # Step 4: Copy output back to CPU and return
            return self._copy_output_to_host(output_shape)

        except Exception as e:
            error_info = traceback.format_exc()
            raise TensorRTError(
                f"Inference failed for input shape {input_content.shape}:\n{error_info}"
            ) from e

    def _set_input_shape(self, input_content: np.ndarray) -> tuple:
        """Set input shape and get corresponding output shape.

        For dynamic shape networks, we need to inform TensorRT of the
        actual input shape before each inference.

        Args:
            input_content: Input array to get shape from.

        Returns:
            Output shape tuple after setting input shape.
        """
        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, input_content.shape)

        output_name = self.engine.get_tensor_name(1)
        return self.context.get_tensor_shape(output_name)

    def _copy_input_to_device(self, input_content: np.ndarray) -> None:
        """Copy input data to GPU asynchronously.

        Uses pre-allocated pinned memory buffer for optimal transfer speed.

        Args:
            input_content: Input data to copy.
        """
        input_flat = input_content.ravel()
        self.inputs[0].host[: input_flat.size] = input_flat

        cudart.cudaMemcpyAsync(
            self.inputs[0].device,
            self.inputs[0].host.ctypes.data,
            input_flat.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )

    def _execute_inference(self) -> None:
        """Execute TensorRT inference asynchronously."""
        self.context.execute_async_v3(stream_handle=self.stream)

    def _copy_output_to_host(self, output_shape: tuple) -> np.ndarray:
        """Copy output from GPU to CPU and reshape.

        Only copies the actual output size, not the full pre-allocated buffer.

        Args:
            output_shape: Expected output shape.

        Returns:
            Output array reshaped to correct dimensions.
        """
        output_size = int(np.prod(output_shape))
        output_nbytes = output_size * self.outputs[0].host.itemsize

        cudart.cudaMemcpyAsync(
            self.outputs[0].host.ctypes.data,
            self.outputs[0].device,
            output_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )

        # Wait for all async operations to complete
        cudart.cudaStreamSynchronize(self.stream)

        return self.outputs[0].host[:output_size].reshape(output_shape)

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _setup_cuda_device(self) -> int:
        """Setup CUDA device for inference.

        Returns:
            Device ID that was set.

        Raises:
            AssertionError: If CUDA device setup fails.
        """
        device_id = self.engine_cfg.get("device_id", 0)
        status_tuple = cudart.cudaSetDevice(device_id)
        status = status_tuple[0]
        assert status.value == 0, (
            f"Failed to set CUDA device {device_id}: {status}. "
            f"Ensure CUDA is properly installed and GPU is available."
        )
        return device_id

    def _get_engine_path(self, cfg: Dict[str, Any]) -> Path:
        """Determine the TensorRT engine file path.

        Engine files are cached per GPU architecture and precision setting
        to avoid rebuilding on subsequent runs.

        Args:
            cfg: Configuration dictionary.

        Returns:
            Path to engine file (may not exist yet).
        """
        cache_dir = self.engine_cfg.get("cache_dir")
        if cache_dir is None:
            cache_dir = self.DEFAULT_MODEL_PATH

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = self._get_model_name(cfg)
        gpu_arch = self._get_gpu_arch()
        precision = "fp16" if self.engine_cfg.get("use_fp16", True) else "fp32"

        return cache_dir / f"{model_name}_{gpu_arch}_{precision}.engine"

    def _get_model_name(self, cfg: Dict[str, Any]) -> str:
        """Extract model name from config for engine filename."""
        if cfg.get("model_path"):
            return Path(cfg["model_path"]).stem

        task_type = cfg.task_type.value
        lang_type = cfg.lang_type.value
        ocr_version = cfg.ocr_version.value
        return f"{lang_type}_{ocr_version}_{task_type}"

    def _get_gpu_arch(self) -> str:
        """Get GPU architecture string for cache key (e.g., 'sm87')."""
        status, major = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, self.device_id
        )
        assert status.value == 0, f"Failed to get compute capability: {status}"

        status, minor = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, self.device_id
        )
        assert status.value == 0, f"Failed to get compute capability: {status}"

        return f"sm{major}{minor}"

    def _load_or_build_engine(
        self, cfg: Dict[str, Any], engine_path: Path
    ) -> trt.ICudaEngine:
        """Load cached engine or build new one from ONNX.

        Args:
            cfg: Configuration dictionary.
            engine_path: Path where engine should be cached.

        Returns:
            TensorRT engine instance.
        """
        force_rebuild = self.engine_cfg.get("force_rebuild", False)

        # Try to load cached engine
        if engine_path.exists() and not force_rebuild:
            try:
                return self._load_engine(engine_path)
            except Exception as e:
                logger.warning(
                    f"Failed to load cached engine {engine_path}: {e}. "
                    f"Will rebuild from ONNX."
                )

        # Build new engine from ONNX
        onnx_path = self._get_onnx_path(cfg)
        logger.info(f"Building TensorRT engine from {onnx_path}")

        builder = TRTEngineBuilder(
            onnx_path=onnx_path,
            engine_path=engine_path,
            cfg=self.engine_cfg,
            task_type=cfg.task_type.value,
            trt_logger=self.trt_logger,
        )
        return builder.build()

    def _get_onnx_path(self, cfg: Dict[str, Any]) -> Path:
        """Get ONNX model path, downloading if necessary."""
        model_path = cfg.get("model_path")

        if model_path is None:
            # Download default ONNX model
            original_engine_type = cfg.engine_type
            cfg.engine_type = EngineType.ONNXRUNTIME

            model_info = self.get_model_url(
                FileInfo(
                    engine_type=EngineType.ONNXRUNTIME,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )

            cfg.engine_type = original_engine_type

            model_path = self.DEFAULT_MODEL_PATH / Path(model_info["model_dir"]).name
            download_params = DownloadFileInput(
                file_url=model_info["model_dir"],
                sha256=model_info["SHA256"],
                save_path=model_path,
                logger=logger,
            )
            DownloadFile.run(download_params)

        model_path = Path(model_path)
        self._verify_model(model_path)
        return model_path

    def _load_engine(self, engine_path: Path) -> trt.ICudaEngine:
        """Load a serialized TensorRT engine from disk."""
        runtime = trt.Runtime(self.trt_logger)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        return runtime.deserialize_cuda_engine(engine_data)

    # =========================================================================
    # Interface Methods (required by InferSession)
    # =========================================================================

    def have_key(self, key: str = "character") -> bool:
        """Check if engine has metadata key.

        TensorRT engines don't store custom metadata like ONNX models.

        Returns:
            Always False for TensorRT engines.
        """
        return False

    @classmethod
    def get_dict_key_url(cls, file_info: FileInfo) -> Optional[str]:
        """Get dictionary URL by falling back to Paddle/ONNX model config.

        TensorRT doesn't have entries in default_models.yaml, so we
        look up the dictionary URL from Paddle or ONNX configurations.

        Args:
            file_info: Model file information.

        Returns:
            Dictionary URL string or None if not found.
        """
        # Try Paddle first (usually has dict_url)
        for engine_type in [EngineType.PADDLE, EngineType.ONNXRUNTIME]:
            try:
                fallback_info = FileInfo(
                    engine_type=engine_type,
                    ocr_version=file_info.ocr_version,
                    task_type=file_info.task_type,
                    lang_type=file_info.lang_type,
                    model_type=file_info.model_type,
                )
                model_dict = cls.get_model_url(fallback_info)
                if model_dict and "dict_url" in model_dict:
                    return model_dict["dict_url"]
            except Exception as e:
                logger.debug(f"Failed to get dict URL from {engine_type.value}: {e}")

        return None


class TensorRTError(Exception):
    """Exception raised for TensorRT inference errors."""

    pass
