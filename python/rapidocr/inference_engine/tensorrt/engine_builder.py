# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Any, Dict

import tensorrt as trt

from ...utils.log import logger


class TRTEngineBuilder:
    """Build TensorRT engine from ONNX model"""

    def __init__(
        self,
        onnx_path: Path,
        engine_path: Path,
        cfg: Dict[str, Any],
        task_type: str,
        trt_logger: trt.Logger,
    ):
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.cfg = cfg
        self.task_type = task_type
        self.trt_logger = trt_logger

    def build(self) -> trt.ICudaEngine:
        """Build TensorRT engine from ONNX"""
        builder = trt.Builder(self.trt_logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.trt_logger)

        # Parse ONNX model
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()

        # Set workspace size
        workspace_size = self.cfg.get("workspace_size", 1 << 30)  # 1GB default
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # Set precision
        if self.cfg.get("use_fp16", True) and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Using FP16 precision")
        else:
            logger.info("Using FP32 precision")

        if self.cfg.get("use_int8", False) and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("Using INT8 precision")

        # Add optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        self._set_dynamic_shapes(network, profile)
        config.add_optimization_profile(profile)

        # Build engine
        logger.info("Building TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine to cache
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.engine_path, "wb") as f:
            f.write(serialized_engine)
        logger.info(f"TensorRT engine saved to {self.engine_path}")

        # Deserialize and return
        runtime = trt.Runtime(self.trt_logger)
        return runtime.deserialize_cuda_engine(serialized_engine)

    def _set_dynamic_shapes(
        self, network: trt.INetworkDefinition, profile: trt.IOptimizationProfile
    ):
        """Set dynamic shape optimization profiles"""
        profile_key = f"{self.task_type}_profile"
        profile_cfg = self.cfg.get(profile_key, {})

        # Default profiles based on task type
        if self.task_type == "det":
            min_shape = profile_cfg.get("min_shape", (1, 3, 32, 32))
            opt_shape = profile_cfg.get("opt_shape", (1, 3, 736, 736))
            max_shape = profile_cfg.get("max_shape", (1, 3, 2000, 2000))
        elif self.task_type == "rec":
            min_shape = profile_cfg.get("min_shape", (1, 3, 48, 32))
            opt_shape = profile_cfg.get("opt_shape", (6, 3, 48, 320))
            max_shape = profile_cfg.get("max_shape", (6, 3, 48, 2000))
        elif self.task_type == "cls":
            min_shape = profile_cfg.get("min_shape", (1, 3, 48, 32))
            opt_shape = profile_cfg.get("opt_shape", (6, 3, 48, 192))
            max_shape = profile_cfg.get("max_shape", (6, 3, 48, 192))
        else:
            # Generic fallback
            min_shape = (1, 3, 32, 32)
            opt_shape = (1, 3, 224, 224)
            max_shape = (1, 3, 2000, 2000)

        # Set shapes for input tensor
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        profile.set_shape(
            input_name,
            min=tuple(min_shape),
            opt=tuple(opt_shape),
            max=tuple(max_shape),
        )
        logger.info(
            f"Set optimization profile for {input_name}: "
            f"min={min_shape}, opt={opt_shape}, max={max_shape}"
        )
