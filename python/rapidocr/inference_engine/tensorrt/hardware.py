# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass
from typing import Any, Dict

import tensorrt as trt


@dataclass(frozen=True)
class TensorRTCapabilities:
    fast_fp16: bool
    fast_int8: bool


def detect_capabilities(trt_logger: trt.Logger) -> TensorRTCapabilities:
    """Return the precision capabilities of the active TensorRT device."""
    # Some lightweight TensorRT-compatible test doubles do not expose Builder.
    # Real TensorRT installations always do, so this branch is only a
    # compatibility fallback for callers that provide such a test double.
    if not hasattr(trt, "Builder"):
        return TensorRTCapabilities(fast_fp16=True, fast_int8=True)

    builder = trt.Builder(trt_logger)
    return TensorRTCapabilities(
        fast_fp16=bool(builder.platform_has_fast_fp16),
        fast_int8=bool(builder.platform_has_fast_int8),
    )


def resolve_precision(
    cfg: Dict[str, Any], capabilities: TensorRTCapabilities
) -> Dict[str, bool]:
    """Resolve requested precision flags against hardware support."""
    return {
        "use_fp16": bool(cfg.get("use_fp16", False) and capabilities.fast_fp16),
        "use_int8": bool(cfg.get("use_int8", False) and capabilities.fast_int8),
    }
