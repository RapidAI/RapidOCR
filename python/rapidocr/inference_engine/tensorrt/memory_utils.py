# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
"""
TensorRT Memory Utilities for RapidOCR.

This module provides memory management utilities for TensorRT inference,
including buffer allocation with optional pinned memory support.

Performance Notes:
- Pre-allocating with MAX shape prevents expensive reallocation during inference
- Pinned memory provides ~2x faster H2D/D2H transfers on discrete GPUs
- On Tegra/Jetson (unified memory), pinned memory benefit is less pronounced
"""
import ctypes
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from cuda.bindings import runtime as cudart
import tensorrt as trt


# =============================================================================
# Constants - Replace magic numbers with named constants
# =============================================================================

# Default maximum shapes for buffer allocation when profile info unavailable
DEFAULT_MAX_INPUT_SHAPE = (1, 3, 2000, 2000)  # Max for detection input
DEFAULT_MAX_OUTPUT_SHAPE = (1, 1000, 1000)  # Generous output buffer

# NumPy dtype to ctypes mapping for pinned memory allocation
_NUMPY_TO_CTYPE = {
    np.float32: ctypes.c_float,
    np.float16: ctypes.c_uint16,  # No native half in ctypes
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.int8: ctypes.c_int8,
    np.uint8: ctypes.c_uint8,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HostDeviceMemory:
    """Host and device memory pair for TensorRT inference.

    This class holds paired CPU (host) and GPU (device) memory buffers
    used for input/output data transfer during TensorRT inference.

    Attributes:
        host: NumPy array for host memory (may be pinned or pageable).
        device: CUDA device pointer as integer.
        is_pinned: Whether host memory is pinned (page-locked).
                   Pinned memory provides faster H2D/D2H transfers.
        host_ptr: Raw pointer for pinned memory (needed for cudaFreeHost).
                  None if using pageable memory.

    Example:
        >>> mem = HostDeviceMemory(host=np.zeros(100), device=ptr, is_pinned=True)
        >>> mem.host[:50] = input_data  # Copy to host buffer
    """

    host: np.ndarray
    device: int
    is_pinned: bool = False
    host_ptr: Optional[int] = None


# =============================================================================
# Main Functions
# =============================================================================


def allocate_buffers(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    use_pinned: bool = True,
) -> Tuple[List[HostDeviceMemory], List[HostDeviceMemory], List[int], int]:
    """Allocate host and device buffers for TensorRT inference.

    This function allocates memory buffers for all input and output tensors
    of a TensorRT engine. It uses the MAXIMUM shape from the optimization
    profile to ensure buffers are large enough for any valid input shape.

    OPTIMIZATION: Pre-allocating with max shape prevents expensive buffer
    reallocation during inference. CUDA malloc/free can cost 1-10ms each.

    Args:
        engine: TensorRT engine containing the optimized network.
        context: TensorRT execution context for inference.
        use_pinned: Whether to use pinned (page-locked) memory for host buffers.
                   Pinned memory provides ~2x faster CPU-GPU transfers.
                   Default is True.

    Returns:
        Tuple containing:
        - inputs: List of HostDeviceMemory for input tensors
        - outputs: List of HostDeviceMemory for output tensors
        - bindings: List of device pointers for TensorRT
        - stream: CUDA stream handle for async operations

    Raises:
        AssertionError: If CUDA operations fail.

    Example:
        >>> inputs, outputs, bindings, stream = allocate_buffers(engine, context)
        >>> inputs[0].host[:data_size] = input_data
    """
    inputs = []
    outputs = []
    bindings = []

    # Create CUDA stream (will be reused for all inference calls)
    status, stream = cudart.cudaStreamCreate()
    assert status.value == 0, f"Failed to create CUDA stream: {status}"

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_mode = engine.get_tensor_mode(tensor_name)

        # Get max shape from optimization profile
        shape = _get_max_shape(engine, tensor_name, tensor_mode)

        # Calculate buffer size
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        # Allocate host memory (pinned or pageable)
        host_mem, host_ptr = _allocate_host_memory(size, dtype, use_pinned)

        # Allocate device memory
        status, device_mem = cudart.cudaMalloc(host_mem.nbytes)
        assert status.value == 0, (
            f"Failed to allocate {host_mem.nbytes} bytes of device memory "
            f"for tensor '{tensor_name}': {status}"
        )

        # Set tensor address in context
        context.set_tensor_address(tensor_name, device_mem)
        bindings.append(device_mem)

        memory = HostDeviceMemory(
            host=host_mem,
            device=device_mem,
            is_pinned=use_pinned and host_ptr is not None,
            host_ptr=host_ptr,
        )

        if tensor_mode == trt.TensorIOMode.INPUT:
            inputs.append(memory)
        else:
            outputs.append(memory)

    return inputs, outputs, bindings, stream


def free_buffers(
    inputs: List[HostDeviceMemory],
    outputs: List[HostDeviceMemory],
    stream: int = None,
) -> None:
    """Free allocated host and device memory buffers.

    This function releases all GPU memory and pinned host memory
    associated with the given buffers.

    Args:
        inputs: List of input HostDeviceMemory to free.
        outputs: List of output HostDeviceMemory to free.
        stream: CUDA stream (optional, currently unused but kept for API).

    Note:
        - Device memory is freed with cudaFree
        - Pinned host memory is freed with cudaFreeHost
        - Pageable host memory (numpy arrays) is garbage collected automatically
    """
    for mem in inputs + outputs:
        # Free device memory
        if mem.device != 0:
            try:
                cudart.cudaFree(mem.device)
            except Exception:
                pass  # Ignore cleanup errors

        # Free pinned host memory if applicable
        if mem.is_pinned and mem.host_ptr is not None:
            try:
                cudart.cudaFreeHost(mem.host_ptr)
            except Exception:
                pass  # Ignore cleanup errors


# =============================================================================
# Private Helper Functions
# =============================================================================


def _get_max_shape(
    engine: trt.ICudaEngine,
    tensor_name: str,
    tensor_mode: trt.TensorIOMode,
) -> Tuple[int, ...]:
    """Get the maximum shape from optimization profile.

    For dynamic shape networks, TensorRT optimization profiles define
    min/opt/max shapes. This function returns the max shape to ensure
    allocated buffers can handle any valid input size.

    Args:
        engine: TensorRT engine.
        tensor_name: Name of the tensor.
        tensor_mode: Whether tensor is input or output.

    Returns:
        Tuple of shape dimensions (max shape from profile or default).
    """
    try:
        profile_shape = engine.get_tensor_profile_shape(tensor_name, 0)
        if profile_shape is not None and len(profile_shape) >= 3:
            return tuple(profile_shape[2])  # Index 2 = max shape
    except Exception:
        pass

    # Fallback to defaults if profile info unavailable
    if tensor_mode == trt.TensorIOMode.INPUT:
        return DEFAULT_MAX_INPUT_SHAPE
    return DEFAULT_MAX_OUTPUT_SHAPE


def _allocate_host_memory(
    size: int,
    dtype: np.dtype,
    use_pinned: bool,
) -> Tuple[np.ndarray, Optional[int]]:
    """Allocate host memory, optionally using pinned (page-locked) memory.

    Pinned memory provides faster CPU-GPU transfers because:
    - No implicit copy to staging buffer required
    - Direct DMA transfers are possible
    - Transfer can be truly asynchronous

    On discrete GPUs (RTX, GTX), pinned memory can be ~2x faster.
    On Tegra/Jetson with unified memory, the benefit is smaller.

    Args:
        size: Number of elements to allocate.
        dtype: NumPy dtype for the array.
        use_pinned: Whether to attempt pinned memory allocation.

    Returns:
        Tuple of:
        - NumPy array (view of pinned memory or regular array)
        - Raw pointer for cleanup (None if pageable memory)
    """
    nbytes = size * np.dtype(dtype).itemsize
    host_ptr = None

    if use_pinned:
        try:
            status, host_ptr = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)
            if status.value == 0 and host_ptr != 0:
                # Create numpy array view of pinned memory
                c_type = _NUMPY_TO_CTYPE.get(np.dtype(dtype).type, ctypes.c_float)
                ptr_type = ctypes.POINTER(c_type)
                c_ptr = ctypes.cast(host_ptr, ptr_type)
                host_mem = np.ctypeslib.as_array(c_ptr, shape=(size,))
                return host_mem, host_ptr
        except Exception:
            pass  # Fall through to pageable allocation

        # Reset pointer if pinned allocation failed
        host_ptr = None

    # Pageable memory fallback
    host_mem = np.empty(size, dtype=dtype)
    return host_mem, None
