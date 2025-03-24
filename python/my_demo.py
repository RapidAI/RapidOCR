# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import time

from rapidocr import RapidOCR
import cv2
import os
import psutil


def get_current_process_memory_usage():
    """
    Gets the memory usage of the current process.

    Returns:
        A tuple containing:
            - resident_memory (int): Resident Set Size (RSS) in MB.  This is the non-swapped physical memory a process has used.
            - virtual_memory (int): Virtual Memory Size (VMS) in MB. This includes all memory the process can access, including swapped out memory.
            - shared_memory (int/None): Shared memory (SHM) in MB, or None if not available.  This is the memory shared with other processes.  This might not be available on all systems (especially Windows, where psutil may return 0.0).
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    resident_memory_mb = mem_info.rss / (1024 * 1024)  # Convert bytes to MB
    virtual_memory_mb = mem_info.vms / (1024 * 1024)

    try:
        shared_memory_mb = mem_info.shared / (1024 * 1024)
    except AttributeError:
        shared_memory_mb = None  # Shared memory might not be a valid metric on some systems

    return resident_memory_mb, virtual_memory_mb, shared_memory_mb


def get_current_process_memory_usage_full():
    """Gets more detailed memory information, including platform-specific details."""
    process = psutil.Process(os.getpid())
    mem_full_info = process.memory_full_info()

    result = {}
    for name in mem_full_info._fields:  # Iterate through all fields in the namedtuple
        value_mb = getattr(mem_full_info, name) / (1024 * 1024)  # Get attribute value and convert to MB
        result[name] = value_mb

    return result


def print_memory_usage():
    rss, vms, _ = get_current_process_memory_usage()  # We don't care about shm here
    print(f" RSS = {rss:.2f} MB, VMS = {vms:.2f} MB")

engine = RapidOCR(params={"Global.use_cls": False, "Global.max_side_len":4000, "Global.min_side_len":0,
"Global.width_height_ratio": -1,
# "EngineConfig.onnxruntime.intra_op_num_threads": 8,
# "EngineConfig.onnxruntime.inter_op_num_threads": 8,
# "Global.with_openvino": True,
# "Global.with_paddle": True,
                          "EngineConfig.onnxruntime.use_dml": True
                          })

img_url = cv2.imread('tests/test_files/negative_text.png')

for i in range(20):
    start = time.time()
    result = engine(img_url)
    print(time.time()-start)
    print_memory_usage()
# print(result)

# result.vis()

