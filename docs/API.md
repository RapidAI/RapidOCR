# C, C++, and Python API

The inference core is unchanged: a built-in PP-OCRv6 ONNX interpreter with
CPU and optional Vulkan execution. This document is the public surface other
programs should call.

## Build and install

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix "$PWD/install"
```

That produces:

- `libppocr.a` (`ppocr::static`)
- `libppocr.so` (`ppocr::ppocr`), SONAME `libppocr.so.5`
- headers under `include/ppocr/`
- `lib/cmake/ppocr/ppocrConfig.cmake`

```cmake
find_package(ppocr REQUIRED)
target_link_libraries(app PRIVATE ppocr::ppocr)   # shared
# or ppocr::static
```

Wheels and the sdist are built by `.github/workflows/build-wheels.yml`.
Publishing to PyPI is tag-only; see [`RELEASING.md`](RELEASING.md).

`PPOCR_BUILD_SHARED` and `PPOCR_BUILD_STATIC` both default to `ON`.
`PPOCR_ENABLE_VULKAN` defaults to `ON`. Shaders are compiled at build time
with `glslangValidator` (or the vendored Windows `glslang.exe`) and embedded
in the library. There is no runtime shader compile and no `shaderc`
dependency. The loader is opened with `dlopen` / `LoadLibrary`
(`libvulkan.so.1`, `vulkan-1.dll`), so a Vulkan-enabled build does not add
`libvulkan` to `DT_NEEDED`. Published wheels pass
`-DPPOCR_ENABLE_VULKAN=OFF` so the manylinux image does not need glslang and
the wheel stays a CPU build. A source build that lacks the headers or
glslang skips the shaders and keeps the CPU engine.

Examples:

```bash
./build/ppocr_c_api_demo det.onnx rec.onnx dict.txt image.ppm
./build/ppocr_cpp_api_demo det.onnx rec.onnx dict.txt image.ppm
./build/ppocr_c_api_demo --cpu-info
```

Images passed to the library are packed RGB, row-major, 8-bit. `ppocr_load_ppm`
reads binary P6 PPM. JPEG/PNG decoding stays in the application (or in the
Python package, which uses Pillow).

## C ABI

`include/ppocr/ppocr.h` is a stable C ABI:

- `extern "C"`
- opaque `ppocr_ocr*`
- plain structs for options, images, and text boxes
- integer status codes (`PPOCR_OK`, `PPOCR_ERR_*`)
- `ppocr_create` / `ppocr_destroy`, `ppocr_result_free`, `ppocr_image_free`
- thread-local `ppocr_last_error()`
- `ppocr_options.struct_size` so new fields can be appended

`ppocr_options_init` fills the same defaults as `ppocr::Options`.
`ppocr_recognize` and `ppocr_recognize_batch` on one handle are serialized.
Create one handle per thread when pages should run at the same time.
`ppocr_recognize_batch` preserves input order.

Runtime CPU selection is reported by `ppocr_query_cpu_info`. `active_isa` is
`avx512`, `avx2`, `neon`, or `scalar`. See `docs/PERFORMANCE.md` for the rule
that keeps AVX-512 instructions out of an AVX2-only process.

## C++ API

`include/ppocr/ppocr.hpp` is the C++ API already used by the in-tree tools:

```cpp
ppocr::Options options;
options.backend = ppocr::Backend::cpu_only;
ppocr::OCR ocr(det_path, rec_path, dict_path, options);
for (const ppocr::Result& result : ocr.Recognize(ppocr::LoadPPM(path))) {
  // result.text, result.confidence, result.bbox, result.box
}
```

`QueryCpuInfo()` and `SetDetectionThresholds()` are part of that header.
Exceptions use `std::runtime_error`. The C ABI catches them.

## Python package

This is **rapidocr 5.0.0a1**, the next major version of the PyPI package
`rapidocr`. The import name stays `rapidocr`. The 3.x line (through tag
`v3.9.2`) remains on `main`. There is no 4.x tag in this repository. A
migration note is in `python/README.md`.

The call shape follows 3.x:

```python
from rapidocr import RapidOCR, EngineType, ModelType, OCRVersion

engine = RapidOCR(params={
    "Det.model_path": "PP-OCRv6_det_tiny.onnx",
    "Rec.model_path": "PP-OCRv6_rec_tiny.onnx",
    "Rec.rec_keys_path": "ppocrv6_tiny_dict.txt",
    "Det.model_type": ModelType.TINY,
    "Rec.model_type": ModelType.TINY,
    "Det.ocr_version": OCRVersion.PPOCRV6,
    "Global.text_score": 0.5,
    "Global.use_cls": False,
})
result = engine("page.png", box_thresh=0.5, unclip_ratio=1.6)
result.txts
result.scores
result.boxes          # float32 array, shape (N, 4, 2), TL TR BR BL
result.elapse         # seconds
result.to_json()
result.to_markdown()
result.vis("vis.png")
```

`params` keys use the same `Section.name` form as `main` (`Det.box_thresh`,
`Global.text_score`, `Rec.rec_batch_num`, ...). A YAML `config_path` is
loaded with PyYAML and then overridden by `params`.

File paths, URLs, bytes, and PIL images are decoded as RGB. A numpy ndarray
is treated as BGR, which is what `main` does for `cv2` images, and converted
to RGB before the native engine.

### Binding choice

Python calls the C ABI with ctypes. One shared library is the contract for C,
for other FFI languages, and for Python, so the result struct cannot drift
from a second pybind11 wrapper. scikit-build-core still builds that library
into a wheel:

```bash
pip install ./python
```

### What does not match `main`

| `main` | this package |
| --- | --- |
| ONNX Runtime, OpenVINO, Paddle, PyTorch, TensorRT, MNN | built-in PP-OCRv6 interpreter only (`EngineType.ONNXRUNTIME` or `EngineType.PPOCR_CPP`) |
| PP-OCRv4 and v5 model zoo | PP-OCRv6 ONNX detector + recognizer |
| angle classifier (`use_cls=True` by default) | flag is accepted and ignored, with one warning |
| `use_det=False` / `use_rec=False` | raises `RapidOCRError` |
| word and character boxes | `return_word_box` warns; `word_results` is empty |
| `elapse_list` is `[det, cls, rec]` | `[None, None, total]` because the native call is one timed region |
| OpenCV drawing in `vis` | Pillow polygons |

`result.txts`, `result.scores`, and `result.boxes` stay the fields existing
callers read. `text_score`, `box_thresh`, and `unclip_ratio` still filter or
retune a call without rebuilding the models.

## Vulkan

The C++ default `Backend::hybrid` stays on CPU unless
`PPOCR_ENABLE_HYBRID_GPU_GRAPH` is set. `Backend::gpu_only` runs the Vulkan
graph and throws `std::runtime_error` when the device or a shader cannot
run. It does not silently switch to CPU. The C ABI returns
`PPOCR_ERR_UNSUPPORTED` from `ppocr_create` in that case, with the text in
`ppocr_last_error()`.

```cpp
ppocr::Options options;
options.backend = ppocr::Backend::gpu_only;
options.vulkan_device_index = 0;  // -1: automatic; CPU/lavapipe are skipped
ppocr::OCR ocr(det, rec, dict, options);
```

The same field is `ppocr_options.vulkan_device_index`. Call
`ppocr_request_vulkan_device(index)` before `ppocr_query_backend_info` or
`ppocr_create` when the index is chosen outside the options struct. The
choice is process-wide and is read on the first Vulkan initialization.
`PPOCR_VULKAN_DEVICE_INDEX` is used only when the API index stays `-1`.
`PPOCR_VULKAN_DEVICE_NAME` and `PPOCR_VULKAN_PREFER_DISCRETE` still apply to
automatic selection.

`PPOCR_VULKAN_VALIDATION=1` enables `VK_LAYER_KHRONOS_validation` and counts
ERROR-severity messages (`ppocr_vulkan_validation_error_count`). Devices
whose `maxComputeSharedMemorySize` is below 32 KiB do not create the compute
pipeline; GPU-only then fails and hybrid stays on CPU.

Python follows the 3.x `EngineConfig` style. The default is CPU, including
in the wheels:

```python
engine = RapidOCR(params={
    "EngineConfig.ppocr_cpp.use_vulkan": True,
    "EngineConfig.ppocr_cpp.device_index": 0,
})
```

`backend` accepts `cpu`, `hybrid`, or `vulkan`. `use_vulkan: true` with
`backend: cpu` tries the GPU graph and, if no compute device is available,
logs a warning and stays on CPU. Lavapipe is a CPU Vulkan device, so it is
used only when `device_index` (or `PPOCR_VULKAN_DEVICE_INDEX`) selects it.

A CPU comparison of a full page is deterministic with `PPOCR_BENCH_THREADS=1`.
The default thread pool can change CTC text between runs; that is independent
of the Vulkan graph, which repeats the serial CPU text. Lavapipe timings are
a software rasterizer, not GPU performance. NVIDIA's driver faults if this
process-wide device is destroyed from the static destructor that runs after
the driver has already shut down, so that teardown is skipped at process
exit and the OS reclaims the device.
