<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Logov2_black.png"  width="60%" height="60%">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Logov2_white.png"  width="60%" height="60%">
    <img alt="RapidOCR" src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Logov2_white.png">
  </picture>

<div>&nbsp;</div>
<div align="center">
    <b><font size="4"><i>信创级开源OCR - 为世界内容安全贡献力量</i></font></b>
</div>
<div>&nbsp;</div>

<a href="https://clawhub.ai/rapidai/rapidocr" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A6%9E%20ClawHub-RapidOCR%20Skill-blue?link=https%3A%2F%2Fclawhub.ai%2Frapidai%2Frapidocr"></a>
<a href="https://huggingface.co/spaces/RapidAI/RapidOCRv3" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
<a href="https://www.modelscope.cn/studios/RapidAI/RapidOCRv3.0.0/summary" target="_blank"><img src="https://img.shields.io/badge/魔搭-Demo-blue"></a>
<a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/assets/RapidOCRDemo.ipynb" target="_blank"><img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.9-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://github.com/RapidAI/RapidOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/RapidAI/RapidOCR?color=9ea"></a>
<a href="https://pepy.tech/project/rapidocr"><img src="https://static.pepy.tech/personalized-badge/rapidocr?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20rapidocr"></a>
<a href="https://pypi.org/project/rapidocr/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr"></a>
<a href="https://github.com/RapidAI/RapidOCR/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOCR?color=ccf"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

加入我们的 [Discord](https://discord.gg/33eyQJq498)

简体中文 | [English](./README.md)
</div>

本分支（`V5_NG`）是 **rapidocr 5.0.0a1**：C++ 实现的 PP-OCRv6 引擎，带 C ABI、C++ API，以及导入名仍为 `rapidocr` 的 Python 包。3.x 线（最新标签 `v3.9.2`）留在 [`main`](https://github.com/RapidAI/RapidOCR/tree/main)。上方徽章里的在线演示跑的是 3.x。

### 📝 简介

RapidOCR 是面向小模型、离线部署的开源 OCR。5.x 保留 3.x 应用已经在用的 `RapidOCR()` 调用，推理改由本仓库里的 PP-OCRv6 引擎完成，不再经过 ONNX Runtime。

解释器实现官方 PP-OCRv6 tiny / small / medium ONNX 检测器和识别器图里用到的算子：卷积、GEMM、池化、归一化、resize、形状算子、attention matmul，以及 DB / CTC 后处理。它是 PP-OCRv6 运行时，通用 ONNX 模型仍需要别的引擎。

如果本项目对你的工作有帮助，欢迎点一颗 ⭐ Star。

### ✨ 功能

- C++ PP-OCRv6 检测与识别。库不链接 ONNX Runtime、Paddle Inference 或 OpenCV。
- 同一个二进制在运行时分派到 AVX-512、AVX2+FMA 或 AArch64 NEON。`PPOCR_FORCE_ISA=scalar|avx2|avx512|neon` 只选择当前 CPU 合法的路径。只有 AVX2 的机器不会执行 AVX-512 目标文件。
- 可选 Vulkan 计算。源码构建在存在 `third_party/Vulkan-Headers` 和 `glslangValidator` 时嵌入 SPIR-V（`PPOCR_ENABLE_VULKAN` 默认 `ON`）。加载器通过 `dlopen` / `LoadLibrary` 打开 `libvulkan.so.1` 或 `vulkan-1.dll`，因此 Vulkan 不是 `DT_NEEDED` 依赖。发布的 wheel 传入 `-DPPOCR_ENABLE_VULKAN=OFF`。
- C ABI（`include/ppocr/ppocr.h`）、C++ API（`include/ppocr/ppocr.hpp`）和 Python 包 `rapidocr`。Python 用 ctypes 调用 C ABI。
- Python 调用形式跟随 3.x：`RapidOCR(params={"Section.key": value})`，然后使用 `result.txts`、`result.scores`、`result.boxes`、`result.elapse`、`to_json()`、`to_markdown()` 和 `vis()`。
- 缺少的模型会按 `default_models.yaml` 里的 ModelScope 地址下载（与 3.x 的 `v3.9.2` 文件名相同），保存到 `Global.model_root_dir`，并校验 SHA256。

### 🎥 效果展示

<div align="center">
    <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/demo.gif" alt="Demo" width="100%" height="100%">
</div>

动画是 RapidOCR 3.x 的网页演示。本分支核对过的单行样例是 [`python/tests/data/hello.png`](python/tests/data/hello.png)（960×240）上的 `Hello RapidOCR 123`。

### ⚡ 性能

4 线程的 x86 行和 16 线程的 NEON 行是在当前代码上用 `ppocr_determinism`（`Backend::cpu_only`）重测的，每个数字是该工具预热之后 4 次的均值。1 线程行仍是 `b71f5b7` 的均值。模型为 PP-OCRv6 **tiny** ONNX。

两张图的 PNG 与 PPM 像素相同：

| 样例 | 尺寸 | 内容 |
| --- | --- | --- |
| hello | 960×240 | 一行：`Hello RapidOCR 123`（`python/tests/data/hello.png`） |
| page | 1700×2200 | 45 行。该页是本地测速图，没有提交进本仓库。 |

**x86。** 4 vCPU Intel Xeon（family 6，model 207），具备 AVX2 与 AVX-512F/DQ/BW/VL。`g++` 13.3。默认线程池为 4（`min(16, hardware_concurrency)`）。AVX2 行是在同一颗 CPU 上设置 `PPOCR_FORCE_ISA=avx2`。hello 为 7 次均值，page 为 5 次均值；1 线程的 page 行为 3 次均值。

**ARM。** 20 核：10× Cortex-X925（最高 3.9 GHz）与 10× Cortex-A725（最高 2.8 GHz），`g++` 13.3，NEON。默认线程池为 16。板上的二进制包含该提交里的原地池化修复。hello 为 7 次均值，page 为 5 次均值，1 线程 hello 为 3 次均值。

C++ 选项默认值：长边限制 960，检测器使用 ImageNet mean/std，不做膨胀。

| ISA | 线程 | hello | page |
| --- | --- | --- | --- |
| AVX-512 | 4 | 22.9 ms | 222.8 ms |
| AVX2 | 4 | 23.5 ms | 378.4 ms |
| AVX-512 | 1 | 35.14 ms | 387.6 ms |
| NEON | 16 | 33.7 ms | 231.4 ms |
| NEON | 1 | 68.52 ms | — |

这些运行的文本、置信度和框与同一二进制的单线程结果按位一致（确定性检查做的是逐位比较）。更早的算子记录（含 GEMM 形状）在 [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)，早于本表。

#### 与 rapidocr 3.9.2 的对比

机器与上面的 x86 行相同，tiny ONNX 文件相同，Python 3.12。`rapidocr==3.9.2` 使用 ONNX Runtime 1.30.0。两边都设置 `use_cls=False`，因此 3.9.2 的方向分类器没有参与计时。3.9.2 发布包自己的默认是 PP-OCRv6 **small** 且分类器开启；那种配置这次没有计时。

下表墙钟是一次预热之后 8 次调用的中位数，括号里是最小–最大。两个包在上面那台 4 线程 Xeon 上紧挨着测。AVX2 只加在 5.0.0a1 进程的 `PPOCR_FORCE_ISA=avx2` 上；这台机器上的 ONNX Runtime 1.30.0 仍走 AVX-512。

**两边 Python 的共同默认**（5.0.0a1 与 3.9.2 的 `config.yaml`：短边限制 736，mean/std 0.5，开启膨胀）。960×240 的短图会被放大到短边 736，所以这组时间高于上面的 C++ 默认值。

| 包 | hello | page |
| --- | --- | --- |
| rapidocr 5.0.0a1，AVX-512 | 121.0 ms（103.5–156.3） | 410.6 ms（391.7–441.2） |
| rapidocr 5.0.0a1，AVX2 | 161.7 ms（152.6–164.4） | 587.4 ms（573.7–611.5） |
| rapidocr 3.9.2（ORT AVX-512） | 205.8 ms（174.8–257.0） | 514.4 ms（434.1–551.6） |

相对 3.9.2 的中位数，AVX-512 的 hello 快 41%，整页快 20%（均值分别是 122.1 ms、412.1 ms，对比 211.1 ms、509.0 ms；整页均值快 19%，因为 3.9.2 自身的极差很大）。强制 AVX2 的 hello 快 21%。强制 AVX2 的整页比这台机器上的 AVX-512 ORT 慢 14%：识别器里的 expand-GELU GEMM 已经接近 AVX2 与 AVX-512 的位宽比（约 1.8 倍），而 ORT 没有被关掉 AVX-512。三行的整页文本都是同样的 45 句。Hello 都是 `Hello RapidOCR 123`。相对改内核之前的 5.0.0a1，两张图的框 IoU 都是 1.0。3.9.2 在这页上的框和 5.0.0a1 本来就不完全重合（最小 IoU 约 0.85），这是这批内核之前就有的检测差异。

**套用 C++ 检测默认**（长边限制 960，ImageNet mean/std，关闭膨胀，阈值 0.20 / 0.45，unclip 1.40）。5.x 的 C++ 行就是上一张表。Python 与 3.9.2 行是墙钟；`elapse` 是包自己的计时。

| 运行方式 | hello | page |
| --- | --- | --- |
| 5.0.0a1 C++ | 22.9 ms | 222.8 ms |
| 5.0.0a1 Python | 20.07 ms（18.61–21.78） | 339.1 ms（332.9–348.5） |
| 3.9.2 Python | 28.73 ms（引擎 `elapse` 20.2 ms） | 548.2 ms（引擎 `elapse` 521.3 ms） |

Hello 文本两边仍是 `Hello RapidOCR 123`。Python 里的 `RapidOCR()` 保持上面的 3.x 预处理，所以直接替换调用对应的是第一张对比表。需要 C++ 默认值时，请设置 `Det.limit_type`、`Det.limit_side_len`、`Det.mean` 和 `Det.std`。

更早在具体显卡上记过的 Vulkan 整页时间没有写进本表。本机上的 lavapipe 是软件光栅器，它的毫秒数不是 GPU 性能。`Backend::gpu_only` 不会在神经网络失败时改走 CPU。

### 🛠️ 安装

`5.0.0a1` 是预发布版。2026-10-05 这一天 PyPI 上最新的 `rapidocr` 仍是 3.9.2，因此在推送 `v5*` 标签之前（见[发布](#-发布)），下面两条命令安装的都是 **3.9.2**：

```bash
pip install rapidocr
pip install --pre rapidocr
```

标签发布之后，`pip install rapidocr` 仍会解析到最新的正式版，预发布版用：

```bash
pip install --pre rapidocr
```

在本分支的检出目录安装（需要 CMake ≥ 3.20、C++20 编译器，以及 `pyproject.toml` 里的 Python 依赖）：

```bash
pip install .
```

该构建跟随 `pyproject.toml`：共享库 `libppocr`，关闭示例和测试，关闭 Vulkan。本地 wheel 是先把同一次构建打成包：

```bash
pip wheel . -w dist
pip install dist/rapidocr-*.whl
```

在检出目录里 `pip install ./python` 也可以，它让 CMake 指向父目录，但打不出自包含的 sdist。仓库根目录的 `pip install .` 可以。

#### CMake

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix "$PWD/install"
```

`PPOCR_BUILD_SHARED` 和 `PPOCR_BUILD_STATIC` 默认都是 `ON`。只要 CPU 库时关闭 Vulkan：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPPOCR_ENABLE_VULKAN=OFF
```

源码包打开 Vulkan（需要头文件和 `glslangValidator`）：

```bash
CMAKE_ARGS="-DPPOCR_ENABLE_VULKAN=ON -DPPOCR_BUILD_STATIC=OFF" pip install .
```

#### CMake 集成

```cmake
cmake_minimum_required(VERSION 3.20)
project(ocr_app C)
find_package(ppocr REQUIRED)
add_executable(ocr_app main.c)
target_link_libraries(ocr_app PRIVATE ppocr::ppocr)
```

安装前缀里有 `libppocr.so`（SONAME `libppocr.so.5`）、`include/ppocr/` 下的头文件，以及 `lib/cmake/ppocr/ppocrConfig.cmake`。静态库目标是 `ppocr::static`。运行时动态加载器必须能找到 `libppocr.so.5`（`LD_LIBRARY_PATH` 或 rpath）。

### 📋 使用

#### Python

下面的路径是 PP-OCRv6 tiny 文件。省略路径时，`RapidOCR()` 会下载配置里的默认模型（PP-OCRv6 small；把 `Det.model_type` / `Rec.model_type` 设为 `tiny` 则下载 tiny）。

```python
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Det.model_path": "PP-OCRv6_det_tiny.onnx",
    "Rec.model_path": "PP-OCRv6_rec_tiny.onnx",
    "Rec.rec_keys_path": "ppocrv6_tiny_dict.txt",
    "Global.use_cls": False,
})
result = engine("hello.png")
print(result.txts)
print(result.scores)
print(result.to_json())
result.vis("vis_result.jpg")
```

`result.boxes` 是形状 `(N, 4, 2)` 的 float32 数组，顺序为左上、右上、右下、左下。文件路径、URL、字节和 PIL 图像按 RGB 读取。numpy 数组按 BGR 处理，与 3.x 的 OpenCV 图像一致，送进原生引擎前会转成 RGB。

自动下载 tiny 一对模型（`Global.model_root_dir` 为空时，文件落在包内 `models/` 目录）：

```python
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Det.model_type": "tiny",
    "Rec.model_type": "tiny",
    "Global.use_cls": False,
})
result = engine("hello.png")
print(result.txts)
```

#### 命令行

```bash
python -m rapidocr hello.png \
  --det PP-OCRv6_det_tiny.onnx \
  --rec PP-OCRv6_rec_tiny.onnx \
  --dict ppocrv6_tiny_dict.txt \
  --model-type tiny
```

`--model-type` 取 `tiny`、`small` 或 `medium`。控制台脚本 `rapidocr` 调用同一个 `main`。

#### C++

```cpp
#include "ppocr/ppocr.hpp"

#include <iostream>

int main(int argc, char** argv) {
  ppocr::Options options;
  options.backend = ppocr::Backend::cpu_only;
  ppocr::OCR ocr(argv[1], argv[2], argv[3], options);
  for (const ppocr::Result& result : ocr.Recognize(ppocr::LoadPPM(argv[4]))) {
    std::cout << result.text << '\t' << result.confidence << '\n';
  }
}
```

```bash
./ppocr_cpp_api_demo det.onnx rec.onnx dict.txt image.ppm
```

交给库的图像是按行排列的 8-bit RGB。`LoadPPM` 读取二进制 P6。JPEG / PNG 解码留在应用程序里（Python 侧由 Pillow 完成）。

#### C

```c
#include "ppocr/ppocr.h"

#include <stdio.h>

int main(int argc, char** argv) {
  ppocr_options options;
  ppocr_options_init(&options);
  options.backend = PPOCR_BACKEND_CPU;

  ppocr_ocr* ocr = ppocr_create(argv[1], argv[2], argv[3], &options);
  ppocr_image_buffer loaded = {0};
  if (ppocr_load_ppm(argv[4], &loaded) != PPOCR_OK) return 1;

  ppocr_image image = {loaded.width, loaded.height, loaded.rgb};
  ppocr_result result = {0};
  if (ppocr_recognize(ocr, &image, &result) != PPOCR_OK) {
    fprintf(stderr, "%s\n", ppocr_last_error());
    return 1;
  }
  for (size_t i = 0; i < result.count; ++i) {
    printf("%s\t%.4f\n", result.items[i].text, result.items[i].confidence);
  }
  ppocr_result_free(&result);
  ppocr_image_free(&loaded);
  ppocr_destroy(ocr);
  return 0;
}
```

同一个句柄上的 `ppocr_recognize` 是串行的。多页同时跑时，每个线程各自创建一个句柄。`ppocr_options_init` 会填好 `ppocr_options.struct_size`，以便以后在结构体末尾追加字段。

#### Vulkan

wheel 只有 CPU。启用了 Vulkan 的构建里，设备选择写在 3.x 放 `use_cuda` 的同一处：

```python
engine = RapidOCR(params={
    "Det.model_path": "PP-OCRv6_det_tiny.onnx",
    "Rec.model_path": "PP-OCRv6_rec_tiny.onnx",
    "Rec.rec_keys_path": "ppocrv6_tiny_dict.txt",
    "Global.use_cls": False,
    "EngineConfig.ppocr_cpp.use_vulkan": True,
    "EngineConfig.ppocr_cpp.device_index": 0,
})
result = engine("hello.png")
print(result.txts)
```

`device_index` 为 `-1` 时会跳过 lavapipe 这类 CPU Vulkan 设备。要用 lavapipe 时设置索引（或 `PPOCR_VULKAN_DEVICE_INDEX`），并把 `VK_ICD_FILENAMES` 指到它的 ICD。没有可用计算设备时，Python 打一条警告并留在 CPU。`backend` 接受 `cpu`、`hybrid` 或 `vulkan`。

C++ 默认的 `Backend::hybrid` 留在 CPU，除非设置了 `PPOCR_ENABLE_HYBRID_GPU_GRAPH`。`Backend::gpu_only` 在设备或着色器无法运行时抛出 `std::runtime_error`。C ABI 在这种情况下从 `ppocr_create` 返回 `PPOCR_ERR_UNSUPPORTED`。

```cpp
ppocr::Options options;
options.backend = ppocr::Backend::gpu_only;
options.vulkan_device_index = 0;  // -1：自动；跳过 CPU/lavapipe
ppocr::OCR ocr(det, rec, dict, options);
```

在部分显卡上，一整条录制好的命令缓冲会丢掉 CTC 时间步。默认 GPU 路径按带栅栏的分段提交。某块显卡验证通过之后，可以用 `PPOCR_ENABLE_GPU_GRAPH_REPLAY=1` 改回单条命令缓冲。

### ⚙️ 配置

Python 键使用 3.x 的 `Section.name` 形式。YAML `config_path` 由 PyYAML 载入，再被 `params` 覆盖。

| 键 | 作用 |
| --- | --- |
| `Det.model_path`、`Rec.model_path`、`Rec.rec_keys_path` | 本地 ONNX 与识别字典。省略的路径会下载。 |
| `Det.model_type`、`Rec.model_type` | PP-OCRv6 URL 表中的 `tiny`、`small` 或 `medium`。 |
| `Det.ocr_version`、`Rec.ocr_version` | `PP-OCRv6`。更老的版本名会抛 `NotImplementedError`。 |
| `Det.limit_side_len`、`Det.limit_type` | 缩放限制。`min` 与 3.x 相同（短边）。`max` 是 C++ 的长边默认。 |
| `Det.thresh`、`Det.box_thresh`、`Det.unclip_ratio` | 检测掩码阈值、框分数和 unclip。 |
| `Det.mean`、`Det.std` | 检测器归一化，BGR 顺序，长度 3。 |
| `Det.use_dilation` | 检测掩码的 2×2 膨胀。Python 配置里开启，`ppocr::Options` 里关闭。 |
| `Global.text_score` | 丢掉低于该分数的行。也可以作为单次调用参数。 |
| `Global.use_cls` | 接受该键。原生引擎不做方向分类，并警告一次。 |
| `Global.model_root_dir` | 下载目录。为空时用包内 `models/`。 |
| `Rec.rec_batch_num` | 识别批大小。 |
| `EngineConfig.ppocr_cpp.use_vulkan` | 请求 Vulkan 计算图。 |
| `EngineConfig.ppocr_cpp.device_index` | Vulkan 设备。`-1` 跳过 lavapipe。 |
| `EngineConfig.ppocr_cpp.backend` | `cpu`、`hybrid` 或 `vulkan`。 |

C++ 默认值在 `include/ppocr/ppocr.hpp` 的 `ppocr::Options` 上（`det_limit_side_len` 960，`det_limit_type` 0，ImageNet `det_mean` / `det_std`，`rec_batch_size` 4）。`PPOCR_BENCH_THREADS` 覆盖常驻 SIMD 线程池（默认 `min(16, hardware_concurrency)`）。

### 🔀 从 rapidocr 3.x 迁移

`RapidOCR(config_path=..., params={"Section.key": value})` 以及

`engine(img, use_det=..., use_cls=..., use_rec=..., text_score=..., box_thresh=..., unclip_ratio=...)`

名字不变。`result.txts`、`result.scores`、`result.boxes`、`result.elapse`、`to_json()`、`to_markdown()` 和 `vis()` 仍在。

| 3.x（`main`，至 `v3.9.2`） | rapidocr 5 |
| --- | --- |
| ONNX Runtime、OpenVINO、Paddle、PyTorch、TensorRT、MNN | 内置 PP-OCRv6 解释器。`EngineType.ONNXRUNTIME` 和 `EngineType.PPOCR_CPP` 都选它。其他引擎名抛 `NotImplementedError`。 |
| PP-OCRv4 与 PP-OCRv5 模型库 | PP-OCRv6 ONNX 检测器和识别器。 |
| 方向分类器（默认 `use_cls=True`） | `use_cls` 为真时仍会下载 cls 文件。原生引擎忽略它，并打一条警告。 |
| `use_det=False` 或 `use_rec=False` | 抛 `RapidOCRError`。检测和识别始终一起跑。 |
| 单词框和单字框 | `return_word_box` / `return_single_char_box` 会警告。`word_results` 为空。 |
| `elapse_list` 为 `[det, cls, rec]` | `[None, None, total]`。空白页是 `RapidOCROutput()`（`boxes` / `txts` / `img` 为 `None`）。 |
| `vis()` 用 OpenCV 按 BGR 绘制 | `vis()` 用 Pillow 绘制并返回 RGB。 |
| `EngineConfig` 的 CUDA、OpenVINO、TensorRT、MNN 等 | 这些键可以载入和读取，不会生效。上面的 Vulkan 键会生效。 |

只读取 PP-OCRv6 模型的 `txts`、`scores` 和 `boxes` 的调用方，装上这个包即可切换。把 `use_cls=False` 设上可以去掉警告。不要传 `use_det=False` 或更老的 OCR 版本。

本仓库没有 4.x 标签。细节和 C ABI 见 [`docs/API.md`](docs/API.md) 与 [`python/README.md`](python/README.md)。

### ⚠️ 已知限制

- 没有方向分类器。`use_cls` 不会旋转文本框。
- 没有只检测或只识别的调用。`use_det=False` 和 `use_rec=False` 会抛 `RapidOCRError`。
- 没有单词框或单字框。`word_results` 保持为空。
- 原生引擎跑 PP-OCRv6 ONNX 图。PP-OCRv4 / v5 文件不会被执行。
- 纯 GPU 模式没有 CPU 神经网络回退。除非设置 `PPOCR_ENABLE_HYBRID_GPU_GRAPH`，hybrid 留在 CPU。
- 发布的 wheel 不含 Vulkan 着色器。

### 📚 文档

- [C / C++ / Python API](docs/API.md)
- [CPU 性能笔记](docs/PERFORMANCE.md)（算子沿革；上面的表是其后的重跑）
- [Python 包说明](python/README.md)
- [3.x 文档站](https://rapidai.github.io/RapidOCRDocs/)（中文，`main`）

### 📦 发布

wheel 和 sdist 由 [`.github/workflows/build-wheels.yml`](.github/workflows/build-wheels.yml) 在推送到 `V5_NG` 时构建。只有推送 `v5*` 标签时才会发布到 PyPI 并创建 GitHub Release。步骤、可信发布者设置和打标签命令见 [`docs/RELEASING.md`](docs/RELEASING.md)。

### 👥 谁在使用？([更多](https://github.com/RapidAI/RapidOCR/network/dependents))

依赖 `rapidocr` 包的项目包括 [Docling](https://github.com/DS4SD/docling)、[CnOCR](https://github.com/breezedeus/CnOCR)、[Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) 和 [LangChain](https://github.com/langchain-ai/langchain)。这些发布物今天安装的是 `main` 上的 3.x。新的使用项目可以在[讨论 286](https://github.com/RapidAI/RapidOCR/discussions/286) 登记。

### 🙏 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 提供了本引擎所实现的模型与 OCR 流程。
- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch) 与 [PaddleX](https://github.com/PaddlePaddle/PaddleX) 为整个 RapidOCR 项目提供了转换模型和文档模型。
- [贡献者](https://github.com/RapidAI/RapidOCR/graphs/contributors) 图上的每一位。

<p align="left">
  <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=RapidAI/RapidOCR&max=400&columns=10" width="60%"/>
  </a>
</p>

### 🌟 赞助商与支持者

RapidOCR 使用 Apache-2.0 许可。赞助说明在 [3.x 文档站](https://rapidai.github.io/RapidOCRDocs/main/sponsor/)。

| 赞助商 | 应用 | 简介 |
| :---: | :---: | --- |
| <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Quicker.jpg" width="65" height="65" alt="Quicker"/> | [Quicker](https://getquicker.net/) | 您的指尖工作箱 |

### 📜 引用

```bibtex
@misc{RapidOCR 2021,
    title={{Rapid OCR}: OCR Toolbox},
    author={RapidAI Team},
    howpublished = {\url{https://github.com/RapidAI/RapidOCR}},
    year={2021}
}
```

### ⭐️ Star 历史

![Star History](https://raw.githubusercontent.com/RapidAI/RapidOCR/star-tracker-data/charts/star-history.svg)

### ⚖️ 许可证

RapidOCR 源代码版权归 RapidOCR 作者所有，采用 Apache License 2.0（[`pyproject.toml`](pyproject.toml) 与 [`python/pyproject.toml`](python/pyproject.toml) 中的 `license = "Apache-2.0"`）。许可文本见 [apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)。

OCR 模型文件衍生自官方 PaddleOCR 模型。上游权重的版权归百度及/或相应的 PaddleOCR 权利人所有。这些上游模型以 Apache License 2.0 分发。RapidOCR 不主张对上游权重的所有权。
