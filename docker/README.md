# Docker Development Environments

Pre-configured Docker images for developing and testing RapidOCR with each supported inference engine.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPU images only)

## Available Images

| Image | Engine | Base Image | GPU Required |
|-------|--------|------------|:------------:|
| `onnxruntime-cpu` | ONNX Runtime (CPU) | `python:3.10-slim-bookworm` | No |
| `onnxruntime-gpu` | ONNX Runtime (CUDA) | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | Yes |
| `tensorrt` | NVIDIA TensorRT | `nvcr.io/nvidia/deepstream:7.0-gc-triton-devel` | Yes |
| `paddle` | PaddlePaddle (CPU) | `python:3.10-slim-bookworm` | No |
| `openvino` | Intel OpenVINO | `python:3.10-slim-bookworm` | No |
| `pytorch` | PyTorch (CPU) | `python:3.10-slim-bookworm` | No |
| `mnn` | MNN | `python:3.10-slim-bookworm` | No |

## Quick Start

All commands run from the **repository root**.

### Build an image

```bash
make build-onnxruntime-cpu
```

### Run tests

```bash
make test-onnxruntime-cpu
```

### Open an interactive shell

```bash
make shell-onnxruntime-cpu
```

### Build all images

```bash
make build-all
```

### Clean up

```bash
make clean
```

## Using docker compose directly

```bash
# Build
docker compose -f docker/docker-compose.yaml build onnxruntime-cpu

# Run tests
docker compose -f docker/docker-compose.yaml run --rm onnxruntime-cpu pytest tests/ -v

# Run a single test
docker compose -f docker/docker-compose.yaml run --rm onnxruntime-cpu pytest tests/test_engine.py -k "onnxruntime" -v

# Interactive shell
docker compose -f docker/docker-compose.yaml run --rm onnxruntime-cpu bash
```

## GPU Images

GPU images (`onnxruntime-gpu`, `tensorrt`) require the NVIDIA Container Toolkit and a compatible NVIDIA GPU.

Verify your GPU is accessible:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Then build and use GPU images normally:

```bash
make build-tensorrt
make test-tensorrt
make shell-tensorrt
```

> **Note:** TensorRT builds optimized engine files from ONNX models on first run. This takes several minutes per model. Subsequent runs use cached engines from the persistent model volume.

## How It Works

### Source Code Mounting

Your local `python/` directory is bind-mounted into the container at `/app`. Any code changes you make on the host are immediately reflected inside the container — no rebuild needed.

### Model Caching

A shared Docker volume (`rapidocr-models`) is mounted at `/app/rapidocr/models/`. Models are automatically downloaded on first use and cached in this volume. The cache persists across container rebuilds, so models are only downloaded once.

To clear the model cache:

```bash
docker volume rm rapidocr-models
```

### Architecture

```
docker/
├── Dockerfile.base              # Shared base: Python 3.10, system deps, core pip packages
├── Dockerfile.onnxruntime-cpu   # extends base + onnxruntime
├── Dockerfile.onnxruntime-gpu   # standalone CUDA image + onnxruntime-gpu
├── Dockerfile.tensorrt          # standalone DeepStream image + tensorrt + cuda-python
├── Dockerfile.paddle            # extends base + paddlepaddle
├── Dockerfile.openvino          # extends base + openvino
├── Dockerfile.pytorch           # extends base + torch
├── Dockerfile.mnn               # extends base + MNN
├── docker-compose.yaml          # Service definitions, volumes, GPU reservations
└── .dockerignore                # Excludes .git, models, build artifacts
Makefile                         # Convenience targets (repo root)
```

CPU images extend `Dockerfile.base`. GPU images (`onnxruntime-gpu`, `tensorrt`) use NVIDIA base images and replicate the base setup because they need CUDA pre-installed.

## Examples

### Run OCR from the command line

```bash
make shell-onnxruntime-cpu
# Inside container:
python -c "
from rapidocr import RapidOCR
engine = RapidOCR()
result = engine('tests/test_files/ch_en_num.jpg')
print(result)
"
```

### Run with a specific engine

```bash
make shell-pytorch
# Inside container:
python -c "
from rapidocr import RapidOCR, EngineType
engine = RapidOCR(params={
    'Det.engine_type': EngineType.TORCH,
    'Cls.engine_type': EngineType.TORCH,
    'Rec.engine_type': EngineType.TORCH,
})
result = engine('tests/test_files/ch_en_num.jpg')
print(result)
"
```

### Run a specific test file

```bash
docker compose -f docker/docker-compose.yaml run --rm onnxruntime-cpu \
    pytest tests/test_engine.py::test_ppocrv5_rec_mobile -v
```

## Troubleshooting

### `docker compose` command not found

Ensure you have Docker Compose v2 installed. On older systems, the command may be `docker-compose` (with hyphen) instead of `docker compose` (with space).

### GPU not detected inside container

1. Verify the NVIDIA driver: `nvidia-smi`
2. Verify the container toolkit: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
3. If using Docker Desktop, enable GPU support in Settings > Resources > GPU.

### TensorRT `CUDA initialization failure with error: 35`

This means the TensorRT or CUDA Python version doesn't match your host driver. The Dockerfile pins `tensorrt>=8.6,<8.7` and `cuda-python>=12.0,<13.0` for DeepStream 7.0. If your driver is older, you may need to adjust these versions.

### Models re-download every time

Ensure the `rapidocr-models` volume is mounted. Check with `docker volume ls | grep rapidocr`.
