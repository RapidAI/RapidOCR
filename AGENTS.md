# AGENTS.md

Notes for AI coding agents (and humans) working with RapidOCR's Python package, especially around GPU/inference-engine behavior that is easy to get wrong. These are documentation notes only — no code behavior described here has changed.

## Default backend runs on CPU, silently

The default engine config has `EngineConfig.onnxruntime.use_cuda=false`. If you install `pip install rapidocr onnxruntime` (the README quickstart command) and construct `RapidOCR()` on a machine with an NVIDIA GPU, it will run entirely on CPU with **no warning or error** — all of det/cls/rec sessions report `CPUExecutionProvider`.

If you expect GPU execution, verify it explicitly after building the engine:

```python
print(engine.text_det.session.session.get_providers())
```

Expect `CUDAExecutionProvider` first in the list. If you only see `CPUExecutionProvider` / `AzureExecutionProvider`, the GPU is not being used.

## Getting the default (onnxruntime) backend onto GPU requires three things together

1. `onnxruntime-gpu` installed instead of plain `onnxruntime` (version must match your CUDA/cuDNN).
2. `EngineConfig.onnxruntime.use_cuda=True` passed via `RapidOCR(params=...)`.
3. A matching CUDA/cuDNN runtime available at runtime (system CUDA, or the pip `nvidia-*-cu12` packages plus `LD_LIBRARY_PATH`).

All three are required — any one missing falls back to CPU. If `use_cuda=True` is set but `onnxruntime-gpu` isn't installed, RapidOCR does print an installation hint log (not silent in that specific case); but the CPU-by-default case above produces no log at all.

## T4/Turing measurements (default onnxruntime backend)

Measured on a real Tesla T4 (16GB, Turing, sm_75), `ch_en_num.jpg` (430x323), det+cls+rec pipeline via the public `RapidOCR()` API:

| | providers | mean latency | peak VRAM |
|---|---|---|---|
| CPU | CPUExecutionProvider x3 | 503.8 ms | 0 |
| CUDA (T4) | CUDAExecutionProvider x3 | 150.6 ms | ~0.55 GB |

That's a **3.34x** speedup with identical OCR output (18 boxes, same recognized text). GPU utilization was confirmed non-idle (peak ~40% over a sustained loop), so this is real GPU execution, not just provider registration.

## Precision / quantization

- The default onnxruntime backend's ONNX models are **fp32** (`tensor(float)` inputs), and ONNX Runtime does not autocast. There is no bf16-on-Turing pitfall here — this backend is safe on T4-class (sm_75) GPUs from a dtype standpoint.
- RapidOCR does **not** ship int8 or fp16 quantized ONNX model variants for the onnxruntime backend (`default_models.yaml` has none).
- fp16/int8 acceleration is only available through the separate **tensorrt** backend (`EngineConfig.tensorrt.use_fp16` defaults to `true`, `use_int8` is an option), which requires its own TensorRT installation and engine build step — it is not a drop-in flag for the onnxruntime backend.

## Quick checklist for GPU work on this repo

- Installing `onnxruntime-gpu` alone is **not** enough — you also need `use_cuda=True`.
- Setting `use_cuda=True` alone is **not** enough — you also need `onnxruntime-gpu` installed and a matching CUDA/cuDNN runtime.
- Always confirm the actual provider via `get_providers()` rather than assuming config flags took effect.
- If you need fp16/int8, use the `tensorrt` backend, not the default `onnxruntime` backend.
