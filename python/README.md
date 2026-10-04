# rapidocr (native PP-OCRv6)

Python package for the `V5_NG` C++ engine. The import surface follows
`rapidocr` on `main`:

```python
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Det.model_path": "PP-OCRv6_det_tiny.onnx",
    "Rec.model_path": "PP-OCRv6_rec_tiny.onnx",
    "Rec.rec_keys_path": "ppocrv6_tiny_dict.txt",
    "Global.use_cls": False,
})
result = engine("page.png")
print(result.txts, result.scores)
print(result.to_json())
```

If model paths are omitted, the package downloads the PP-OCRv6 ONNX bundle
for `Det.model_type` / `Rec.model_type` (`tiny`, `small`, or `medium`).

Build a wheel from a checkout that already has the Vulkan-Headers submodule:

```bash
pip install scikit-build-core
pip install ./python
```

The binding is ctypes over the C ABI (`include/ppocr/ppocr.h`), so the wheel
ships `libppocr` and does not add a second C++ wrapper.

See `docs/API.md` for the differences from `main` (no angle classifier, no
word boxes, PP-OCRv6 only).
