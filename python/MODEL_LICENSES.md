# RapidOCR Model License and Attribution Notice

This document records the license, copyright attribution, provenance, and
redistribution requirements for OCR model artifacts distributed by RapidOCR.
It does not replace or modify the Apache License, Version 2.0.

**All models registered in `rapidocr/default_models.yaml` are licensed under
the Apache License, Version 2.0 (`Apache-2.0`), without exception.**

## 1. Separate licensed works

RapidOCR contains two separately copyrighted categories of work:

1. **Source code and engineering components.** Copyright is held by the
   RapidOCR Authors. These components are licensed under the Apache License,
   Version 2.0, as described in the RapidOCR project `LICENSE` file.
2. **OCR model artifacts.** Copyright in the upstream model weights is held by
   Baidu and/or the applicable PaddleOCR rights holders. RapidOCR does not
   claim ownership of the upstream model weights.

Both categories are distributed under the Apache License, Version 2.0, but
their copyright ownership and provenance remain separate.

## 2. Scope of this notice

This notice applies to every model registered in
`rapidocr/default_models.yaml`. The registry is the definitive boundary of
this notice: all model artifacts recorded there, across all model versions and
all supported inference formats, are licensed under Apache-2.0. This includes,
without limitation, the legacy `ch_ppocr_mobile_v2.0_cls` model, the PP-OCRv3,
PP-OCRv4, PP-OCRv5, and PP-OCRv6 generations, and their detection,
recognition, classification, and text-line orientation variants, in ONNX, MNN,
Paddle, and PyTorch representations. The declaration is not limited to the
default bundled artifacts listed in Section 6.

The set of models covered by this notice is the set of models registered in
`rapidocr/default_models.yaml`. A model becomes covered when it is registered
there. Downstream redistributors should consult the source record of each
entry to confirm its upstream origin, format, and cryptographic hash. Such
verification identifies the artifact and does not narrow the license scope
described here.

The registry also records non-model resources, including recognition
dictionaries (`dict_url`) and fonts (`fonts`). These resources are not model
artifacts and are not covered by the Apache-2.0 declaration in this notice;
they remain subject to their own applicable terms. Third-party models,
datasets, sample images, inference runtimes, and other assets that are not
derived from official PaddleOCR models are likewise not covered.

## 3. Upstream attribution and licensing evidence

All models registered in `rapidocr/default_models.yaml` are released by, or
derived from models released by, the PaddleOCR project, and are licensed under
the Apache License, Version 2.0. PaddleOCR itself is released under
Apache-2.0. Official PaddlePaddle model cards identify the PP-OCRv6 model
weights and ONNX representations as `Apache-2.0`. PaddleOCR's official model
lists identify the earlier generations, including the legacy text angle
classification model and the PP-OCRv3, PP-OCRv4, and PP-OCRv5 models, and
their official distribution sources.

Upstream references:

- PaddleOCR project: <https://github.com/PaddlePaddle/PaddleOCR>
- PaddleOCR license: <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE>
- PaddleOCR model list: <https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/model_list.md>
- PaddleOCR ONNX conversion documentation: <https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/inference_deployment/others/obtaining_onnx_models.html>
- Official PP-OCRv6 small detection ONNX model card: <https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx>
- Official PP-OCRv6 small recognition ONNX model card: <https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx>
- Official legacy text angle classification model list: <https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_list.en.md>

Legacy models that predate official Hugging Face model hosting, including
`ch_ppocr_mobile_v2.0_cls` and the PP-OCRv3 detection and recognition models,
are identified through PaddleOCR's official model lists and model download
service, and through the source records in `rapidocr/default_models.yaml`.

## 4. Applicable license

Every model registered in `rapidocr/default_models.yaml` is made available
under the Apache License, Version 2.0 (`Apache-2.0`), regardless of model
version, model family, or inference format. This declaration applies to every
entry in the registry without exception. The complete license text is
available at <https://www.apache.org/licenses/LICENSE-2.0>.

Subject to the conditions of Apache-2.0, the license permits use,
reproduction, modification, preparation of derivative works, public display,
sublicensing, and distribution in source or object form. These permissions are
not restricted to non-commercial use.

The license applies only to rights that the relevant licensors are authorized
to grant. No additional model-specific restriction has been identified for any
model registered in `rapidocr/default_models.yaml`.

## 5. Converted model artifacts

RapidOCR converts or repackages upstream PaddleOCR inference models for use
with supported inference engines. These conversions are mechanical
transformations of the model representation. Conversion does not transfer
ownership of the upstream weights to RapidOCR and does not remove the upstream
copyright or attribution.

All models registered in `rapidocr/default_models.yaml`, including their
converted and repackaged representations for every supported inference engine
and format, are distributed under the same Apache-2.0 terms that apply to the
corresponding upstream PaddleOCR models. The conversion and packaging scripts,
configuration, and RapidOCR-authored metadata are separately copyrighted by
the RapidOCR Authors and licensed under Apache-2.0 as part of the RapidOCR
source code.

A RapidOCR-converted artifact may not be byte-for-byte identical to an ONNX or
other representation hosted by PaddlePaddle. Artifact identity must therefore
be established using its source record and cryptographic hash rather than its
filename alone.

## 6. Models registered in default_models.yaml

Every `model_dir` entry in `rapidocr/default_models.yaml` is covered by the
Apache-2.0 declaration in Section 4. The versions recorded in the registry
include:

- Legacy models: `ch_ppocr_mobile_v2.0_cls` (text angle classification) and
  the PP-OCRv3 detection and recognition models.
- PP-OCRv4, PP-OCRv5, and PP-OCRv6 detection, recognition, classification,
  and text-line orientation models, across mobile, server, tiny, small, and
  medium variants and their language variants.
- The corresponding ONNX, MNN, Paddle (`.pdmodel`), and PyTorch (`.pth`)
  representations of those models.

The table below lists representative artifacts distributed with RapidOCR's
default ONNX Runtime configuration. It is illustrative and does not limit the
scope of this notice.

| RapidOCR artifact | Upstream model | SHA-256 | RapidOCR source |
| --- | --- | --- | --- |
| `PP-OCRv6_det_small.onnx` | `PP-OCRv6_small_det` | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` | [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx) |
| `PP-OCRv6_rec_small.onnx` | `PP-OCRv6_small_rec` | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` | [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx) |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | `ch_ppocr_mobile_v2.0_cls` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` | [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx) |

The `dict_url` and `fonts` entries in the registry are resources, not models,
and are excluded from this declaration. `rapidocr/default_models.yaml` is the
authoritative release record for the source URL and SHA-256 value of every
registered model artifact.

## 7. Redistribution requirements

When redistributing a covered model artifact, or a modified or converted form
of one, comply with Section 4 of Apache-2.0. In particular, you must:

1. Give recipients a copy of the Apache License, Version 2.0.
2. Cause modified files to carry prominent notices stating that you changed
   them.
3. Retain applicable copyright, patent, trademark, and attribution notices
   from the upstream work, except notices that do not pertain to the
   redistributed work.
4. If an upstream distribution contains a `NOTICE` file, include the relevant
   contents of that notice in a permitted form.

At the time of publication, RapidOCR has not identified an additional
model-specific `NOTICE` file for any model registered in
`rapidocr/default_models.yaml`. If a future upstream release supplies one, its
applicable contents must be retained.

For converted files, an appropriate modification notice is:

> Converted from an official PaddleOCR model to the indicated inference format
> and packaged for use with RapidOCR. The upstream model weights are copyright
> Baidu and/or the applicable PaddleOCR rights holders and are licensed under
> the Apache License, Version 2.0.

## 8. Trademarks and endorsement

Apache-2.0 does not grant permission to use the trade names, trademarks,
service marks, or product names of Baidu, PaddlePaddle, PaddleOCR, or RapidOCR,
except as required for reasonable and customary description of the origin of
the model artifacts. Redistribution must not imply endorsement, sponsorship,
or affiliation without separate permission.

## 9. Disclaimer

Unless required by applicable law or agreed to in writing, the model artifacts
are provided on an **AS IS** basis, without warranties or conditions of any
kind. The warranty disclaimer and limitation of liability in Apache-2.0 apply.

Model accuracy, fitness, security, and regulatory suitability remain the
responsibility of the user and downstream distributor. RapidOCR is not a
medical device, and neither the model license nor this notice represents a
certification or guarantee for medical, safety-critical, or regulated use.

## 10. Release maintenance

When a RapidOCR release adds or changes a model artifact or model version,
maintainers should:

1. Verify that the upstream model is licensed under Apache-2.0 before adding
   it to `rapidocr/default_models.yaml`.
2. Record the upstream source and immutable version or revision where
   available.
3. Record the conversion tool, tool version, model format, and conversion
   options where available.
4. Record and verify the artifact SHA-256 value.
5. Preserve any applicable upstream copyright, attribution, and `NOTICE`
   material.
6. Keep `rapidocr/default_models.yaml` and the version coverage list in
   Section 6 in sync when model versions or applicable terms change.
