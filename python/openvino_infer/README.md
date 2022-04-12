#### 安装
```bash
# Windows端
pip install openvino==2022.1.0

# 里面含有mo
pip install openvino-dev==2022.1.0
```

#### 模型问题
- 因为OpenVINO可以直接推理ONNX模型，故这里暂时不作转换，直接推理之前ONNX模型即可
- 这里仍然给出转换的代码，用作参考:
    ```bash
    mo --input_model models/ch_PP-OCRv2_det_infer.onnx --output_dir models/IR/

    mo --input_model models/ch_PP-OCRv2_det_infer.onnx \
    --output_dir models/IR/static \
    --input_shape "[1,3,12128,800]"
    ```

#### 使用步骤
同[onnxruntime_infer](../onnxruntime_infer/README.md)