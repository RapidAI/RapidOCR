#### 安装
```bash
# Windows端
pip install openvino==2022.1.0
pip install openvino-dev==2022.1.0
```

#### 模型转换为IR
```bash
# 转动态模型 占用内存平均为3.5G左右
mo --input_model models/ch_PP-OCRv2_det_infer.onnx --output_dir models/IR/

# 转静态模型 占用内存平均为1G左右
mo --input_model models/ch_PP-OCRv2_det_infer.onnx \
   --output_dir models/IR/static \
   --input_shape 1,3, 12128,800
```

#### 测试图像下载
- [long1.jpg](https://drive.google.com/file/d/1iJcGvOVIdUlyOS52bBdvO8uzx8QORo5M/view?usp=sharing)
