## 模型相关
- [模型相关](#模型相关)
  - [模型转换](#模型转换)
  - [模型下载](#模型下载)
  - [各个版本ONNX模型效果对比](#各个版本onnx模型效果对比)
    - [文本检测模型(仅供参考)](#文本检测模型仅供参考)
    - [文本识别模型(仅供参考)](#文本识别模型仅供参考)

### 模型转换
- [PaddleOCRModelConverter](https://github.com/RapidAI/PaddleOCRModelConverter)
- [Paddle2OnnxConvertor](https://github.com/RapidAI/Paddle2OnnxConvertor)
- [手把手教你使用ONNXRunTime部署PP-OCR](https://aistudio.baidu.com/aistudio/projectdetail/1479970?channelType=0&channel=0)

### 模型下载
- 可以直接下载使用的模型 ([百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing))

|模型名称|模型简介|模型大小|备注|
|:---:|:---:|:---:|:---:|
|⭐ ch_PP-OCRv3_det_infer.onnx|轻量文本检测模型|2.23M|较v1轻量检测，精度有较大提升 from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/README_ch.md#pp-ocr%E7%B3%BB%E5%88%97%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8%E6%9B%B4%E6%96%B0%E4%B8%AD)|
|⭐ ch_PP-OCRv2_rec_infer.onnx|轻量文本识别模型|7.79M||

### 各个版本ONNX模型效果对比
#### 文本检测模型(仅供参考)
- 测试集：自己构建`中英文(111个，包含卡证、文档和自然图像)`

|                模型                  | infer_Speed(s/img) | precision | recall | hmean  | 模型大小 |
| :---------------------------------: | :----------------: | :-------: | :----: | :----: | :------: |
| ch_ppocr_mobile_v2.0_det_infer.onnx |     0.4345742      |  0.7277   | 0.8413 | 0.7785 |   2.3M   |
|     ch_PP-OCRv2_det_infer.onnx      |     0.5116553      |  0.7817   | 0.8472 | 0.8123 |   2.3M   |
|     ch_PP-OCRv3_det_infer.onnx      |     0.5723512      |  **0.7740**   | **0.8837** | **0.8237** |   2.4M   |

#### 文本识别模型(仅供参考)
- 测试集: 自己构建`中英文(168个)`

|                模型                 | infer_Speed(s/img)   | Score     |    Exact_Match   |   Char_Match | 模型大小 |
| :---------------------------------: | :------------------: | :-------: | :--------------: | :-------------: | :--: |
| ch_ppocr_mobile_v2.0_rec_infer.onnx |       0.0111        |  **0.7287**   |      **0.5595**      |     0.8979      | 4.3M |
|     ch_PP-OCRv2_rec_infer.onnx      |       0.0193        |  0.6955   |      0.4881      |     **0.9029**      | 8.0M |
|     ch_PP-OCRv3_rec_infer.onnx      |       0.0145        |  0.5537   |      0.3274      |     0.7800      |  11M |
| ch_PP-OCRv3_rec_train_student.onnx  |       0.0157        |  0.5537   |      0.3274      |     0.7800      | 11M  |
| ch_PP-OCRv3_rec_train_teacher.onnx  |       0.0140        |  0.5381   |      0.3095      |     0.7667      | 11M  |
