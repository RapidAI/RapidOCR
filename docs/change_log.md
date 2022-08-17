#### 🥹2022-08-17 update:
- python/ocrweb部分 v1.1.0发布，详情参见[v1.1.0](https://github.com/RapidAI/RapidOCR/releases/tag/v1.1.0)

#### 🕶2022-08-14 update:
- ocrweb部分增加以API方式部署调用的功能，可以通过发送POST请求，来获得OCR识别结果。
- 详情参见：[API方式调用](../ocrweb/README.md#以api方式运行和调用)

#### ✨2022-07-07 update:
- 修复python版中v3 rec推理bug，并将v3 rec与v2 rec合并为同一套推理代码，更加简洁和方便
- 添加python模块下的单元测试
- 该页面添加[致谢模块](../README.md#致谢)，感谢为这个项目作出贡献的小伙伴。

#### 😁2022-07-05 update:
- 添加对单行文本的处理能力，对于单行文本，可自行设定阈值，不过检测模块，直接识别即可。详情参见[README](./python/README.md#configyamlconfigyaml中常用参数介绍)
- 优化python部分代码逻辑，更优雅简洁。

#### 🏝2022-06-30 update:
- python推理部分，增加参数选择使用GPU推理的配置选项，在正确安装`onnxruntime-gpu`版本前提下，可以一键使用（Fix [issue#30](https://github.com/RapidAI/RapidOCR/issues/30)）
- 具体基于GPU的推理情况，需要等我后续整理一下，再更新出来
- 详情参见：[onnxruntime-gpu版推理配置](./python/README.md#onnxruntime-gpu版推理配置)

#### 📌2022-06-25 update:
- 重新整理python部分推理代码，将常用调节参数全部放到yaml文件中，便于调节，更加容易使用
- 详情参见：[README](./python/README.md)

#### 🍿2022-05-15 update
- 增加PaddleOCR v3 rec模型转换后的ONNX模型，直接去网盘下载替换即可。([百度网盘](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing))
- 增加文本识别模型各个版本效果对比表格，详情点击[各个版本ONNX模型效果对比](#各个版本onnx模型效果对比)。v3的文本识别模型从自己构建测试集上的指标来看不如之前的好。

#### 😀2022-05-12 upadte
- 增加PaddleOCR v3 det模型转换的ONNX模型，直接去网盘下载，替换即可。([百度网盘](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing))
- 增加各个版本文本检测模型效果对比表格，详情点击[各个版本ONNX模型效果对比](#各个版本onnx模型效果对比)。v3的文本检测模型从指标来看是好于之前的v2的，推荐使用。

#### 🎧2022-04-04 update
- 增加python下的基于OpenVINO推理引擎的支持
- 给出OpenVINO和ONNXRuntime的性能对比表格
- 详情参见:[python/README](./python/README.md)

#### 2022-02-24 update
- 优化python目录下的推理代码
- 添加调用不同语言模型的推理代码示例
- 详情参见：[python/onnxruntime_infer/README](./python/onnxruntime_infer/README.md)

#### 2021-12-18 update
- 添加[Google Colab Demo](https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/RapidOCRDemo.ipynb)

#### 2021-11-28 update
- 更新[ocrweb](http://rapidocr.51pda.cn:9003/)部分
  - 添加显示各个阶段处理时间
  - 更新说明文档
  - 更换文本检测模型为`ch_PP-OCRv2_det_infer.onnx`,推理更快，更准

#### 2021-11-13 update
- 添加python版本中文本检测和识别可调节的超参数，主要有`box_thresh|unclip_ratio|text_score`，详情见[参数调节](python/README.md#相关调节参数)
- 将文本识别中字典位置以参数方式给出，便于灵活配置，详情见[keys_path](python/rapidOCR.sh)

#### 2021-10-27 update
- 添加使用onnxruntime-gpu版推理的代码（不过gpu版本的onnxruntime不太好用，按照[官方教程](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)配置，感觉没有调用起来GPU）
- 具体使用步骤参见: [onnxruntime-gpu推理配置](python/README.md)

#### 2021-09-13 update
- 添加基于`python`的whl文件，便于使用，详情参见`release/python_sdk`

#### 2021-09-11 update
- 添加PP-OCRv2新增模型onnx版本
  - 使用方法推理代码不变，直接替换对应模型即可。
- 经过在自有测试集上评测：
  - PP-OCRv2检测模型效果有大幅度提升，模型大小没变。
  - PP-OCRv2识别模型效果无明显提升，模型大小增加了3.58M。
- 模型上传到[百度网盘 提取码：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)

#### 2021-08-07 update
- 现在正在做的
  - [ ] PP-Structure 表格结构和cell坐标预测 正在整理中

- 之前做的,未完成的，欢迎提PR
  - [ ] 打Dokcer镜像
  - [x] 尝试onnxruntime-gpu推理

#### 2021-07-17 update
- 完善README文档
- 增加**英文、数字识别**onnx模型，具体参见`python/en_number_ppocr_mobile_v2_rec`，用法同其他
- 整理一下[模型转onnx](#模型转onnx)

#### 2021-07-04 update
- 目前仓库下的python程序已经可以在树莓派4B上，成功运行，详细信息请进群，询问群主
- 更新整体结构图，添加树莓派的支持

#### 2021-06-20 update
- 优化ocrweb中识别结果显示，同时添加识别动图演示
- 更新`datasets`目录，添加一些常用数据库链接(搬运一下^-^)
- 更新[FAQ](./FAQ.md)

#### 2021-06-10 update
- 添加server版文本识别模型，详情见[提取码：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)

#### 2021-06-08 update
- 整理仓库，统一模型下载路径
- 完善相关说明文档

#### 2021-03-24 update
- 新模型已经完全兼容ONNXRuntime 1.7 或更高版本。 特别感谢：@Channingss
- 新版onnxruntime比1.6.0 性能提升40%以上。
