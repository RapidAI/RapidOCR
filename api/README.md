### See [Documentation](https://rapidai.github.io/RapidOCRDocs/install_usage/rapidocr_api/usage/)

### API 修改说明

* uvicorn启动时，reload参数设置为False，避免反复加载；
* 增加了启动参数： workers，可启动多个实例，以满足多并发需求。
* 可通过环境变量传递模型参数：det_model_path, cls_model_path, rec_model_path；
* 接口中可传入参数，控制是否使用检测、方向分类和识别这三部分的模型；客户端调用见`demo.py`
* 增加了Dockerfile，可自行构建镜像。

启动服务端：

Windows下启动：

```shell
set det_model_path=I:\models\图像相关\OCR\RapidOCR\PP-OCRv4\ch_PP-OCRv4_det_server_infer.onnx
set det_model_path=

set rec_model_path=I:\models\图像相关\OCR\RapidOCR\PP-OCRv4\ch_PP-OCRv4_rec_server_infer.onnx
rapidocr_api
```

Linux下启动：

```shell
# 默认参数启动
rapidocr_api

# 指定参数：端口与进程数量；
rapidocr_api -ip 0.0.0.0 -p 9005 -workers 2

# 指定模型
expert det_model_path=/mnt/sda1/models/PP-OCRv4/ch_PP-OCRv4_det_server_infer.onnx
expert rec_model_path=/mnt/sda1/models/PP-OCRv4/ch_PP-OCRv4_rec_server_infer.onnx
rapidocr_api -ip 0.0.0.0 -p 9005 -workers 2
```

客户端调用说明：

```bash
cd api
python demo.py
```

构建镜像:

```bash
cd api
sudo docker build -t="rapidocr_api:0.1.1" .
```

启动镜像：

```bash
docker run -p 9003:9003 --name rapidocr_api1 --restart always -d rapidocr_api:0.1.1
```
