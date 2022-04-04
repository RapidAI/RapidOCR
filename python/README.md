### 目录说明
- `onnxruntime_infer`: 基于ONNXRuntime推理引擎推理
- `openvino_infer`: 基于OpenVINO推理引擎推理

### TODO
- [ ] 模型转INT8，尚未找到转换代码
- [ ] 其他推理模型代码整理
- [ ] 模型转换代码整理

### 关于OpenVINO
- OpenVINO可以直接推理ONNX模型，可以不用转换直接使用之前ONNX模型
- OpenVINO推理速度更快，但是从对比来看，占用内存更多，目前正在排查原因

### OpenVINO与ONNXRuntime性能对比
- 推理设备：`Windows 64位 Intel(R) Core(TM) i5-4210M CPU @ 2.60GHz   2.59 GHz`
- 测试图像宽高: `12119x810`

| 测试模型                             | 推理框架             | 占用内存(3次平均) | 推理时间(3次平均) |
| ------------------------------------ | -------------------- | ----------------- | ----------------- |
| `ch_PP-OCRv2_det_infer.onnx`         | `ONNXRuntime=1.10.0` | 0.8G              | 5.354s            |
| `ch_PP-OCRv2_det_infer.onnx`         | `openvino=2022.1.0`  | 3.225G            | 2.53s             |
| `ch_PP-OCRv2_det_infer.xml` FP32 | `openvino=2022.1.0`  | 3.175G            | 2.0455s           |


### OpenVINO与ONNXRuntime推理代码写法对比
NOTE: 以`ch_ppocr_mobile_v2_det`中推理代码为例子

- ONNXRuntime
```python
import onnxruntime

# 声明
sess_opt = onnxruntime.SessionOptions()
sess_opt.log_severity_level = 4
sess_opt.enable_cpu_mem_arena = False
self.session = onnxruntime.InferenceSession(det_model_path, sess_opt)
self.input_name = self.session.get_inputs()[0].name
self.output_name = self.session.get_outputs()[0].name

# 推理
preds = self.session.run([self.output_name],
                            {self.input_name: img})
```

- OpenVINO
```python
from openvino.runtime import Core

# 初始化
ie = Core()
model_onnx = ie.read_model(det_model_path)
compile_model = ie.compile_model(model=model_onnx,
                                    device_name='CPU')
self.vino_session = compile_model.create_infer_request()

# 推理
self.vino_session.infer(inputs=[img])
vino_preds = self.vino_session.get_output_tensor().data
```

