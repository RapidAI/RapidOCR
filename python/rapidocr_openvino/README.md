## rapidocr_openvino

❗❗❗注意：基于目前`openvino==2022.3.0`版，存在申请内存不释放的问题，这也就意味着当推理图像很大时，推理完之后，内存会一直占用。详情可参见[issue11939](https://github.com/openvinotoolkit/openvino/issues/11939)

<details open>
<summary>目录</summary>

- [rapidocr\_openvino](#rapidocr_openvino)
    - [安装](#安装)
    - [模型问题](#模型问题)
    - [关于OpenVINO](#关于openvino)
    - [OpenVINO与ONNXRuntime性能对比](#openvino与onnxruntime性能对比)
    - [OpenVINO与ONNXRuntime推理代码写法对比](#openvino与onnxruntime推理代码写法对比)
</details>


#### 安装
```bash
$ pip install openvino

# 里面含有mo
$ pip install openvino-dev
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


#### 关于OpenVINO
- OpenVINO可以直接推理IR、ONNX和PaddlePaddle模型，具体如下(图来源:[link](https://docs.openvino.ai/latest/openvino_docs_OV_UG_OV_Runtime_User_Guide.html#doxid-openvino-docs-o-v-u-g-o-v-runtime-user-guide))：

    <div align="center">
        <img src="https://docs.openvino.ai/latest/_images/BASIC_FLOW_IE_C.svg">
    </div>

- 和ONNXRuntime同时推理同一个ONNX模型，OpenVINO推理速度更快
- 但是从对比来看，OpenVINO占用内存更大，其原因是拿空间换的时间
  - 当指定`input_shape`在一个区间范围时，推理时内存占用会减少一些
  - 示例命令:
    ```bash
    mo --input_model models/ch_PP-OCRv2_det_infer.onnx \
    --output_dir models/IR/static \
    --input_shape "[1,3,960:1200,800]"
    ```

#### OpenVINO与ONNXRuntime性能对比
- 推理设备：`Windows 64位 Intel(R) Core(TM) i5-4210M CPU @ 2.60GHz   2.59 GHz`
- [测试图像宽高](https://drive.google.com/file/d/1iJcGvOVIdUlyOS52bBdvO8uzx8QORo5M/view?usp=sharing): `12119x810`

| 测试模型                             | 推理框架             | 占用内存(3次平均) | 推理时间(3次平均) |
| ------------------------------------ | -------------------- | ----------------- | ----------------- |
| `ch_PP-OCRv2_det_infer.onnx`         | `ONNXRuntime=1.10.0` | 0.8G              | 5.354s            |
| `ch_PP-OCRv2_det_infer.onnx`         | `openvino=2022.1.0`  | 3.225G            | 2.53s             |
| `ch_PP-OCRv2_det_infer.xml` FP32 动态图 | `openvino=2022.1.0`  | 3.175G            | 2.0455s           |


#### OpenVINO与ONNXRuntime推理代码写法对比
NOTE: 以`ch_ppocr_mobile_v2_det`中推理代码为例子

- ONNXRuntime
    ```python
    import onnxruntime

    # 声明
    sess_opt = onnxruntime.SessionOptions()
    sess_opt.log_severity_level = 4
    sess_opt.enable_cpu_mem_arena = False
    session = onnxruntime.InferenceSession(det_model_path, sess_opt)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 推理
    preds = session.run([output_name], {input_name: img})
    ```

- OpenVINO
    ```python
    from openvino.runtime import Core

    # 初始化
    ie = Core()
    model_onnx = ie.read_model(det_model_path)
    compile_model = ie.compile_model(model=model_onnx, device_name='CPU')
    vino_session = compile_model.create_infer_request()

    # 推理
    vino_session.infer(inputs=[img])
    vino_preds = vino_session.get_output_tensor().data
    ```
