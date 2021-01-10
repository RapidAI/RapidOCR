import numpy as np
import onnx
import onnxruntime as ort
x=np.random.rand(1,3,32,256)
session = ort.InferenceSession("ch_ppocr_mobile_v2.0_det_infer.onnx.new")
onnx_inputs = { session.get_inputs()[0].name: x.astype(np.float32)}
onnx_outputs=session.run(None, onnx_inputs)
print(onnx_outputs)
