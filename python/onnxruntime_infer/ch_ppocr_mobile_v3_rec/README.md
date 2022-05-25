
#### 简单对比

|输入尺寸|前处理函数|模型|测试集|Score|Exact_Match|Char_Match|Rec_Speed(s/条)|
|---|---|---|---|---|---|---|---|
|48 |resize_norm_img_svtr |ch_PP-OCRv3_rec_infer.onnx|中英印刷体|0.5011|0.3571|0.6450|0.0203|
|48 |resize_norm_img |ch_PP-OCRv3_rec_infer.onnx|中英印刷体|0.7248|0.5417|0.9079|0.0209|
|32 | resize_norm_img_svtr |ch_PP-OCRv3_rec_infer.onnx|中英印刷体|0.3019|0.1607|0.4431|0.0161|
|32| resize_norm_img |ch_PP-OCRv3_rec_infer.onnx|中英印刷体|0.5537|0.3274|0.7800|0.0152|
