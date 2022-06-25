### 📌2022-06-25 update:
- Reorganize some of the python inference code, put all the common adjustment parameters into the yaml file, easy to adjust, easier to use, see: [README](./python/README.md)
- The old inference code is located in the branch: [old_python_infer](https://github.com/RapidAI/RapidOCR/tree/old_python_infer)

#### 🍿2022-05-15 update
- Add the ONNX model converted from the PaddleOCR v3 rec model, just go to the network disk to download and replace it. ([Baidu Netdisk](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing ))
- Added a comparison table of the effects of each version of the text recognition model. For details, click [Comparison of the effects of various versions of ONNX models] (#Comparison of the effects of various versions of onnx models). The text recognition model of v3 is not as good as the previous one in terms of the indicators on the test set constructed by itself.

#### 😀2022-05-12 upadte
- Add the ONNX model converted from the PaddleOCR v3 det model, download it directly from the network disk, and replace it. ([Baidu Netdisk](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing ))
- Added a comparison table of text detection model effects of various versions. For details, click [Comparison of the effects of various versions of ONNX models] (#Comparison of the effects of various versions of onnx models). The text detection model of v3 is better than the previous v2 in terms of the indicators on the test set constructed by itself, and it is recommended to use it.

#### 2022-02-24 update
- Optimize the inference code of the python part.
- Add inference code examples that use the different language models.
- For details, see: [python/README](./python/README.md)

#### 2021-12-18 update
- Add [Google Colab Demo](https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/RapidOCRDemo.ipynb).
- Change the default det model of the `python/rapidOCR.sh`

#### 2021-11-28 update
- Update the [ocrweb](http://rapidocr.51pda.cn:9003/) part
  - Add the display of the inference time of each stage.
  - Add docs of the ocrweb.
  - Change the det model(`ch_PP-OCRv2_det_infer.onnx`), faster and more accurate.

#### 2021-11-13 update
- Add adjustable super parameters for text detection and recognition in Python version, mainly `box_thresh|unclip_ratio|text_score`, see [parameter adjustment](python/README.md#相关调节参数) for details
- The dictionary position in text recognition is given in parameter mode to facilitate flexible configuration. See [keys_path](python/rapidOCR.sh) for details

#### 2021-10-27 update
- Add the code that uses the onnxruntime GPU version of infering follow the [official tutorial](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) Configuration. (however, the onnxruntime GPU version is not stable to use)

- See: `python/README.md` for specific steps.

#### 2021-09-13 update
- Add a whl file based on `Python` for ease of use. See `release/python` for details.

#### 2021-09-11 update
- Add `PP-OCRv2` new model onnx version.
- The infering code of the method is unchanged, and the corresponding model can be directly replaced.
- After evaluation on its own test set:
    - The effect of `PP-OCRv2` detection model has been greatly improved, and the model size has not changed.
    - The effect of `PP-OCRv2` recognition model was not significantly improved, and the model size increased by 3.58M.

- Upload the model to [Baidu online disk extraction code: 30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg) or [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)

#### 2021-08-07 update
- TODO:
    - [ ] PP structure table structure and cell coordinate prediction are being sorted out.

- Previously done, unfinished, welcome to PR
    - [ ] make dokcer image
    - [x] try onnxruntime GPU reasoning

#### 2021-07-17 update
- Improve the README document
- Add **English, number recognition**onnx model, please refer to `python/en_number_ppocr_mobile_v2_rec` for details, the usage is the same as others
- Organize [Model to onnx](#model-related)

#### 2021-07-04 update
- The python program under the repository can be successfully run on the Raspberry Pi 4B. For more information, please enter the QQ group and ask the group owner
- Update the overall structure diagram and add support for Raspberry Pi

#### 2021-06-20 update
- Optimize the display of recognition results in ocrweb, and add recognition animations to demonstrate at the same time
- Update the `datasets` directory, add some commonly used database links

#### 2021-06-10 update
- Add server version text recognition model, see details [Extract code：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)

#### 2021-06-08 update
- Organize the warehouse and unify the model download path
- Improve related documentation

#### 2021-03-24 update
- The new model is fully compatible with ONNXRuntime 1.7 or higher. Special thanks: @Channingss
- The performance of the new version of onnxruntime is improved by more than 40% compared to 1.6.0.