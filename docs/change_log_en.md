#### 🎉 2023-01-21 update:
- \[python\] Add an image orientation classification module containing text. For details, see [Rapid Orientation](../python/rapid_structure/docs/README_Orientation.md)

#### ⚽2022-12-19 update:
- \[python\] Add the table recovery module, See [Rapid Table](../python/rapid_structure/docs/README_Table.md) for details.

#### 🤖2022-12-14 update:
- \[python\] Move the configuration parameters and model into the module, and at the same time put the model into the whl package, which can be directly installed and used by pip, which is more convenient and quicker.
- For details, see: [README](../python/README.md#推荐pip安装快速使用)

#### 🧻2022-11-20 update:
- \[python\] Add the layout analysis part, which supports the detection and analysis of three layouts: Chinese, English and tables. See: [Rapid Structure](../python/rapid_structure/README.md) section for details.

#### 🎃2022-11-01 update:
- Add Hugging Face Demo, add module that can adjust hyperameters, for details, please visit [Hugging Face Demo](https://huggingface.co/spaces/SWHL/RapidOCRDemo).

#### 🚩2022-10-01 update:
- Fix some minor bugs under python section.
- Merge the [OCRWeb Implementation of Multilingual Deployment](https://github.com/RapidAI/RapidOCR/pull/46) demo from [AutumnSun1996](https://github.com/AutumnSun1996), see for details: [ocrweb_mutli-README](../ocrweb_multi/README.md)
- Add a description of the problem that onnxruntime-gpu inference speed is slower than CPU. For details, please refer to: [onnxruntime-gpu version related instructions](../python/README.md#onnxruntime-gpu版相关说明)

#### 🛴2022-09-01 update:
- Since openvino released version 2022.2.0.dev20220829, this version solves the problem of `cls` partial model inference. So far, the openvino-based rapidocr has been unified, and it is all completed by the openvino inference engine.
- For detailed usage, see: [python/README](../python/README.md#源码使用步骤).

#### 🧸2022-08-17 update:
- The python/ocrweb part v1.1.0 is released, see the [link]((https://github.com/RapidAI/RapidOCR/releases/tag/v1.1.0)) for details.

#### 🕶2022-08-14 update:
- The ocrweb part adds the function of deploying calls by API, and you can send POST requests to get OCR recognition results.
- For details, see: [API deploy](../ocrweb/README.md#以api方式运行和调用)

#### 🎧2022-07-10 update:
- Add test case for onnxruntime-gpu → [link](../python/README.md#onnxruntime-gpu version inference configuration)
- Add benchamark test set → [link](../images/README.md)
- Add actions to automatically publish whl packages, when modifying python/rapidocr_onnxruntime, it will automatically update the published whl packages. See [WHL README](../docs/doc_whl_en.md)

#### ✨2022-07-07 update:
- Fix the bug of v3 rec inference in python version, and merge v3 rec and v2 rec into the same set of inference code, which is more concise and convenient.
- Add unit test under python module.
- Add [Acknowledgement module](../docs/README_en.md#Acknowledgement) to this page to thank the partners who contributed to this project.

#### 😁2022-07-05 update:
- Add the ability to handle single line text, for single line text, you can set your own threshold, but the detection module, direct recognition can be. For details, see [README](../python/README.md#configyamlconfigyaml中常用参数介绍)
- Optimize some code logic of python, more elegant and concise.

#### 🏝2022-06-30 update:
- In the python inference section, add a configuration option for whether to use GPU inference, which can be used with one click if the `onnxruntime-gpu` version is correctly installed (Fix [issue#30](https://github.com/RapidAI/RapidOCR/issues/30))
- The specific GPU-based reasoning will need to wait for me to sort it out and update it later.
- For details, see: [onnxruntime-gpu version reasoning configuration](../python/README.md#onnxruntime-gpu version reasoning configuration)

#### 📌2022-06-25 update:
- Reorganize some of the python inference code, put all the common adjustment parameters into the yaml file, easy to adjust, easier to use, see: [README](../python/README.md)
- The old inference code is located in the branch: [old_python_infer](https://github.com/RapidAI/RapidOCR/tree/old_python_infer)

#### 🍿2022-05-15 udpate:
- Add the ONNX model converted from the PaddleOCR v3 rec model, just go to the network disk to download and replace it. ([Baidu Netdisk](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing ))
- Added a comparison table of the effects of each version of the text recognition model. For details, click [Comparison of the effects of various versions of ONNX models] (#Comparison of the effects of various versions of onnx models). The text recognition model of v3 is not as good as the previous one in terms of the indicators on the test set constructed by itself.

#### 😀2022-05-12 udpate:
- Add the ONNX model converted from the PaddleOCR v3 det model, download it directly from the network disk, and replace it. ([Baidu Netdisk](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing ))
- Added a comparison table of text detection model effects of various versions. For details, click [Comparison of the effects of various versions of ONNX models] (#Comparison of the effects of various versions of onnx models). The text detection model of v3 is better than the previous v2 in terms of the indicators on the test set constructed by itself, and it is recommended to use it.

#### 2022-02-24 udpate:
- Optimize the inference code of the python part.
- Add inference code examples that use the different language models.
- For details, see: [python/README](../python/README.md)

#### 2021-12-18 udpate:
- Add [Google Colab Demo](https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/RapidOCRDemo.ipynb).
- Change the default det model of the `python/rapidOCR.sh`

#### 2021-11-28 udpate:
- Update the [ocrweb](http://rapidocr.51pda.cn:9003/) part
  - Add the display of the inference time of each stage.
  - Add docs of the ocrweb.
  - Change the det model(`ch_PP-OCRv2_det_infer.onnx`), faster and more accurate.

#### 2021-11-13 udpate:
- Add adjustable super parameters for text detection and recognition in Python version, mainly `box_thresh|unclip_ratio|text_score`, see [parameter adjustment](../python/README.md#相关调节参数) for details
- The dictionary position in text recognition is given in parameter mode to facilitate flexible configuration. See [keys_path](python/rapidOCR.sh) for details

#### 2021-10-27 udpate:
- Add the code that uses the onnxruntime GPU version of infering follow the [official tutorial](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) Configuration. (however, the onnxruntime GPU version is not stable to use)

- See: `python/README.md` for specific steps.

#### 2021-09-13 udpate:
- Add a whl file based on `Python` for ease of use. See `release/python` for details.

#### 2021-09-11 udpate:
- Add `PP-OCRv2` new model onnx version.
- The infering code of the method is unchanged, and the corresponding model can be directly replaced.
- After evaluation on its own test set:
    - The effect of `PP-OCRv2` detection model has been greatly improved, and the model size has not changed.
    - The effect of `PP-OCRv2` recognition model was not significantly improved, and the model size increased by 3.58M.

- Upload the model to [Baidu online disk extraction code: 30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg) or [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)

#### 2021-08-07 udpate:
- [x] PP structure table structure and cell coordinate prediction are being sorted out.

- Previously done, unfinished, welcome to PR
    - [ ] make dokcer image
    - [x] try onnxruntime GPU reasoning

#### 2021-07-17 udpate:
- Improve the README document
- Add **English, number recognition**onnx model, please refer to `python/en_number_ppocr_mobile_v2_rec` for details, the usage is the same as others
- Organize [Model to onnx](#model-related)

#### 2021-07-04 udpate:
- The python program under the repository can be successfully run on the Raspberry Pi 4B. For more information, please enter the QQ group and ask the group owner
- Update the overall structure diagram and add support for Raspberry Pi

#### 2021-06-20 udpate:
- Optimize the display of recognition results in ocrweb, and add recognition animations to demonstrate at the same time
- Update the `datasets` directory, add some commonly used database links

#### 2021-06-10 udpate:
- Add server version text recognition model, see details [Extract code：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)

#### 2021-06-08 udpate:
- Organize the warehouse and unify the model download path
- Improve related documentation

#### 2021-03-24 udpate:
- The new model is fully compatible with ONNXRuntime 1.7 or higher. Special thanks: @Channingss
- The performance of the new version of onnxruntime is improved by more than 40% compared to 1.6.0.