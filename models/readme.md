## onnx模型下载
### 最新模型 (`2021-06-07 update`)
- 支持`onnxruntime>=1.7.0`，速度更快，内存占用更少
- 下载链接：[提取码：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)

### 之前模型
- 只支持`onnxruntime==1.5.0`
- [点击下载](https://github.com/RapidOCR/RapidOCR/releases/download/V1.0/rapid-model.tgz)

---
## 2021-06-07 update: 后续会逐步更新这块，以下只是暂时的
## onnx模型转换说明
### 一、如何获得模型

* 您可以通过使用paddle全家桶，通过其完善的说明，事无巨细的指引，手把手地让您从零开始，训练自己的模型。
* 或者，您也可以直接去paddle官网下载预先训练好的模型。
* 这个预先训练好的模型叫作train模型，server模型体积较大(准确度高速度慢些)，mobile模型体积较小(准确度相对低速度快)。
* det和rec有server和mobile两种模型，cls只有mobile模型。
* 预训练模型下载地址: [model_list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/models_list.md)

* 使用`tools`下的`getmodels.sh`脚本下载并准备。

### 二、准备环境和工具

* 开始之前，需要先准备好Linux，paddle的pip包不支持macOS，而windows部署起来麻烦一些，以下说明以deepin20为范例。
* 安装python3: ```sudo apt-get install python3```
* 安装pip3:
     ```shell
     wget https://bootstrap.pypa.io/get-pip.py
     sudo python3 get-pip.py
     ```

* pip安装paddlepaddle( 2.0的模型需要2.0版):
     ```shell
     sudo python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
     ```

- pip安装转换工具附加依赖:
     ```shell
     sudo pip3 install setuptools tqdm opencv-python imgaug pyclipper -i https://mirror.baidu.com/pypi/simple
     ```

* 安装lmdb依赖:
     ```shell
     sudo apt-get install python3-yaml libxml2-dev libxslt1-dev python3-dev
     ```
* pip安装lmdb:
     ```shell
     sudo pip3 install lmdb -i https://mirror.baidu.com/pypi/simple
     ```
* pip安装onnx:
     ```shell
     sudo pip3 install onnx
     ```
* 下载PaddleOCR(用于预训练模型转paddle推理模型):
     ```shell
     git clone https://github.com/PaddlePaddle/PaddleOCR.git
     ```
* 安装转换工具库Paddle2ONNX(develop分支,用于paddle推理模型转onnx模型):
     ```shell
     # 转换工具修复，来自于官方仓库，包括未合并代码
     git clone https://github.com/RapidOCR/paddle2onnx-wild.git
     cd paddle2onnx-wild
     sudo python3 setup.py install
     ```

### 三、开始转换

* 接下来的操作都在PaddleOCR文件夹里进行
* PaddleOCR切换到release/2.0分支(因为train训练模型也是v2.0格式的)
     ```shell
     git clone https://github.com/PaddlePaddle/PaddleOCR.git
     cd PaddleOCR
     git checkout release/2.0
     ```

* 生成多语言识别模型配置文件

     ```shell
     #安装模块依赖
     sudo pip3 install pyyaml
     cd PaddleOCR/configs/rec/multi_language

     #生成多语言配置文件
     python3 generate_multi_language_configs.py -l it
     python3 generate_multi_language_configs.py -l es
     python3 generate_multi_language_configs.py -l pt
     python3 generate_multi_language_configs.py -l ru
     python3 generate_multi_language_configs.py -l ar
     python3 generate_multi_language_configs.py -l ta
     python3 generate_multi_language_configs.py -l ug
     python3 generate_multi_language_configs.py -l fa
     python3 generate_multi_language_configs.py -l ur
     python3 generate_multi_language_configs.py -l rs_latin
     python3 generate_multi_language_configs.py -l oc
     python3 generate_multi_language_configs.py -l rs_cyrillic
     python3 generate_multi_language_configs.py -l bg
     python3 generate_multi_language_configs.py -l uk
     python3 generate_multi_language_configs.py -l be
     python3 generate_multi_language_configs.py -l te
     python3 generate_multi_language_configs.py -l kn
     python3 generate_multi_language_configs.py -l ch_tra
     python3 generate_multi_language_configs.py -l hi
     python3 generate_multi_language_configs.py -l mr
     python3 generate_multi_language_configs.py -l ne
     ```

* 新建文件夹`PaddleOCR/inference`
* 需要确认系统中有没有`wget`工具，运行下`wget`检查确认
     ```shell
     # 在ubuntu下安装
     $ apt install wget

     # 在centos安装：
     $ yum install wget
     ```

* 将本仓库`tools`目录下的`getmodels.sh` 放`PaddleOCR/inference`下

* 执行 chmod +x getmodels.h
     应该是如下结构：
     ```text
     PaddleOCR/inference
     |
     |-- getmodels.sh

     ```
* 运行`getmodels.sh`脚本： `./getmodels.sh`
*  运行完后将有如下的目录结构(大约是这样，有一些小的名字差异，懒得改了，不要太介意）
     <details>
     <summary>PaddleOCR/inference目录结构 (click to expand)</summary>

          ```text
          PaddleOCR/inference
          ├── ar_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── be_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── bg_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ch_ppocr_mobile_v2.0_cls_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ch_ppocr_mobile_v2.0_det_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ch_ppocr_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdparams
          │    └── train.log
          ├── ch_ppocr_server_v2.0_det_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ch_ppocr_server_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── chinese_cht_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── en_number_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── fa_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── french_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── german_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── hi_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── it_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── japan_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ka_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── korean_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── mr_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ne_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── oc_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── pu_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── rs_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── rsc_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ru_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ta_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── te_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ug_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── uk_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          ├── ur_mobile_v2.0_rec_train
          │    ├── best_accuracy.pdopt
          │    ├── best_accuracy.pdparams
          │    ├── best_accuracy.states
          │    └── train.log
          └── xi_mobile_v2.0_rec_train
               ├── best_accuracy.pdopt
               ├── best_accuracy.pdparams
               ├── best_accuracy.states
               └── train.log
          ```
     </details>

* 把本项目`tools`文件夹里的`export_custom.py`复制到`PaddleOCR/tools`里，把`program.py`覆盖到`PaddleOCR/tools`里
* 把本项目`tools`文件夹里的`gen-models.sh`复制到`PaddleOCR`根目录
* 开始转换

     ```shell
     cd PaddleOCR
     chmod a+x gen-models.sh
     ./gen-models.sh
     ```

* 转换成功后，会在`PaddleOCR/inference`文件夹里生成31个onnx模型
