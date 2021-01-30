## Download models

* from baiduNetDisk:

链接：https://pan.baidu.com/s/1UhDrC7iOMWiYaQiHk3acdw 提取码：yjgv

## onnx模型转换说明

### 一、如何获得模型

* 您可以通过使用paddle全家桶，通过其完善的说明，事无巨细的指引，手把手地让您从零开始，训练自己的模型。
* 或者，您也可以直接去paddle官网下载预先训练好的模型。
* 这个预先训练好的模型叫作train模型，server模型体积较大(准确度高速度慢些)，mobile模型体积较小(准确度相对低速度快)。
* det和rec有server和mobile两种模型，cls只有mobile模型。
* 预训练模型下载地址:

|模型名称|模型简介|配置文件|下载地址|
| --- | --- | --- | --- |
|mobile cls train|文本方向分类|rec_chinese_lite_train_v2.0.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) |
|ch_ppocr_mobile_v2.0_det|超轻量模型，多语种文本检测|ch_det_mv3_db_v2.0.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|
|ch_ppocr_server_v2.0_det|通用模型，多语种文本检测|ch_det_res18_db_v2.0.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)|
|ch_ppocr_mobile_v2.0_rec|超轻量模型，简体中英文数字|rec_chinese_lite_train_v2.0.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) |
|ch_ppocr_server_v2.0_rec|通用模型，简体中英文数字|rec_chinese_common_train_v2.0.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) |
|en_number_mobile_v2.0_rec|英文数字|rec_en_number_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |
|french_mobile_v2.0_rec|法文|rec_french_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_train.tar) |
|german_mobile_v2.0_rec|德文|rec_german_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_train.tar) |
|korean_mobile_v2.0_rec|韩文|rec_korean_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_train.tar) |
|japan_mobile_v2.0_rec|日文|rec_japan_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_train.tar) |
|it_mobile_v2.0_rec|意大利文|rec_it_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_train.tar) |
|xi_mobile_v2.0_rec|西班牙文|rec_xi_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/xi_mobile_v2.0_rec_train.tar) |
|pu_mobile_v2.0_rec|葡萄牙文|rec_pu_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pu_mobile_v2.0_rec_train.tar) |
|ru_mobile_v2.0_rec|俄罗斯文|rec_ru_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_train.tar) |
|ar_mobile_v2.0_rec|阿拉伯文|rec_ar_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_train.tar) |
|hi_mobile_v2.0_rec|印地文|rec_hi_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_train.tar) |
|chinese_cht_mobile_v2.0_rec|繁体中文|rec_chinese_cht_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_train.tar) |
|ug_mobile_v2.0_rec|维吾尔文|rec_ug_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_train.tar) |
|fa_mobile_v2.0_rec|波斯文|rec_fa_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_train.tar) |
|ur_mobile_v2.0_rec|乌尔都文|rec_ur_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_train.tar) |
|rs_mobile_v2.0_rec|塞尔维亚文(latin)|rec_rs_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_mobile_v2.0_rec_train.tar) |
|oc_mobile_v2.0_rec|欧西坦文|rec_oc_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_train.tar) |
|mr_mobile_v2.0_rec|马拉地文|rec_mr_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_train.tar) |
|ne_mobile_v2.0_rec|尼泊尔文|rec_ne_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_train.tar) |
|rsc_mobile_v2.0_rec|塞尔维亚文(cyrillic)|rec_rsc_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rsc_mobile_v2.0_rec_train.tar) |
|bg_mobile_v2.0_rec|保加利亚文|rec_bg_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_train.tar) |
|uk_mobile_v2.0_rec|乌克兰文|rec_uk_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_train.tar) |
|be_mobile_v2.0_rec|白俄罗斯文|rec_be_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_train.tar) |
|te_mobile_v2.0_rec|泰卢固文|rec_te_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_train.tar) |
|ka_mobile_v2.0_rec|卡纳达文|rec_ka_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_train.tar) |
|ta_mobile_v2.0_rec|泰米尔文|rec_ta_lite_train.yml|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_train.tar) |

* 先把这些模型通过训练或者下载的方式准备好备用。

### 二、准备环境和工具

* 开始之前，需要先准备好Linux，paddle的pip包不支持macOS，而windows部署起来麻烦一些，以下说明以deepin20为范例。
* 安装python3: ```sudo apt-get install python3```
* 安装pip3:

```
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```

* pip安装paddlepaddle(
  2.0的模型需要2.0版): ```sudo pip3 install paddlepaddle==2.0.0rc1 -i https://mirror.baidu.com/pypi/simple```
*

pip安装转换工具附加依赖: ```sudo pip3 install setuptools tqdm opencv-python imgaug pyclipper -i https://mirror.baidu.com/pypi/simple```

* 安装lmdb依赖: ```sudo apt-get install python3-yaml libxml2-dev libxslt1-dev python3-dev```
* pip安装lmdb: ```sudo pip3 install lmdb -i https://mirror.baidu.com/pypi/simple```
* pip安装onnx: ```sudo pip3 install onnx```
* 下载PaddleOCR(用于预训练模型转paddle推理模型):```git clone https://github.com/PaddlePaddle/PaddleOCR.git```
* 安装转换工具库Paddle2ONNX(develop分支,用于paddle推理模型转onnx模型):

```shell
#官方仓库已修复fix_adative_pool
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git checkout develop
sudo python3 setup.py install

#以下为修复mobile rec转换错误的开发版
#修复代码还未同步进官方仓库之前，先使用此版本
#git clone https://github.com/Channingss/paddle2onnx.git
#cd paddle2onnx
#git checkout fix_adative_pool
#sudo python3 setup.py install
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
python3 generate_multi_language_configs.py -l xi
python3 generate_multi_language_configs.py -l pu
python3 generate_multi_language_configs.py -l ru
python3 generate_multi_language_configs.py -l ar
python3 generate_multi_language_configs.py -l hi
python3 generate_multi_language_configs.py -l chinese_cht
python3 generate_multi_language_configs.py -l ug
python3 generate_multi_language_configs.py -l fa
python3 generate_multi_language_configs.py -l ur
python3 generate_multi_language_configs.py -l rs
python3 generate_multi_language_configs.py -l oc
python3 generate_multi_language_configs.py -l mr
python3 generate_multi_language_configs.py -l ne
python3 generate_multi_language_configs.py -l rsc
python3 generate_multi_language_configs.py -l bg
python3 generate_multi_language_configs.py -l uk
python3 generate_multi_language_configs.py -l be
python3 generate_multi_language_configs.py -l te
python3 generate_multi_language_configs.py -l ka
python3 generate_multi_language_configs.py -l ta
```

* 新建文件夹PaddleOCR/inference
* 把预训练模型解压到inference目录，严格按照如下目录结构和文件名放置

```
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

* 把本项目tools文件夹里的export_custom.py复制到PaddleOCR/tools里，把program.py覆盖到PaddleOCR/tools里
* 把本项目tools文件夹里的gen-models.sh复制到PaddleOCR根目录
* 开始转换

```shell
cd PaddleOCR
chmod a+x gen-models.sh
./gen-models.sh
```

* 转换成功后，会在PaddleOCR/inference文件夹里生成31个onnx模型
