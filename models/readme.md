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
1. [mobile det train](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)
2. [mobile cls train](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar)
3. [mobile rec train](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar)
4. [server det train](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)
5. [server rec train](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar)
* 先把这些模型通过训练或者下载的方式准备好备用。

### 二、准备环境和工具
* 开始之前，需要先准备好Linux，paddle的pip包不支持macOS，而windows部署起来麻烦一些，以下说明以deepin20为范例。
* 安装python3: ```sudo apt-get install python3```
* 安装pip3: 
```
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```
* pip安装paddlepaddle(2.0的模型需要2.0版): ```sudo pip3 install paddlepaddle==2.0.0rc1 -i https://mirror.baidu.com/pypi/simple```
* pip安装转换工具附加依赖: ```sudo pip3 install setuptools tqdm opencv-python imgaug pyclipper -i https://mirror.baidu.com/pypi/simple```
* 安装lmdb依赖: ```sudo apt-get install python3-yaml libxml2-dev libxslt1-dev python3-dev```
* pip安装lmdb: ```sudo pip3 install lmdb -i https://mirror.baidu.com/pypi/simple```
* pip安装onnx: ```sudo pip3 install onnx```
* 下载PaddleOCR(用于预训练模型转paddle推理模型):```git clone https://github.com/PaddlePaddle/PaddleOCR.git```  
* 安装转换工具库Paddle2ONNX(develop分支,用于paddle推理模型转onnx模型):
```shell
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
sudo python3 setup.py install
```

### 三、开始转换
* 接下来的操作都在PaddleOCR文件夹里进行
* PaddleOCR切换到2.0分支(因为预训练模型也是v2.0格式的)
```shell
cd PaddleOCR
git checkout release/2.0-rc1-0
```
* 新建文件夹PaddleOCR/models
* 把预训练模型解压到models目录，严格按照如下目录结构和文件名放置
```
PaddleOCR/models
    ├── ch_ppocr_mobile_v2.0_cls_train
    │   ├── best_accuracy.pdopt
    │   ├── best_accuracy.pdparams
    │   ├── best_accuracy.states
    │   └── train.log
    ├── ch_ppocr_mobile_v2.0_det_train
    │   ├── best_accuracy.pdopt
    │   ├── best_accuracy.pdparams
    │   ├── best_accuracy.states
    │   └── train.log
    ├── ch_ppocr_server_v2.0_det_train
    │   ├── best_accuracy.pdopt
    │   ├── best_accuracy.pdparams
    │   ├── best_accuracy.states
    │   └── train.log
    ├── ch_ppocr_server_v2.0_rec_train
    │   ├── best_accuracy.pdopt
    │   ├── best_accuracy.pdparams
    │   ├── best_accuracy.states
    │   └── train.log
```
* 把本项目tools文件夹里的export_custom.py复制到PaddleOCR/tools里
* 把本项目tools文件夹里的gen-models.sh复制到PaddleOCR根目录
* 开始转换
```shell
cd PaddleOCR
chmod a+x gen-models.sh
./gen-models.sh
```
* 因为Paddle2ONNX还不支持mobile的rec模型转换，所以脚本里注释掉了相关内容
* 转换成功后，会在PaddleOCR/models文件夹里生成4个onnx模型