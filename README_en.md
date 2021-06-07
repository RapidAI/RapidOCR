# RapidOCR

### Note:
- For PC (political correctness ), we changed our name from BaiPiaoOCR to RapidOCR
- Special thanks to Channingss@baidu (the owner of project Paddle2Onnx)

### Directory structure
<details>
    <summary>click to expand</summary>

    ```text
    RapidOCR
            |
            |--android     安卓工程目录
            |
            |--api4cpp      c语言跨平台接口库源码目录，直接用根下的CMakelists.txt 编译
            |
            |--build        编译脚本
            |
            |--cpp          基于c++的工程项目文件夹
            |
            |--datasets     额外的训练数据集，百度网盘下载地址
            |
            |--dotnet       .Net程序目录
            |
            |--include      编译c语言接口库时的头文件目录
            |
            |--ios          苹果手机平台工程目录
            |
            |--images        测试用图片，两张典型的测试图，一张是自然场景，另一个为长文本
            |
            |--jvm          基于java的工程目录
            |
            |--lib          编译用库文件目录，用于编译c语言接口库用，默认并不上传二进制文件
            |
            |--models       放置可使用的模型文件下载信息，基于百度网盘
            |
            |--python       python推理代码目录
            |
            |--test_imgs    一些演示用的图片，不是测试集
            |
            |--tools        一些转换脚本之类
            |
            |--training     训练自己的模型使用的脚本或程序。
    ```
</details>


### **Copyright announcement:**

- If you use or reference code or code snippet from the repository, please add our url https://github.com/RapidOCR/RapidOCR  in your prodouct derived from the repository.

### **Author:**
- [benjaminwan](https://github.com/benjaminwan)
- [znsoftm](https://github.com/znsoftm)

All contributors is in the contributor list on the right side of this page.

### **Description**
RapidOCR means an  OCR Engine who is fair-skinned, rich and beautiful from Baidu, it's based on PaddleOCR & OnnxRuntime.

**Note**： The project is derived from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

We create the project to transfer PaddleOCR's model into the version of Onnx model to deploy the model on almost every devices, such as x86 PC, Android & IOS.

We are using [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) to transform it to Onnx format.

You can visualize the model by [netron](https://github.com/lutzroeder/netron/)

Welcome to our Group on **QQ: 887298230**
or visit [this link](https://jq.qq.com/?_wv=1027&k=P9b3olx6) to join us.

### Download OCR models
- [Extract Code：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)

### Demonstration with C++/JVM
![avatar](./test_imgs/test_cpp.png)

### Demonstration with .Net

![avatar](./test_imgs/test_cs.png)

### LICENSE:
- Apache License 2.0