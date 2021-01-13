# BaiPiaoOcrOnnx

### Project下载
* 有整合好源码和依赖库的完整工程项目，文件比较大，可到Q群共享内下载，找以Project开头的压缩包文件
* 如果想自己折腾，则请继续阅读本说明

### Demo下载(win、mac、linux)
编译好的demo文件比较大，可以到Q群共享内下载

### 介绍
白富美OCR onnxruntime推理

onnxruntime框架[https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

### 关于模型：
* det模型用于分割文字块，有两种模型(server和mobile)，其中server体积大且较慢但效果好，mobile体积小且较快但效果差点。
* cls模型用于检测文字方向，只有一种模型
* rec模型用于文字识别，虽然有两种模型(server和mobile)，但mobile由于转换失败，目前只有server模型。未来转换工具成熟后可以直接支持mobile模型。

### 模型下载
[模型下载地址](https://github.com/znsoftm/BaiPiaoOCR/tree/main/models)
也可以到Q群共享内下载，下载后解压到项目根目录
```
BaiPiaoOcrOnnx/models
    ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
    ├── ch_ppocr_mobile_v2.0_det_infer.onnx
    ├── ch_ppocr_server_v2.0_det_infer.onnx
    ├── ch_ppocr_server_v2.0_rec_infer.onnx
    └── ppocr_keys_v1.txt
```

### 依赖的第三方库下载
下载opencv和onnxruntime，[下载地址](https://gitee.com/benjaminwan/ocr-lite-onnx/releases/v1.0)
* OpenCv动态库：opencv-(版本号)-sharedLib.7z
* OpenCv静态库：opencv-(版本号)-staticLib.7z
* OnnxRuntime动态库：onnxruntime-(版本号)-sharedLib.7z
* OnnxRuntime静态库：onnxruntime-(版本号)-staticLib.7z
* 可以选择只下载两者的动态库或两者的静态库(要么都是静态库要么都是动态库)，或者4种全部下载
* 把压缩包解压到项目根目录，解压后目录结构
```
BaiPiaoOcrOnnx
    ├── onnxruntime-shared
    ├── onnxruntime-static
    ├── opencv-shared
    ├── opencv-static
```

### 编译环境
1. Windows 10 x64
2. macOS 10.15
3. Linux Ubuntu 1604

### Windows编译说明
#### Windows nmake编译
1. 安装VS2017或VS2019，安装时，至少选中'使用C++的桌面开发'
2. cmake请自行下载&配置，[下载地址](https://cmake.org/download/)
3. 开始菜单打开"x64 Native Tools Command Prompt for VS 2019"或"适用于 VS2017 的 x64 本机工具"，并转到本项目根目录
4. 运行```build.bat```并按照提示输入选项，最后选择'编译成可执行文件'
5. 编译完成后运行```run-test.bat```进行测试
6. 编译JNI动态运行库(可选，可用于java调用)
* 下载jdk-8u221-windows-x64.exe，安装选项默认(确保“源代码”项选中)，安装完成后，打开“系统”属性->高级->环境变量
* 新建“系统变量”，变量名```JAVA_HOME``` ，变量值```C:\Program Files\Java\jdk1.8.0_221``
* 新建“系统变量”，变量名```CLASSPATH``` ，变量值```.;%JAVA_HOME%\lib\dt.jar;%JAVA_HOME%\lib\tools.jar;``
* 编辑“系统变量”Path，Win7在变量值头部添加```%JAVA_HOME%\bin;``` ，win10直接添加一行```%JAVA_HOME%\bin```
* 开始菜单打开"x64 Native Tools Command Prompt for VS 2019"或"适用于 VS2017 的 x64 本机工具"，并转到本项目根目录
* 运行```build.bat```并按照提示输入选项，最后选择'编译成JNI动态库'

#### Windows Visual Studio编译说明
1. VS2017/VS2019，cmake……等安装配置参考上述步骤。
2. 运行generate-vs-project.bat，输入数字选择要生成的visual studio项目解决方案版本。
3. 根据你的编译环境，进入build-xxxx-x86或x64文件夹，打开BaiPiaoOcrOnnx.sln。
4. 在顶部工具栏选择Release，在右边的"解决方案"窗口，右键选中"ALL_BUILD"->生成。要选择Debug，则您必须自行编译Debug版的opencv或onnxruntime。

#### Windows部署说明
1. 编译选项选择第三方库为动态库时，部署的时候记得把dll复制到可执行文件目录。
2. 部署时如果提示缺少"VCRUNTIME140_1.dll"，下载安装适用于 Visual Studio 2015、2017 和 2019 的 Microsoft Visual C++ 可再发行软件包，
   [下载地址](https://support.microsoft.com/zh-cn/help/2977003/the-latest-supported-visual-c-downloads)

### Mac编译说明
1. macOS Catalina 10.15.x，安装Xcode 12.1，并安装Xcode Command Line Tools, 终端运行```xcode-select –install```
2. 自行下载安装HomeBrew，cmake >=3.1[下载地址](https://cmake.org/download/)
3. libomp: ```brew install libomp```
4. 终端打开项目根目录，```./build.sh```并按照提示输入选项，最后选择'编译成可执行文件'
5. 测试：```./run-test.sh```
6. 编译JNI动态运行库(可选，可用于java调用)
* 下载jdk-8u221-macosx-x64.dmg，安装。
* 编辑用户目录下的隐藏文件```.zshrc``` ，添加```export JAVA_HOME=$(/usr/libexec/java_home)```
* 运行```build.sh```并按照提示输入选项，最后选择'编译成JNI动态库'

#### macOS部署说明
opencv或onnxruntime使用动态库时，参考下列方法：
* 把动态库所在路径加入DYLD_LIBRARY_PATH搜索路径
* 把动态库复制或链接到到/usr/lib

### Linux编译说明
1. Ubuntu16.04 LTS 或其它发行版
2. ```sudo apt-get install build-essential```
3. g++>=5，cmake>=3.1[下载地址](https://cmake.org/download/)
4. 终端打开项目根目录，```./build.sh```并按照提示输入选项，最后选择'编译成可执行文件'
5. 测试：```./run-test.sh```
6. 编译JNI动态运行库(可选，可用于java调用)
* 下载jdk-8u221并安装配置
* 运行```build.sh```并按照提示输入选项，最后选择'编译成JNI动态库'

#### Linux部署说明
opencv或onnxruntime使用动态库时，参考下列方法：
* 把动态库所在路径加入LD_LIBRARY_PATH搜索路径
* 把动态库复制或链接到到/usr/lib

### 编译参数说明
build.sh编译参数：
1. ```BAIPIAO_OPENMP=ON```：启用(ON)或禁用(OFF) ON时AngleNet和CrnnNet阶段使用OpenMP并行运算，OFF时单线程运算
2. ```BAIPIAO_LIB=ON```： 启用(ON)或禁用(OFF) ON时编译为jni lib，OFF时编译为可执行文件
3. ```BAIPIAO_STATIC=ON```： 启用(ON)或禁用(OFF) ON时选择opencv和onnxruntime的静态库进行编译，OFF时则选择动态库编译

### 输入参数说明
* 请参考main.h中的命令行参数说明。
* 每个参数有一个短参数名和一个长参数名，用短的或长的均可。
1. ```-d或--models```：模型所在文件夹路径，可以相对路径也可以绝对路径。
2. ```-1或--det```:det模型文件名
3. ```-2或--cls```:cls模型文件名
4. ```-3或--rec```:rec模型文件名
5. ```-4或--keys```:keys.txt文件名
6. ```-i或--image```：目标图片路径，可以相对路径也可以绝对路径。
7. ```-t或--numThread```：线程数量。
8. ```-p或--padding```：图像预处理，在图片外周添加白边，用于提升识别率，文字框没有正确框住所有文字时，增加此值。
9. ```-s或--maxSideLen```：按图片最长边的长度，此值为0代表不缩放，例：1024，如果图片长边大于1024则把图像整体缩小到1024再进行图像分割计算，如果图片长边小于1024则不缩放，如果图片长边小于32，则缩放到32。
6. ```-b或--boxScoreThresh```：文字框置信度门限，文字框没有正确框住所有文字时，减小此值。
7. ```-o或--boxThresh```：请自行试验。
9. ```-u或--unClipRatio```：单个文字框大小倍率，越大时单个文字框越大。此项与图片的大小相关，越大的图片此值应该越大。
10. ```-a或--noAngle```：启用(1)/禁用(0) 文字方向检测，只有图片倒置的情况下(旋转90~270度的图片)，才需要启用文字方向检测。
11. ```-A或--mostAngle```：启用(1)/禁用(0) 角度投票(整张图片以最大可能文字方向来识别)，当禁用文字方向检测时，此项也不起作用。
12. ```-?或--help```：打印命令行帮助。
