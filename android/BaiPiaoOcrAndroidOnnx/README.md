# BaiPiaoOcrAndroidOnnx

### Project下载

* 有整合好源码和依赖库的完整工程项目，文件比较大，可到Q群共享内下载，找以Project开头的压缩包文件
* 如果想自己折腾，则请继续阅读本说明

### Demo APK下载

编译好的demo文件比较大，可以到Q群共享内下载

### 介绍

白富美OCR onnxruntime推理 for Android

onnxruntime框架[https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

### 关于模型：

* det模型用于分割文字块，有两种模型(server和mobile)，其中server体积大且较慢但效果好，mobile体积小且较快但效果差点。
* cls模型用于检测文字方向，只有一种模型
* rec模型用于文字识别，有两种模型(server和mobile)，其中server体积大且较慢但效果好，mobile体积小且较快但效果差点。

### 模型下载

* [模型下载地址](https://github.com/znsoftm/BaiPiaoOCR/tree/main/models)
也可以到Q群共享内下载，下载后解压到如下目录。 目前det模型有2种，如果想让apk体积小，那么就选mobile模型即可
```
BaiPiaoOcrAndroidOnnx/OcrLibrary/src/main/assets
    ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
    ├── ch_ppocr_mobile_v2.0_det_infer.onnx det二选一
    ├── ch_ppocr_server_v2.0_det_infer.onnx det二选一
    ├── ch_ppocr_mobile_v2.0_rec_infer.onnx rec二选一
    ├── ch_ppocr_server_v2.0_rec_infer.onnx rec二选一
    └── ppocr_keys_v1.txt
```
* 代码中配置使用哪个模型
BaiPiaoOcrAndroidOnnx/OcrLibrary/src/main/java/com/benjaminwan/ocrlibrary/OcrEngine.kt，在init方法中配置：
```kotlin
val ret = init(
            context.assets, numThread,
            "ch_ppocr_mobile_v2.0_det_infer.onnx",
            "ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "ch_ppocr_mobile_v2.0_rec_infer.onnx",
            "ppocr_keys_v1.txt"
        )
```

### 总体说明

1. 封装为独立的Library。
2. Native层以C++编写。
3. demo app以Kotlin-JVM编写。
4. Android版与其它版本不同，包含了几个应用场景，包括相册识别、摄像头识别、手机IMEI号识别、摄像头身份证识别这几个功能页面。
5. 自己编译的opencv 3.4.10，并精简了不需要的模块，减小apk体积。
6. onnxruntime动态库为自己编译的1.6.0版

### 编译说明

1. AndroidStudio 4.1.2或以上
2. NDK
3. cmake 3.4.1或以上
4. 下载opencv-3.4.10-android-sdk-static-lite.7z，[下载地址](https://gitee.com/benjaminwan/ocr-lite-android-onnx/releases/v1.0.0.20201022)
解压后目录结构为

```
BaiPiaoOcrAndroidOnnx/OcrLibrary/src/sdk
    └── native
        ├── 3rdparty
        ├── jni
        └── staticlibs
```

5. 下载onnxruntime-1.6.0-android.7z，[下载地址](https://gitee.com/benjaminwan/ocr-lite-android-onnx/releases/v1.0.0.20201022)

```
BaiPiaoOcrAndroidOnnx/OcrLibrary/src/main/onnx
├── ONNXConfig.cmake
├── arm64-v8a
│   └── libonnxruntime.so
├── armeabi-v7a
│   └── libonnxruntime.so
├── include
├── x86
│   └── libonnxruntime.so
└── x86_64
    └── libonnxruntime.so
```

6. 如果想让apk体积进一步缩小，可以修改BaiPiaoOcrAndroidOnnx/OcrLibrary/build.gradle
注释掉x86_64和x86，这两个一般用于模拟器。
```groovy
externalNativeBuild {
    cmake {
        abiFilters 'armeabi-v7a', 'arm64-v8a'//, 'x86_64', 'x86'
    }
}
```

### 输入参数说明

请参考[C++项目](https://github.com/znsoftm/BaiPiaoOCR/tree/main/cpp/BaiPiaoOcrOnnx)

### 删除缓存，重新编译

删除项目根目录下的如下文件夹

```
.idea
build
app/build
OcrLibrary/.cxx
OcrLibrary/build
```

### 编译Release包

使用命令编译```./gradlew assembleRelease```
