#### SDK包中的目录说明

### C++项目使用onnxruntime和ncnn两种推理框架，实现了2种版本
### onnxruntime版：[RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx)
### ncnn版：[RapidOcrNcnn](https://github.com/RapidAI/RapidOcrNcnn)
### 以下内容停止更新

```text
x86  32位二进制制文件
libx86  32位链接库文件
x64  64位二进制制文件
libx64  64位链接库文件
include 头文件目录
```

####  测试程序与dll/so位于同一个目录中，使用方法
```shell
/path/to/test  /path/to/model/dir/  /path/to/img/file

# example
c:\test\x64\rapidocrtester.exe  e:\rapidocr\models\  e:\test\test.jpg
```


#### 下载多语言模型文件 [link](https://github.com/RapidAI/RapidOCR/releases/download/V1.0/rapid-model.tgz)
