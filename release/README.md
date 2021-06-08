下载多语言模型文件：

```
https://github.com/RapidOCR/RapidOCR/releases/download/V1.0/rapid-model.tgz

```

## SDK包中的目录说明

x86  32位二进制制文件

libx86  32位链接库文件

x64  64位二进制制文件

libx64  64位链接库文件

include 头文件目录



##  测试程序与dll/so位于同一个目录中，使用方法

/path/to/test  /path/to/model/dir/  /path/to/img/file


如

c:\test\x64\rapidocrtester.exe  e:\rapidocr\models\  e:\test\test.jpg
