It's only a wrapper for cpp version. 

We give it a set of pure C interface APIs for CPP project, so that you can call the APIs in your server or client end APP.


To start, you should open the CMakeLists.txt from the root directory with Visual studio 2019 or CMAKE directly.

About files in this directory:


rapidocr_api.cpp     : the implementation of APIs


rapidocrtest.cpp    : the tester


Supported platform:

1. Windows 32/64 bits

2. Linux  64bits

3. Macos 

4. IOS

5. Android


gcc:  4.8.5 or above

ms vs: vs2015 or higher

Onnxruntime: 1.6

opencv :  3.x

Mainly, it shoud link to library libopencv_imgcodecs 	libopencv_imgproc 	libopencv_core
