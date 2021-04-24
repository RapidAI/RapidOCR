# C++项目

## 简介
本目录存放C++相关demo项目
1. BaiPiaoOcrOnnx项目: 模型源自paddleOCR，转为onnx格式后，使用onnxruntime框架进行推理。



怎样添加provider 以支持gpu等，下面以cuda为例

  OrtEnv* env;
  OrtInitialize(ORT_LOGGING_LEVEL_WARNING, "test", &env)
  OrtSessionOptions* session_option = OrtCreateSessionOptions();
  OrtProviderFactoryInterface** factory;
  OrtCreateCUDAExecutionProviderFactory(0, &factory);
  OrtSessionOptionsAppendExecutionProvider(session_option, factory);
  OrtReleaseObject(factory);
  OrtCreateSession(env, model_path, session_option, &session);

`
