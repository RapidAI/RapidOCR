FROM python:3.10.11-slim-buster

ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 安装vim，如果不需要临时修改容器文件，此步骤可以删
RUN apt-get update && \
    apt-get install -y --no-install-recommends vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    pip install --no-cache-dir rapidocr_api pillow rapidocr-onnxruntime==1.3.25 -i https://mirrors.aliyun.com/pypi/simple; \
    pip uninstall -y opencv-python; \
    pip install --no-cache-dir opencv-python-headless -i https://mirrors.aliyun.com/pypi/simple

EXPOSE 9003

CMD ["bash", "-c", "rapidocr_api -ip 0.0.0.0 -p 9003 -workers 2"]
