FROM python:3.10.11-slim-buster

ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

RUN pip install --no-cache-dir rapidocr_api -i https://mirrors.aliyun.com/pypi/simple 

RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless -i https://mirrors.aliyun.com/pypi/simple

EXPOSE 9003

# 升级后可用
# CMD ["bash", "-c", "rapidocr_api -ip 0.0.0.0 -p 9003 -workers 2"]
CMD ["rapidocr_api"]

