### See [Documentation](https://rapidai.github.io/RapidOCRDocs/install_usage/rapidocr_api/usage/)

build时使用宿主的网络
<code>
  docker build -t rapidocr_api:0.1.4 --network host .
</code>

使用宿主上的代理
<code>
  docker build -t rapidocr_api:0.1.4 --network host --build-arg HTTP_PROXY=http://127.0.0.1:8888 --build-arg HTTPS_PROXY=http://127.0.0.1:8888 .
</code>

调试运行：
<code>
docker run --rm -p 9003:9003 --name rapidocr_api rapidocr_api:0.1.4
</code>

运行：
<code>
docker run -d -p 9003:9003 --name rapidocr_api rapidocr_api:0.1.4
</code>
