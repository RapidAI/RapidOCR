### See [Documentation](https://rapidai.github.io/RapidOCRDocs/install_usage/rapidocr_api/usage/)

build时使用宿主的网络
```
  docker build -t rapidocr_api:0.1.4 --network host .
```

使用宿主上的代理
```
  docker build -t rapidocr_api:0.1.4 --network host --build-arg HTTP_PROXY=http://127.0.0.1:8888 --build-arg HTTPS_PROXY=http://127.0.0.1:8888 .
```

调试运行：
```
docker run --rm -p 9003:9003 --name rapidocr_api rapidocr_api:0.1.4
```

运行：
```
docker run -d -p 9003:9003 --name rapidocr_api rapidocr_api:0.1.4
```
