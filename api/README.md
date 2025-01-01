### See [Documentation](https://rapidai.github.io/RapidOCRDocs/install_usage/rapidocr_api/usage/)

### Dockerfile简单用法：
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
docker run --rm -p 9003:9003 --name rapidocr_api -e TZ=Asia/Shanghai rapidocr_api:0.1.4
```

运行：
```
docker run -d -p 9003:9003 --name rapidocr_api -e TZ=Asia/Shanghai rapidocr_api:0.1.4
```

接口web界面：
```
http://<ip>:9003/docs
```

### Docker 临时修改并验证的方法：
```
docker run -p 9003:9003 --name rapidocr_api -e TZ=Asia/Shanghai rapidocr_api:0.1.4
```
进入container修改python源文件，Dockerfile最好加上apt-get install vim安装
```
docker exec -it rapidocr_api /bin/bash
cd /usr/local/lib/python3.10/site-packages/rapidocr_api
...
# 修改参数文件
vi /usr/local/lib/python3.10/site-packages/rapidocr_onnxruntime/config.yaml
# 改好后exit退出
```
重启container
```
docker restart rapidocr_api
```
查看日志：
```
docker logs -f rapidocr_api
```
