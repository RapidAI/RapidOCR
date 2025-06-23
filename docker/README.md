# RapidOCR API Docker

## Quickstart

```bash
docker pull qingchen0607/rapid-ocr-api:v20250619
docker run -itd --restart=always --name rapidocr_api -p 9005:9005 qingchen0607/rapid-ocr-api:v20250619 

# http://<ip>:9005/docs
```

## Local Build

```shell
cd docker

#chmod +x docker_build&run.sh docker_stop&clean.sh

# build image and run
./docker_build&run.sh

# stop and rm image
./docker_stop&clean.sh
```

