docker build -t rapidocr_api --network host .
docker run -itd --restart=always --name rapidocr_api -p 9005:9005 rapidocr_api