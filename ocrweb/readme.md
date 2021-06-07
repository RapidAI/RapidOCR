
#### 目录结构
```text
.
├── main.py
├── readme.md
├── resources
│   ├── bpocr.py
│   ├── ch_ppocr_mobile_v2_cls
│   ├── ch_ppocr_mobile_v2_det
│   ├── ch_ppocr_mobile_v2_rec
│   ├── __init__.py
│   ├── models
│   └── __pycache__
├── static
│   ├── css
│   ├── images
│   └── js
├── task.py
└── templates
    └── index.html
```

#### Redis队列web界面使用步骤
- 安装redis,并开启redis服务
- `rq worker` 启动工作进程，处理实际任务
- `python main_redis.py` 开启前端接受数据界面