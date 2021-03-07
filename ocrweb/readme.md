
#### 目录结构
```text
.
├── main.py
├── main_redis.py
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
├── task.py  # 单独开启rq worker时使用
└── templates
    ├── index.html
    └── index_redis.html  # 带有redis队列web界面
```

#### Redis队列web界面使用步骤
- 安装redis,并开启redis服务
- `rq worker` 启动工作进程，处理实际任务
- `python main_redis.py` 开启前端接受数据界面