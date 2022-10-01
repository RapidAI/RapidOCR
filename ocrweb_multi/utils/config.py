import sys
from pathlib import Path
import yaml

root_dir = Path(__file__).parent.parent


def get_resource_path(name: str):
    """依次检查资源文件的多个可能路径, 返回首个存在的路径"""
    for path in [
        # wrapper.exe 所在目录
        Path(root_dir.parent, name),
        # main.exe 所在目录 / main.py 所在目录
        Path(root_dir, name),
        # main.exe 所在目录
        Path(sys.argv[0]).parent / name,
        # 工作目录
        Path(name),
    ]:
        if path.exists():
            print('Loaded:', path)
            return path
    raise FileNotFoundError(name)


conf = yaml.safe_load(get_resource_path('config.yaml').read_text())
