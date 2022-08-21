import sys
from pathlib import Path
import yaml

root_dir = Path(__file__).parent.parent

def get_resource_path(name: str):
    """依次检查资源文件的多个可能路径, 返回首个存在的路径"""
    for path in [
        Path(name),
        Path(root_dir, name),
        Path(sys.argv[0]).parent / name,
    ]:
        if path.exists():
            return path
    raise FileNotFoundError(name)

conf = yaml.safe_load(get_resource_path('config.yaml').read_text())
