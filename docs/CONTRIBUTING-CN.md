# RapidOCR Python 贡献指南

感谢你对 RapidOCR Python 部分的关注！本文档说明如何参与 Python 目录下的代码开发与贡献，包括环境准备、开发流程和提交流程。

## 前置要求

- Python >= 3.6（推荐 3.8+）
- Git
- 已注册的 GitHub 账号

---

## 一、克隆源码

从 RapidOCR 主仓库克隆项目到本地：

```bash
git clone https://github.com/RapidAI/RapidOCR.git
cd RapidOCR
```

若网络受限，可使用镜像或代理；也可先 fork 到个人账号后再克隆（见后文「准备提交」部分）。

---

## 二、进入 Python 目录并配置环境

```bash
cd python
```

建议使用虚拟环境，避免与系统 Python 冲突：

```bash
# 使用 venv
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 或使用 conda
conda create -n rapidocr python=3.10
conda activate rapidocr
```

安装依赖（开发时建议可编辑安装以便本地修改生效）：

```bash
pip install -r requirements.txt
pip install pytest  # 运行单元测试需要
# 可选：以可编辑模式安装当前包，便于调试
pip install -e .
```

如需使用 ONNX Runtime 等推理后端，请按 [文档](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/) 安装对应依赖（如 `rapidocr_onnxruntime` 等）。

---

## 三、安装代码格式化与 pre-commit 钩子

在 **开发者环境** 下安装 pre-commit，并启用 Git 提交前钩子，以便自动做代码格式检查与整理（如 black、autoflake 等）：

```bash
# 在 python 目录下、已激活的虚拟环境中安装
pip install pre-commit

# 到仓库根目录 RapidOCR 下安装 Git 钩子（.pre-commit-config.yaml 在根目录）
cd ..   # 若当前在 python 目录，先回到仓库根目录
pre-commit install
```

安装成功后，每次执行 `git commit` 时会自动运行配置好的格式化工具；若检查未通过，提交会被拒绝，请根据提示修改后再次提交。也可在提交前手动跑一遍：

```bash
# 在仓库根目录执行
pre-commit run --all-files
```

---

## 四、运行单元测试

在 **`python` 目录下** 执行：

```bash
# 运行全部测试
pytest tests/ -v

# 仅运行部分测试文件
pytest tests/test_input.py -v
pytest tests/test_det_cls_rec.py -v

# 查看测试覆盖率（需先安装 pytest-cov）
pytest tests/ -v --cov=rapidocr
```

确认当前主分支在你本机环境下测试通过，再进行修改。

---

## 五、复现问题 / 增加新功能

### 复现 Bug

1. 在 [Issues](https://github.com/RapidAI/RapidOCR/issues) 中选定或创建对应 issue。
2. 根据 issue 描述与报错信息，在本地用 `python` 目录下的代码复现问题。
3. 在 `rapidocr/` 或 `tests/` 下定位并修改代码，直到问题消失。

### 增加新功能

1. 与 maintainer 或现有 issue 讨论需求与实现方式（可选但推荐）。
2. 在 `rapidocr/` 下实现新逻辑，保持与现有代码风格一致（项目使用 [black](https://github.com/psf/black) 等规范）。
3. 新功能应有对应单元测试覆盖。

---

## 六、编写对应单元测试

- 测试文件放在 **`python/tests/`** 下，命名建议 `test_*.py`。
- 使用 **pytest** 编写用例，可参考现有 `test_input.py`、`test_det_cls_rec.py`、`test_cli.py` 等。
- 测试用图片等资源放在 `tests/test_files/`。
- 新增测试应：
    - 能稳定复现你要验证的行为（Bug 修复或新功能）；
    - 不依赖未在仓库或文档中说明的外部服务（必要时用 mock 或跳过）。

示例：

```python
# tests/test_xxx.py
import pytest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
tests_dir = root_dir / "tests" / "test_files"

@pytest.fixture()
def engine():
    from rapidocr import RapidOCR
    return RapidOCR()

def test_your_new_feature(engine):
    img_path = tests_dir / "ch_en_num.jpg"
    result = engine(img_path)
    assert result is not None
    # 更多断言...
```

---

## 七、运行所有单元测试

在 **`python` 目录下** 再次全量跑测，确保无回归：

```bash
pytest tests/ -v
```

若有测试被跳过（如缺少某推理引擎），请确认你修改或新增的测试在现有环境下已执行并通过。

---

## 八、准备提交到仓库

### 8.1 Fork RapidOCR 主仓库到个人账号

1. 打开 [RapidOCR 主仓库](https://github.com/RapidAI/RapidOCR)。
2. 点击右上角 **Fork**，将仓库 fork 到你自己的 GitHub 账号下（例如 `https://github.com/你的用户名/RapidOCR`）。

### 8.2 将代码提交到个人 Fork

若最初是克隆的主仓库，需要把远程改为你的 fork，并推送到 fork：

```bash
# 在项目根目录 RapidOCR 下执行
git remote add myfork https://github.com/你的用户名/RapidOCR.git
# 若已有 origin 且就是主仓库，可保留；推送时用 myfork

# 创建分支（推荐为每个 issue/功能单独分支）
git checkout -b fix/xxx   # 或 feat/xxx

# 添加并提交你在 python 目录下的修改
git add python/
git status   # 确认只提交预期文件
git commit -m "fix(python): 简短描述"

# 推送到你的 fork
git push myfork fix/xxx
```

**请按约定式提交规范（Conventional Commits）书写 commit 信息** ，便于维护者阅读与自动生成 Changelog。格式为：

```text
<类型>[可选范围]: <简短描述>

[可选正文]
[可选脚注]
```

常用类型示例：

| 类型     | 说明         |
|----------|--------------|
| `feat`   | 新功能       |
| `fix`    | Bug 修复     |
| `docs`   | 文档变更     |
| `style`  | 代码格式（不影响逻辑） |
| `refactor` | 重构       |
| `test`   | 测试相关     |
| `chore`  | 构建/工具等  |

示例：`fix(python): 修复某条件下识别结果为空`、`feat(python): 支持 xxx 输入格式`。

### 8.3 向 RapidOCR 主仓库提交 Pull Request（PR）

1. 打开你 fork 后的仓库页面（如 `https://github.com/你的用户名/RapidOCR`）。
2. 若刚推送分支，页面上通常会出现 **Compare & pull request**，点击即可；否则在 **Branches** 里选择你刚推送的分支，再点 **New pull request** 。
3. 确认 **base 仓库** 为 `RapidAI/RapidOCR`、**base 分支** 为 `main`（或仓库默认主分支），**head 仓库** 为你的 fork、**head 分支** 为你的分支（如 `fix/xxx`）。
4. 填写 PR 标题和说明：
   - 标题：简要概括修改内容（如「Fix: 修复 Python 下 xxx 问题」）。
   - 说明中建议包含：
     - 对应 Issue 编号（若有）：`Fixes #123` 或 `Related to #123`。
     - 修改原因与主要改动。
     - 如何验证：例如「在 python 目录下执行 `pytest tests/ -v` 通过」。
5. 提交 PR，等待 maintainer 审查；根据反馈再在本地修改并推送同一分支，PR 会自动更新。

---

## 流程小结

| 步骤 | 说明 |
|------|------|
| 1 | 克隆 RapidOCR 源码 |
| 2 | 进入 `python` 目录，配置虚拟环境并安装依赖与 pytest |
| 3 | 安装 pre-commit（`pip install pre-commit`），在仓库根目录执行 `pre-commit install` 安装 Git 钩子 |
| 4 | 运行单元测试，确认基线通过 |
| 5 | 复现问题或实现新功能 |
| 6 | 编写/补充对应单元测试 |
| 7 | 在 `python` 目录下运行全部测试并确认通过 |
| 8 | Fork 主仓库到个人账号 |
| 9 | 按约定式提交规范编写 commit，将修改提交并推送到个人 Fork 的对应分支 |
| 10 | 在主仓库创建 PR，从个人 Fork 分支指向主仓库 main |

---

## 其他说明

- **代码风格** ：项目采用 [black](https://github.com/psf/black)、autoflake 等规范，已通过 pre-commit 钩子在提交时自动检查；也可在仓库根目录执行 `pre-commit run --all-files` 手动跑一遍。
- **文档** ：更多安装与使用说明见 [RapidOCR 文档](https://rapidai.github.io/RapidOCRDocs/)。
- **问题与讨论** ：Bug 与功能建议可通过 [GitHub Issues](https://github.com/RapidAI/RapidOCR/issues) 反馈。

再次感谢你的贡献！
