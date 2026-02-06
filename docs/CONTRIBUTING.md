# RapidOCR Python Contributing Guide

Thanks for your interest in contributing to the RapidOCR Python codebase! This guide explains how to set up your environment, develop, and submit changes for the `python` directory, including running tests and opening pull requests.

## Prerequisites

- Python >= 3.6 (3.8+ recommended)
- Git
- A GitHub account

---

## 1. Clone the repository

Clone the RapidOCR repository to your machine:

```bash
git clone https://github.com/RapidAI/RapidOCR.git
cd RapidOCR
```

If you have network restrictions, use a mirror or proxy; you can also fork the repo to your account first and clone your fork (see “Preparing to submit” below).

---

## 2. Enter the Python directory and set up the environment

```bash
cd python
```

Use a virtual environment to avoid conflicts with the system Python:

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Or using conda
conda create -n rapidocr python=3.10
conda activate rapidocr
```

Install dependencies (editable install is recommended for local development):

```bash
pip install -r requirements.txt
pip install pytest  # required to run tests
# Optional: install the package in editable mode for debugging
pip install -e .
```

For inference backends such as ONNX Runtime, follow the [documentation](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/) to install the corresponding packages (e.g. `rapidocr_onnxruntime`).

---

## 3. Install code formatting and pre-commit hooks

Install pre-commit in your **development environment** and enable Git pre-commit hooks so that code is automatically formatted and checked (e.g. black, autoflake):

```bash
# From the python directory with your venv activated
pip install pre-commit

# Go to the repository root to install Git hooks (.pre-commit-config.yaml is in the root)
cd ..   # if you are in python/, go back to the repo root
pre-commit install
```

After installation, each `git commit` will run the configured checks; if they fail, the commit will be rejected. Fix the reported issues and commit again. You can also run checks manually before committing:

```bash
# From the repository root
pre-commit run --all-files
```

---

## 4. Run unit tests

From the **`python`** directory:

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_input.py -v
pytest tests/test_det_cls_rec.py -v

# Run with coverage (requires pytest-cov)
pytest tests/ -v --cov=rapidocr
```

Make sure the current main branch passes tests in your environment before making changes.

---

## 5. Reproduce the issue or add a new feature

### Reproducing a bug

1. Pick or open an issue on [GitHub Issues](https://github.com/RapidAI/RapidOCR/issues).
2. Reproduce the problem locally using the code under the `python` directory and the issue description.
3. Locate and fix the code in `rapidocr/` or `tests/` until the issue is resolved.

### Adding a new feature

1. (Optional but recommended) Discuss the requirement and approach with maintainers or in an existing issue.
2. Implement the feature under `rapidocr/`, following the existing style (the project uses [black](https://github.com/psf/black) and related tools).
3. Add unit tests for the new feature.

---

## 6. Write the corresponding unit tests

- Place test files under **`python/tests/`** with names like `test_*.py`.
- Use **pytest**. You can refer to existing tests such as `test_input.py`, `test_det_cls_rec.py`, and `test_cli.py`.
- Put test assets (e.g. images) in `tests/test_files/`.
- New tests should:
    - Reliably verify the behavior you changed (bug fix or new feature).
    - Avoid depending on external services not documented in the repo (use mocks or skip when needed).

Example:

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
    # more assertions...
```

---

## 7. Run all unit tests

From the **`python`** directory, run the full test suite again to avoid regressions:

```bash
pytest tests/ -v
```

If some tests are skipped (e.g. missing an inference engine), ensure that the tests you added or changed run and pass in your environment.

---

## 8. Prepare to submit to the repository

### 8.1 Fork the RapidOCR repository to your account

1. Open the [RapidOCR repository](https://github.com/RapidAI/RapidOCR).
2. Click **Fork** to create a fork under your GitHub account (e.g. `https://github.com/YOUR_USERNAME/RapidOCR`).

### 8.2 Commit and push to your fork

If you cloned the upstream repo, add your fork as a remote and push your branch:

```bash
# Run from the repository root (RapidOCR)
git remote add myfork https://github.com/YOUR_USERNAME/RapidOCR.git
# If origin points to upstream, keep it; use myfork for pushing

# Create a branch (one branch per issue or feature is recommended)
git checkout -b fix/xxx   # or feat/xxx

# Stage and commit your changes under python/
git add python/
git status   # confirm only intended files are staged
git commit -m "fix(python): short description"

# Push to your fork
git push myfork fix/xxx
```

**Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification** for commit messages so maintainers can read and generate changelogs easily. Format:

```
<type>[optional scope]: <short description>

[optional body]
[optional footer]
```

Common types:

| Type       | Description                    |
|------------|--------------------------------|
| `feat`     | New feature                    |
| `fix`      | Bug fix                        |
| `docs`     | Documentation changes          |
| `style`    | Code style (no logic change)   |
| `refactor` | Refactoring                    |
| `test`     | Tests                          |
| `chore`    | Build / tooling, etc.          |

Examples: `fix(python): empty result under certain conditions`, `feat(python): support xxx input format`.

### 8.3 Open a Pull Request (PR) to the main repository

1. Open your fork in the browser (e.g. `https://github.com/YOUR_USERNAME/RapidOCR`).
2. After pushing, you will usually see **Compare & pull request**; click it. Otherwise, select your branch under **Branches** and click **New pull request**.
3. Set **base** to `RapidAI/RapidOCR` and branch `main` (or the default branch). Set **head** to your fork and your branch (e.g. `fix/xxx`).
4. Fill in the PR title and description:
   - **Title**: Short summary (e.g. “Fix: xxx in Python”).
   - **Description** should include:
     - Related issue: `Fixes #123` or `Related to #123` if applicable.
     - Reason for the change and what was done.
     - How to verify (e.g. “`pytest tests/ -v` in the python directory passes”).
5. Submit the PR. After review, update your branch locally and push; the PR will update automatically.

---

## Summary

| Step | Description |
|------|-------------|
| 1 | Clone the RapidOCR repository |
| 2 | Go to the `python` directory, set up a venv, and install dependencies and pytest |
| 3 | Install pre-commit (`pip install pre-commit`) and run `pre-commit install` from the repo root |
| 4 | Run unit tests and confirm they pass |
| 5 | Reproduce the issue or implement the new feature |
| 6 | Add or update the corresponding unit tests |
| 7 | Run the full test suite from the `python` directory and confirm it passes |
| 8 | Fork the main repository to your account |
| 9 | Write commits using Conventional Commits and push to your fork |
| 10 | Open a PR from your fork’s branch to the main repository’s `main` |

---

## Notes

- **Code style**: The project uses [black](https://github.com/psf/black), autoflake, etc. Pre-commit runs these on commit. You can also run `pre-commit run --all-files` from the repo root.
- **Documentation**: See the [RapidOCR docs](https://rapidai.github.io/RapidOCRDocs/) for installation and usage.
- **Issues and discussion**: Report bugs and suggest features via [GitHub Issues](https://github.com/RapidAI/RapidOCR/issues).

Thank you for contributing!
