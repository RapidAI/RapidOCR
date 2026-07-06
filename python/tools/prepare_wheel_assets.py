# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import hashlib
import sys
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

# Register lightweight package stubs so that Python can resolve
# ``rapidocr.utils.*`` without executing ``rapidocr/__init__.py``,
# which pulls in cv2, numpy, and other heavy runtime dependencies
# that this script does not need.
_pkg = types.ModuleType("rapidocr")
_pkg.__path__ = [str(PYTHON_ROOT / "rapidocr")]
sys.modules.setdefault("rapidocr", _pkg)

_utils_pkg = types.ModuleType("rapidocr.utils")
_utils_pkg.__path__ = [str(PYTHON_ROOT / "rapidocr" / "utils")]
sys.modules.setdefault("rapidocr.utils", _utils_pkg)

from rapidocr.utils.model_resolver import normalize_lang, resolve_model_key
from rapidocr.utils.typings import ModelType, OCRVersion, TaskType

DEFAULT_CONFIG_YAML = PYTHON_ROOT / "rapidocr" / "config.yaml"
DEFAULT_MANIFEST_IN = PYTHON_ROOT / "MANIFEST.in"
DEFAULT_MODEL_DIR = PYTHON_ROOT / "rapidocr" / "models"
DEFAULT_MODELS_YAML = PYTHON_ROOT / "rapidocr" / "default_models.yaml"
CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT = 60
TASK_SECTIONS = ("Det", "Cls", "Rec")


@dataclass(frozen=True)
class ModelSpec:
    engine: str
    ocr_version: str
    task: str
    lang: str
    model_type: str


@dataclass(frozen=True)
class WheelAsset:
    relative_path: Path
    url: str
    sha256: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare model assets that should be bundled into the RapidOCR wheel."
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=DEFAULT_CONFIG_YAML,
        help=f"RapidOCR default config used to select wheel models. Default: {DEFAULT_CONFIG_YAML}",
    )
    parser.add_argument(
        "--models-yaml",
        type=Path,
        default=DEFAULT_MODELS_YAML,
        help=f"Model registry YAML used as the source of URLs and checksums. Default: {DEFAULT_MODELS_YAML}",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory used to store wheel model assets. Default: {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--manifest-in",
        type=Path,
        default=DEFAULT_MANIFEST_IN,
        help=f"MANIFEST.in file to write with exact wheel asset includes. Default: {DEFAULT_MANIFEST_IN}",
    )
    parser.add_argument(
        "--skip-manifest-in",
        action="store_true",
        help="Do not write MANIFEST.in after preparing assets.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that required assets exist and match their checksums.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download assets even when an existing file passes checksum validation.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required to read RapidOCR YAML config files. "
            "Install it with `python -m pip install PyYAML` or "
            "`python -m pip install -r requirements.txt`."
        ) from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML mapping: {path}")

    return data


def resolve_model_specs(config: dict) -> List[ModelSpec]:
    global_cfg = config.get("Global", {})
    specs = []

    for section in TASK_SECTIONS:
        task_cfg = config.get(section)
        if not isinstance(task_cfg, dict):
            raise RuntimeError(f"Missing config section: {section}")

        task = _required_str(task_cfg, "task_type", section)
        if not global_cfg.get(f"use_{task}", True):
            continue

        specs.append(
            ModelSpec(
                engine=_required_str(task_cfg, "engine_type", section),
                ocr_version=_required_str(task_cfg, "ocr_version", section),
                task=task,
                lang=_required_str(task_cfg, "lang_type", section),
                model_type=_required_str(task_cfg, "model_type", section),
            )
        )

    if not specs:
        raise RuntimeError("No enabled OCR tasks found in config.")

    return specs


def resolve_assets(specs: Iterable[ModelSpec], registry: dict) -> List[WheelAsset]:
    assets = []
    for spec in specs:
        model_info = select_model_info(spec, registry)
        assets.extend(model_info_to_assets(spec, model_info))

    return _dedupe_assets(assets)


def select_model_info(spec: ModelSpec, registry: dict) -> dict:
    try:
        task_models = registry[spec.engine][spec.ocr_version][spec.task]
    except KeyError as e:
        raise RuntimeError(
            "Missing model registry section: "
            f"{spec.engine}.{spec.ocr_version}.{spec.task}"
        ) from e

    model_key = _resolve_model_key(spec)
    if model_key is not None:
        model_info = task_models.get(model_key)
        if not isinstance(model_info, dict):
            raise RuntimeError(
                "No model registry entry matches routed default config: "
                f"{spec.engine}.{spec.ocr_version}.{spec.task} "
                f"lang={spec.lang} model_type={spec.model_type} "
                f"model_key={model_key}"
            )
        return model_info

    lang = normalize_lang(spec.lang)
    if spec.model_type == ModelType.SERVER.value:
        for model_name, model_info in task_models.items():
            if model_name.startswith(lang) and spec.model_type in model_name:
                if not isinstance(model_info, dict):
                    raise RuntimeError(f"Invalid model registry entry: {model_name}")
                return model_info

    for model_name, model_info in task_models.items():
        if model_name.startswith(lang) and spec.model_type in model_name:
            if not isinstance(model_info, dict):
                raise RuntimeError(f"Invalid model registry entry: {model_name}")
            return model_info

    raise RuntimeError(
        "No model registry entry matches default config: "
        f"{spec.engine}.{spec.ocr_version}.{spec.task} "
        f"lang={spec.lang} model_type={spec.model_type}"
    )


def _resolve_model_key(spec: ModelSpec) -> Optional[str]:
    try:
        task_type = TaskType(spec.task)
        ocr_version = OCRVersion(spec.ocr_version)
        model_type = ModelType(spec.model_type)
    except ValueError:
        return None

    return resolve_model_key(task_type, ocr_version, spec.lang, model_type)


def model_info_to_assets(spec: ModelSpec, model_info: dict) -> List[WheelAsset]:
    model_url = _required_str(model_info, "model_dir", "model registry")

    if spec.engine == "paddle":
        assets = [
            WheelAsset(
                relative_path=Path(Path(model_url).name) / filename,
                url=f"{model_url.rstrip('/')}/{filename}",
                sha256=sha256,
            )
            for filename, sha256 in model_info.items()
            if filename not in ("model_dir", "dict_url")
        ]
    else:
        assets = [
            WheelAsset(
                relative_path=Path(Path(model_url).name),
                url=model_url,
                sha256=model_info.get("SHA256"),
            )
        ]

    dict_url = model_info.get("dict_url")
    if spec.task == "rec" and dict_url:
        assets.append(
            WheelAsset(
                relative_path=Path(Path(dict_url).name),
                url=dict_url,
                sha256=None,
            )
        )

    return assets


def prepare_assets(
    assets: Iterable[WheelAsset], model_dir: Path, check_only: bool, force: bool
) -> int:
    model_dir = model_dir.expanduser().resolve()
    missing_or_invalid = []

    for asset in assets:
        save_path = model_dir / asset.relative_path
        if not force and validate_asset(save_path, asset.sha256):
            print(f"OK {save_path}")
            continue

        if check_only:
            status = "missing" if not save_path.exists() else "invalid checksum"
            print(f"FAIL {save_path} ({status})")
            missing_or_invalid.append(save_path)
            continue

        download_asset(asset, save_path)

    if missing_or_invalid:
        print(
            "\nRequired wheel assets are missing or invalid. "
            "Run without --check to download them.",
            file=sys.stderr,
        )
        return 1

    print("\nWheel assets are ready.")
    return 0


def write_manifest_in(manifest_path: Path, assets: Iterable[WheelAsset]) -> None:
    manifest_path = manifest_path.expanduser().resolve()
    lines = [
        "# Generated by tools/prepare_wheel_assets.py.",
        "# Re-run that script after changing rapidocr/config.yaml defaults.",
    ]
    for asset in assets:
        package_path = Path("rapidocr") / "models" / asset.relative_path
        lines.append(f"include {package_path.as_posix()}")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_asset(path: Path, expected_sha256: Optional[str]) -> bool:
    if not path.is_file():
        return False
    if expected_sha256 is None:
        return True
    return file_sha256(path) == expected_sha256


def download_asset(asset: WheelAsset, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(f"{save_path.suffix}.part")

    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "RapidOCR wheel asset preparer"},
    )
    print(f"Downloading {asset.relative_path.as_posix()}")
    print(f"  from {asset.url}")
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        total_size = _get_content_length(response.headers.get("Content-Length"))
        downloaded = 0
        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                _print_progress(downloaded, total_size)

    if total_size is not None:
        print()

    if asset.sha256 is not None:
        actual_sha256 = file_sha256(tmp_path)
        if actual_sha256 != asset.sha256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {asset.relative_path}: "
                f"expected {asset.sha256}, got {actual_sha256}"
            )

    tmp_path.replace(save_path)
    print(f"Saved {save_path}")


def _dedupe_assets(assets: Iterable[WheelAsset]) -> List[WheelAsset]:
    deduped = {}
    for asset in assets:
        existing = deduped.get(asset.relative_path)
        if existing and existing != asset:
            raise RuntimeError(
                f"Conflicting wheel asset definitions: {asset.relative_path}"
            )
        deduped[asset.relative_path] = asset
    return list(deduped.values())


def _get_content_length(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _print_progress(downloaded: int, total_size: Optional[int]) -> None:
    if not total_size:
         return
    percent = downloaded / total_size * 100
    print(
        f"\r  {downloaded / 1024 / 1024:.1f}MB / "
        f"{total_size / 1024 / 1024:.1f}MB ({percent:.1f}%)",
        end="",
    )


def _required_str(data: dict, key: str, section: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Missing required string field {section}.{key}")
    return value


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config_yaml)
    registry = load_yaml(args.models_yaml)
    specs = resolve_model_specs(config)
    assets = resolve_assets(specs, registry)

    status = prepare_assets(assets, args.model_dir, args.check, args.force)
    if status == 0 and not args.check and not args.skip_manifest_in:
        write_manifest_in(args.manifest_in, assets)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
