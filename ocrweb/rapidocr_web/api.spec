# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

import rapidocr_onnxruntime

block_cipher = None

package_name = 'rapidocr_onnxruntime'
install_dir = Path(rapidocr_onnxruntime.__file__).resolve().parent

onnx_paths = list(install_dir.rglob('*.onnx'))
yaml_paths = list(install_dir.rglob('*.yaml'))

onnx_add_data = [(str(v.parent), f'{package_name}/{v.parent.name}')
                 for v in onnx_paths]

yaml_add_data = []
for v in yaml_paths:
    if package_name == v.parent.name:
        yaml_add_data.append((str(v.parent / '*.yaml'), package_name))
    else:
        yaml_add_data.append(
            (str(v.parent / '*.yaml'), f'{package_name}/{v.parent.name}'))

add_data = list(set(yaml_add_data + onnx_add_data))

a = Analysis(
    ['api.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='api',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['./static/css/favicon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='api',
)
