import os
import shutil

print('Compile ocrweb')
os.system('pyinstaller -y main.spec')

print('Compile wrapper')
os.system('windres .\wrapper.rc -O coff -o wrapper.res')
os.system('gcc .\wrapper.c wrapper.res -o dist/ocrweb.exe')

print('Copy config.yaml')
shutil.copy2('config.yaml', 'dist/config.yaml')

print('Copy models')
shutil.copytree('models', 'dist/models', dirs_exist_ok=True)
os.remove('dist/models/.gitkeep')

print('Pack to ocrweb.zip')
shutil.make_archive('ocrweb', 'zip', 'dist')

print('Done')
