#! /bin/bash

sdk_dir="sdk_rapidocr_v1.0.0"

echo ">>>>>>打包现有的"
python setup.py bdist_wheel
mv dist/*.whl ${sdk_dir}/
rm -r build
rm -r dist
rm -r *.egg-info

echo ">>>>>>安装在当前环境下"
pip install ${sdk_dir}/*.whl

echo ">>>>>>覆盖README.md"
cp "requirements.txt" "${sdk_dir}/"

echo ">>>>>>测试whl是否有误"

cd ${sdk_dir}
python test_demo.py
