#! /bin/bash

sdk_dir="sdk_rapidocr_v1.0.0"
if [ -d "${sdk_dir}" ]; then
    echo "存在，删除掉"
    rm -r ${sdk_dir}

mkdir ${sdk_dir}

python setup.py bdist_wheel
mv dist/*.whl ${sdk_dir}/
rm -r build
rm -r dist
rm -r *.egg-info

cp "requirements.txt" "${sdk_dir}/"
cp -r resources ${sdk_dir}/
cp -r images ${sdk_dir}/
cp test_demo.py ${sdk_dir}/

mv ${sdk_dir} ../