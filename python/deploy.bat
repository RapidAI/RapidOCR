python setup_directml.py clean --all
RD /S /Q dist
RD /S /Q build

:: Build the wheel package
python setup_directml.py bdist_wheel
twine upload dist/*