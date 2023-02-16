## Rapid Table
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapid-table/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid-table"></a>
    <a href="https://pepy.tech/project/rapid-table"><img src="https://static.pepy.tech/personalized-badge/rapid-table?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
</p>

#### 简介和说明
- 该部分主要是做文档类图像的表格结构还原。模型来源：[PaddleOCR 表格识别](https://github.com/PaddlePaddle/PaddleOCR/blob/133d67f27dc8a241d6b2e30a9f047a0fb75bebbe/ppstructure/table/README_ch.md)
- 具体来说，就是分析给定的表格图像，将表格还原为对应的html格式。
- 目前支持两种类别的表格识别模型：中文和英文表格识别模型，具体可参见下面表格：

    | 模型类型  |        模型名称         | 模型大小 |
    |:---:|:---:|:---:|
    |   英文   | `en_ppstructure_mobile_v2_SLANet.onnx` |  7.3M   |
    |   中文   |   `ch_ppstructure_mobile_v2_SLANet.onnx`    |  7.4M   |
- 模型下载地址为：[百度网盘](https://pan.baidu.com/s/1PI9fksW6F6kQfJhwUkewWg?pwd=p29g) | [Google Drive](https://drive.google.com/drive/folders/1DAPWSN2zGQ-ED_Pz7RaJGTjfkN2-Mvsf?usp=sharing)

#### 使用方式
1. pip安装
   - 由于模型较小，预先将英文表格识别模型(`en_ppstructure_mobile_v2_SLANet.onnx`)打包进了whl包内，如果做英文表格识别，可直接安装使用
        ```bash
        $ pip install rapid-table
        ```
2. python脚本运行
   ```python
    import cv2
    from rapid_table import RapidTable

    # RapidTable类提供model_path参数，可以自行指定上述2个模型，默认是en_ppstructure_mobile_v2_SLANet.onnx
    # table_engine = RapidTable(model_path='ch_ppstructure_mobile_v2_SLANet.onnx')
    table_engine = RapidTable()

    img = cv2.imread('test_images/table.jpg')
    table_html_str, _ = table_engine(img)
    print(table_html_str)
   ```
3. 终端运行
   - 用法:
     ```bash
     $ rapid_table -h
     usage: rapid_table [-h] [-v] -img IMG_PATH [-m MODEL_PATH]

     optional arguments:
     -h, --help            show this help message and exit
     -v, --vis             Wheter to visualize the layout results.
     -img IMG_PATH, --img_path IMG_PATH
                           Path to image for layout.
     -m MODEL_PATH, --model_path MODEL_PATH
                           The model path used for inference.
     ```
   - 示例:
     ```bash
     $ rapid_table -v -img test_images/table.jpg
     ```

4. 结果
    - 返回结果
        ```html
        <html><body><table><thead><tr><td>Methods</td><td></td><td></td><td></td><td>FPS</td></tr></thead><tbody><tr><td>SegLink [26]</td><td>70.0</td><td>86d><td.0</td><td>77.0</td><td>8.9</td></tr><tr><td>PixelLink [4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td></td></tr><tr><td>TextSnake [18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField [37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2</td></tr><tr><td>MSR[38]</td><td>76.7</td><td>87.87.4</td><td>81.7</td><td></td></tr><tr><td>FTSN [3]</td><td>77.1</td><td>87.6</td><td>82.0</td><td></td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><><ttd></td></tr><tr><td>CRAFT [2]</td><td>78.2</td><td>88.2</td><td>82.9</td><td>8.6</td></tr><tr><td>MCN[16]</td><td>79</td><td>88</td><td>83</td><td></td></tr><tr><td>ATRR</>[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td></td></tr><tr><td>PAN [34]</td><td>83.8</td><td>84.4</td><td>84.1</td><td>30.2</td></tr><tr><td>DB[12]</td><td>79.2</t91/d><td>91.5</td><td>84.9</td><td>32.0</td></tr><tr><td>DRRG[41]</td><td>82.30</td><td>88.05</td><td>85.08</td><td></td></tr><tr><td>Ours (SynText)</td><td>80.68</td><td>85<t..40</td><td>82.97</td><td>12.68</td></tr><tr><td>Ours (MLT-17)</td><td>84.54</td><td>86.62</td><td>85.57</td><td>12.31</td></tr></tbody></table></body></html>
        ```
   - 可视化结果
    <div align="center">
        <table><thead><tr><td>Methods</td><td></td><td></td><td></td><td>FPS</td></tr></thead><tbody><tr><td>SegLink [26]</td><td>70.0</td><td>86d><td.0</td><td>77.0</td><td>8.9</td></tr><tr><td>PixelLink [4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td></td></tr><tr><td>TextSnake [18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField [37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2</td></tr><tr><td>MSR[38]</td><td>76.7</td><td>87.87.4</td><td>81.7</td><td></td></tr><tr><td>FTSN [3]</td><td>77.1</td><td>87.6</td><td>82.0</td><td></td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><><ttd></td></tr><tr><td>CRAFT [2]</td><td>78.2</td><td>88.2</td><td>82.9</td><td>8.6</td></tr><tr><td>MCN[16]</td><td>79</td><td>88</td><td>83</td><td></td></tr><tr><td>ATRR</>[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td></td></tr><tr><td>PAN [34]</td><td>83.8</td><td>84.4</td><td>84.1</td><td>30.2</td></tr><tr><td>DB[12]</td><td>79.2</t91/d><td>91.5</td><td>84.9</td><td>32.0</td></tr><tr><td>DRRG[41]</td><td>82.30</td><td>88.05</td><td>85.08</td><td></td></tr><tr><td>Ours (SynText)</td><td>80.68</td><td>85<t..40</td><td>82.97</td><td>12.68</td></tr><tr><td>Ours (MLT-17)</td><td>84.54</td><td>86.62</td><td>85.57</td><td>12.31</td></tr></tbody></table>
    </div>