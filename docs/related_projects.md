<details open>
<summary>目录</summary>

- [相关项目](#相关项目)
  - [版面分析](#版面分析)
  - [表格识别](#表格识别)
  - [视频OCR](#视频ocr)
  - [印章OCR](#印章ocr)
  - [卡证OCR](#卡证ocr)

</details>

## 相关项目
- 以下几个方向，每个都是比较独立的方向，但是却和OCR有着千丝万缕的关系，关于它们的资料就像散落在天空中的星星一般，散发着微弱的光芒，这里要做的就是将这些点点光芒聚集起来。
- 这里将会汇总出以下几个OCR周边项目的一些文档和资源，包括学术动态和一些工程化代码。
- 欢迎各位小伙伴提供PR。

### 版面分析
- 相关论文和帖子：
  - [版面分析方法汇总](https://zhuanlan.zhihu.com/p/392058153)
- 相关工程：
  - [PaddleOCR Layout](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/ppstructure/layout/README_ch.md)
  - [海康DAVAR VSR](https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/text_layout/VSR)
- 数据集汇总：
  - 英文版面分析数据集：
    - [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet): IBM构建，34万张图像，分为5类：text, title list table figure。
    - [DocBank](https://doc-analysis.github.io/docbank-page/index.html)：微软亚洲研究院构建，50万英文文档图像，分为12类：摘要、作者、标题、公式、图形、页脚、列表、段落、参考、节标题、表格和文章标题。

  - 中文版面分析数据集：
    - [CDLA](https://github.com/buptlihang/CDLA)：中文文档版面分析数据集，面向中文文献类（论文）场景，总共6000张（5000训练，1000测试），分为10类：正文、标题、图片、图片标题、表格、表格标题、页眉、页脚、注释和公式。

### 表格识别
- 相关论文和帖子：
  - [OCR之表格结构识别综述](https://blog.csdn.net/shiwanghualuo/article/details/123726879)
  - [合合信息：表格识别与内容提炼技术理解及研发趋势](https://blog.csdn.net/INTSIG/article/details/123000010?spm=1001.2014.3001.5502)
  - [论文阅读: （ICDAR2021 海康威视）LGPMA（表格识别算法）及官方源码对应解读](https://blog.csdn.net/shiwanghualuo/article/details/125047732?spm=1001.2014.3001.5501)
- 相关工程：
  - [海康官方LGPMA源码](https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/table_recognition/lgpma)
  - [LGPMA Inference](https://github.com/SWHL/LGPMA_Infer)
  - [PaddleOCR Table](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/ppstructure/table/README_ch.md)
- 数据集汇总：
  - 英文表格识别数据集：
    - [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet): IBM构建，568k+文档图像数据，包括表格图像和对应的HTML标注。

### 视频OCR

### 印章OCR

### 卡证OCR