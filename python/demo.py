# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2

from rapidocr_onnxruntime import RapidOCR, VisRes

# from rapidocr_paddle import RapidOCR, VisRes
# from rapidocr_openvino import RapidOCR, VisRes

if __name__ == '__main__':

    engine = RapidOCR()
    vis = VisRes(0.4)

    image_path = "tests/test_files/bad.png"
    with open(image_path, "rb") as f:
        img = f.read()

    result, elapse_list = engine(img, rec_word_box=True, text_score=0.4)
    word_result = []
    for res in result:
        score = res[2]
        for word,word_box in zip(res[3], res[4]):
            word_res = []
            word_res.append(word_box)
            word_res.append(word)
            word_res.append(score)
            word_result.append(word_res)
    # print(result)
    print(elapse_list)

    # boxes, txts, scores = list(zip(*result))
    boxes, txts, scores = list(zip(*word_result))

    font_path = "resources/fonts/FZYTK.TTF"
    vis_img = vis(img, boxes, txts, scores, font_path)
    cv2.imwrite("vis.png", vis_img)
