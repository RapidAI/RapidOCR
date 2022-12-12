# -*- encoding: utf-8 -*-
# @Author: innerVoi
import cv2
from task import check_pic_type
from detection import Detection
from rapidocr_onnxruntime import RapidOCR

class TestDetection(object):
    """
    对如下安全功能提供测试样例：
    检查是否是图片，识别文本中是否有js/css/url注入，文本是否包含违规词汇
    Provided by BUPT
    """
    def test_detection_is_pic(self):
        img_path = 'test_files/is_pic.jpg'
        img = cv2.imread(img_path)
        is_pic = check_pic_type(img)
        assert is_pic == False

    def test_detection_js(self):
        rec_res_data = self.read_rec_res_data('test_files/js.png')
        det = Detection(rec_res_data)
        assert det.js_test() == True

    def test_detection_css(self):
        rec_res_data = self.read_rec_res_data('test_files/css.png')
        det = Detection(rec_res_data)
        assert det.css_test() == True

    def test_detection_url(self):
        rec_res_data = self.read_rec_res_data('test_files/url_test.png')
        det = Detection(rec_res_data)
        assert det.url_test() == True

    def test_detection_il_word(self):
        rec_res_data = self.read_rec_res_data('test_files/il_word.jpg')
        det = Detection(rec_res_data)
        assert det.il_word_test() == True

    def load_ocr(self, image):
        text_sys = RapidOCR('config.yaml')
        final_result, img, elapse_part = text_sys(image)
        return final_result

    def read_rec_res_data(self, image_path):
        image = cv2.imread(image_path)
        rec_res_data = self.load_ocr(image)
        return  rec_res_data