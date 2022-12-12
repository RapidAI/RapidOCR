# -*- encoding: utf-8 -*-
# Provided by BUPT
# 检测文本扫描结果是否含有恶意信息
# 识别文本扫描结果中的URL，并对其安全性进行检测
import re
import urllib.request
import requests
import numpy as np
import ahocorasick

class Detection():
    """
    检查OCR过程中的安全问题，包括：js/css/url注入和文本中是否包含违规词汇
    : js_test() -- js注入检测
    : css_test() -- css注入检测
    : url_test() -- url注入检测
    : il_word_test() -- 违规词汇检测
    Provided by BUPT
    """
    def __init__(self, rec_res_data):
        self.rec_res_data = rec_res_data
        boxes, txts, scores = list(zip(*self.rec_res_data))
        self.rec_res_str = ''.join(txts)

    def js_test(self):
        regex = r"[\s\S]*<script[\s\S]+</script *>[\s\S]*"
        res = re.match(regex, self.rec_res_str)
        if res:
            return True
        return False

    def css_test(self):
        regex = r"[\s\S]*<style[\s\S]+</style *>[\s\S]*"
        res = re.match(regex, self.rec_res_str)
        if res:
            return True
        return False

    def url_test(self):
        urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', self.rec_res_str)
        is_url = False
        for url in urls:
            state = self.url_check_wechat(url)
            if state == 0:
                print("Security Risk: False")
            else:
                print("Security Risk: True")
                is_url = True
        return is_url


    def url_check_wechat(self, url):
        """
        :params url: 表示被检查的url链接
        """
        link = requests.get(
            "https://api.kit9.cn/api/wechat_security/api.php",
            params={"url": url},
        )
        result = link.json()

        return result["data"]["state"]

    def il_word_test(self):
        """
        illegal_words.txt:
        将公开的违规词汇数据集由 'base64' 格式转为'utf-8' 格式得到
        （原数据链接：https://gitee.com/xstudio/badwords/tree/master）
        """
        word_list = []
        is_illegal = False
        with open('./illegal_words.txt', 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                t = line.split()[0].strip()
                word_list.append(t)
                line = f.readline()

        actree = ahocorasick.Automaton()
        for index, word in enumerate(word_list):
            actree.add_word(word, (index, word))
        actree.make_automaton()

        for text in self.rec_res_data:
            for i in actree.iter(text[1]):
                # 是否屏蔽违规词汇：若启用屏蔽，需更新rec_res_data并返回
                # text[1] = text[1].replace(i[1][1], "**")
                is_illegal = True

        return is_illegal