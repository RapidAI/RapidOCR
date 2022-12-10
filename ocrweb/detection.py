# -*- encoding: utf-8 -*-
# Provided by BUPT
# 检测文本扫描结果是否含有恶意信息
import re


# 用户自定义检测设置，若有异常则返回True
def detection(s, js=True, css=True, url=True, il_word=True, is_pic=False):
    res = False
    if js:
        res = js_test(s)
    if css:
        res = css_test(s)
    return res


# 检测文本结果是否含有js代码，若含有则返回True
def js_test(s):
    regex = r"[\s\S]*<script[\s\S]+</script *>[\s\S]*"
    res = re.match(regex, s)
    if res:
        return True
    return False


# 检测文本结果是否含有css代码，若含有则返回True
def css_test(s):
    regex = r"[\s\S]*<style[\s\S]+</style *>[\s\S]*"
    res = re.match(regex, s)
    if res:
        return True
    return False
