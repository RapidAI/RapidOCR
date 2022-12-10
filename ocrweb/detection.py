# -*- encoding: utf-8 -*-
# Provided by BUPT
# 检测文本扫描结果是否含有恶意信息
# 识别文本扫描结果中的URL，并对其安全性进行检测
import re
import urllib.request
import requests

# 用户自定义检测设置，若有异常则返回True
def detection(s, js=True, css=True, url=True, il_word=True, is_pic=False):
    res = False
    if js:
        res = js_test(s)
    if css:
        res = css_test(s)
    if url:
        # 对检测到的每个URL安全性进行检测；
        # 若有安全风险，后台发出警示：“Security Risk: True”，前端文本检测结果显示不变；
        url_test(s)
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

# 本函数实现从检测到的文本中查找URL的功能
def read_url(string): 
    url = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', string)
    return url

# 本函数利用微信域名拦截API，实现URL安全性检测功能
def url_check_wechat(url):
    link = requests.get(
        "https://api.kit9.cn/api/wechat_security/api.php",
        params={"url": url},
    )
    result = link.json()
    return result["data"]["state"]

# 本函数输出URL安全性检测结果
def url_test(input):
    urls = read_url(input)
    for url in urls:
        state = url_check_wechat(url)
        if state == 0:    
            print("Security Risk: False")
        else:    
            print("Security Risk: True")
