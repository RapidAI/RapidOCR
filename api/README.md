# API

## 功能
### registration
POST传入username和password来进行注册
### login
POST传入username和password来登陆 获取access_token和refresh_token
### /logout/access
注销access token
### /logout/refresh
注销refresh token
### /token/refresh
未完成
### /users
查看全部用户和删除用户
### /ocr
通过access token进行Authorization:Bearer认证来获取ORC数据。
代码需要进行改写，此处POST仅作为功能测试，建议合并到/orcweb/rapidocr_web/orcweb.py中。

## 工作流程
registration -> login -> 利用access token进行 ocr。(access token具有expire)