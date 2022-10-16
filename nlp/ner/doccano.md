# doccano安装及使用

## step.1 本地安装doccano
本地测试环境python=3.8
```bash
pip install doccano
```

## step.2 初始化数据库和账户（用户名和密码可替换为自定义值）
```bash
doccano init
doccano createuser --username my_admin_name --password my_password
```

## step.3 配置外网访问
启动时会提示允许访问的ip，根据这个去找。另外记得设置防火墙
```bash
./python3/lib/python3.8/site-packages/gunicorn/config.py
```

## step.4 启动doccano
在一个窗口启动doccano的WebServer，保持窗口
```bash
doccano webserver --port 8000
```
在另一个窗口启动doccano的任务队列
```bash
doccano task
```

## step.5 进行标注
http://82.156.187.109:8801/