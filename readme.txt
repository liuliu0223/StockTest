一、学习Python爬虫需要注意以下几点：

1、确保你已经掌握了Python的基础语法和面向对象编程的基本概念。
2、了解HTTP协议和HTML语言的基本知识，这是爬虫的基础。
3、了解常见的爬虫框架和库，如Scrapy、BeautifulSoup、Requests等，选择一个适合自己的工具进行学习。
4、遵守网站的爬虫规则，不要过度频繁地访问同一个网站，以免被封IP或者被视为恶意攻击。
5、学会使用代理IP和User-Agent等技术，以避免被网站识别为爬虫。
6、学会数据清洗和数据存储，将爬取到的数据进行处理和保存。
7、不要违反法律法规，不要爬取敏感信息或者侵犯他人隐私。

二、python爬出六部曲
第一步：安装requests库和BeautifulSoup库：
import` `requests``from` `bs4 ``import` `BeautifulSoup
第二步：获取爬虫所需的header和cookie：
第三步：获取网页：
第四步：解析网页：
第五步：分析得到的信息，简化地址
第六步：爬取内容，清洗数据

生成可执行文件方法一：
用pyInstaller
[root@localhost ~]# pip install pyinstaller
pyinstaller打包
C:\03 My Program\PyProject1.0> pyinstaller -F -w main.py # 有input函数的时候，就不带-w，不然会报错。
C:\03 My Program\PyProject1.0> pyinstaller -F main.py
#用自定义图片打包成可执行文件
pyinstaller -F -w -i G:\automation\tpian.ico G:\automation\test.py

结果文件地址：C:\03 My Program\PyProject1.0\dist\目录下


三、数据分析与挖掘
利用Akshare获取股票数据 #https://blog.csdn.net/zhh_920509/article/details/129757516
1. 使用 tushare 库：tushare 是一个免费的、开源的 Python 股票数据接口，可以获取 A股的历史行情、实时行情、财务数据等。使用 tushare 库需要先安装，然后注册 tushare 账号，获取 token，即可使用 tushare 提供的 API 获取数据。
2. 使用 akshare 库：akshare 是一个开源的 Python 股票数据接口，可以获取 A股的历史行情、实时行情、财务数据等。使用 akshare 库需要先安装，然后使用 akshare 提供的 API 获取数据。
3. 使用 baostock 库：baostock 是一个免费的、开源的 Python 股票数据接口，可以获取 A股的历史行情、实时行情、财务数据等。使用 baostock 库需要先安装，然后注册 baostock 账号，获取 token，即可使用 baostock 提供的 API 获取数据。
4. 使用聚宽数据接口：聚宽是一个提供 A股数据的商业服务，可以获取 A股的历史行情、实时行情、财务数据等。使用聚宽需要先注册账号，然后购买相应的数据服务，使用聚宽提供的 API 获取数据。


'''
# 绘制表格
import matplotlib.pyplot as plt

# 创建表格数据
data = {'姓名': ['张三', '李四', '王五'], '年龄': [18, 20, 22], '成绩': [85, 90, 92]}

# 创建表格
plt.table(cellText=[data['姓名'], data['年龄'], data['成绩']], colLabels=list(data.keys()), loc='center')

# 显示表格
plt.show()
'''

#JSON 处理的是文件而不是字符串，你可以使用 json.dump() 和 json.load() 来编码和解码JSON数据.
#!/usr/bin/python3

import json

# Python 字典类型转换为 JSON 对象
data1 = {
    'no' : 1,
    'name' : 'Runoob',
    'url' : 'http://www.runoob.com'
}
json_str = json.dumps(data1)
print ("Python 原始数据：", repr(data1))
print ("JSON 对象：", json_str)

# 将 JSON 对象转换为 Python 字典
data2 = json.loads(json_str)
print ("data2['name']: ", data2['name'])
print ("data2['url']: ", data2['url'])


# 写入 JSON 数据
with open('data.json', 'w') as f:
    json.dump(data, f)

# 读取数据
with open('data.json', 'r') as f:
    data = json.load(f)