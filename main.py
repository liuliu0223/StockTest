# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import requests  # 导入requests库
from urllib import request
from bs4 import BeautifulSoup
import time
import fitz
import os

base_path = r'C:\Download'
docname = '中国数据要素市场体系总体框架和发展路径研究'
url = 'https://mp.weixin.qq.com/s/2XhRp3i2ZWX9Fxxe44y-NA'  # 要爬取的网址
path = base_path + '\\' + docname
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
page = request.Request(url, headers=headers)
page_info = request.urlopen(page).read().decode('utf-8')
soup = BeautifulSoup(page_info, 'html.parser')
titles = soup.find_all('img', class_="rich_pages wxw-img js_insertlocalimg")
data = []
i = 0

try:
    # 在C盘以只写的方式打开/创建一个名为 titles 的txt文件
    file = open(r'C:\Download\titles.txt', 'w')
    doc = fitz.open()
    if os.path.exists(path):
        print("文件夹已存在！")
    else:
        os.mkdir(path)
    for title in titles:
        # 将内容写入txt中
        if title is None:
            print('title:is none')
        else:
            text = str(title)
            midstrings = text.split('data-src="')
            file.write(text + '\n')
            urladds = midstrings[1].split('"')
            urladd = urladds[0]
            data.append(urladd)
            i = i + 1
            newpath = path + '\\' + str(i) + '.png'
            img = requests.get(urladd, headers=headers).content
            # url是img的url
            f = open(newpath, 'wb')  # 打开一个二进制文件
            f.write(img)
            time.sleep(1)
            print(newpath)
            if os.path.getsize(newpath):
                imgdoc = fitz.open(newpath)  # 打开图片
                pdfbytes = imgdoc.convert_to_pdf()  # 使用图片创建单页的 PDF
                imgpdf = fitz.open("pdf", pdfbytes)
                doc.insert_pdf(imgpdf)  # 将当前页插入文档
                if os.path.exists(path + '\\' + docname + ".pdf"):
                    os.remove(path + '\\' + docname + ".pdf")
                doc.save(path + '\\' + docname + ".pdf")  # 保存pdf文件

finally:
    if file:
        # 关闭文件（很重要）
        file.close()
    if doc:
        doc.close()


