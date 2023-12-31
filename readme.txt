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

按照可用库文件
PyProject1.0> pip install tensorflow -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

1. 8个国内镜像源
以下是中国常见的pip镜像源，按照完全度和下载速度排序，需要注意的是，镜像源的完全度和速度可能因地域和时间而异，建议根据自己的实际情况选择合适的镜像源。

1.1 清华大学（完全度和速度都很好，是一个优秀的pip镜像源）
https://pypi.tuna.tsinghua.edu.cn/simple
1.2 阿里云（完全度和速度也很好，是一个不错的选择）
https://mirrors.aliyun.com/pypi/simple/
1.3 网易（速度比较快，但是完全度有限）
https://mirrors.163.com/pypi/simple/
1.4 豆瓣（速度较快，但是完全度也有限）
https://pypi.douban.com/simple/
1.5 百度云（速度较快，但是完全度也有限）
https://mirror.baidu.com/pypi/simple/
1.6 中科大（速度较快，但完全度不如前面几个镜像源）
https://pypi.mirrors.ustc.edu.cn/simple/
1.7 华为云（完全度和速度均中等）
https://mirrors.huaweicloud.com/repository/pypi/simple/
1.8 腾讯云（速度一般，完全度也一般）
https://mirrors.cloud.tencent.com/pypi/simple/

生成可执行文件方法一：
用pyInstaller
[root@localhost ~]# pip install pyinstaller
pyinstaller打包
C:\03 My Program\PyProject1.0> pyinstaller -F -w main.py # 有input函数的时候，就不带-w，不然会报错。
C:\03 My Program\PyProject1.0> pyinstaller -F main.py
#用自定义图片打包成可执行文件
pyinstaller -F -w -i G:\automation\tpian.ico G:\automation\test.py

结果文件地址：C:\03 My Program\PyProject1.0\dist\目录下
PS C:\01 Work\13 program\PyProject1.0> pip install Ta-Lib -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com


三、数据分析与挖掘
利用Akshare获取股票数据 #https://blog.csdn.net/zhh_920509/article/details/129757516
1. 使用 tushare 库：tushare 是一个免费的、开源的 Python 股票数据接口，可以获取 A股的历史行情、实时行情、财务数据等。使用 tushare 库需要先安装，然后注册 tushare 账号，获取 token，即可使用 tushare 提供的 API 获取数据。
2. 使用 akshare 库：akshare 是一个开源的 Python 股票数据接口，可以获取 A股的历史行情、实时行情、财务数据等。使用 akshare 库需要先安装，然后使用 akshare 提供的 API 获取数据。
3. 使用 baostock 库：baostock 是一个免费的、开源的 Python 股票数据接口，可以获取 A股的历史行情、实时行情、财务数据等。使用 baostock 库需要先安装，然后注册 baostock 账号，获取 token，即可使用 baostock 提供的 API 获取数据。
4. 使用聚宽数据接口：聚宽是一个提供 A股数据的商业服务，可以获取 A股的历史行情、实时行情、财务数据等。使用聚宽需要先注册账号，然后购买相应的数据服务，使用聚宽提供的 API 获取数据。

四、机器学习算法实现路径：
4.1 导入数据
4.2 研究数据
4.3 数据预处理(数据清洗）
    4.3.1 无效数据处理（删除空行，空数据替换）
    4.3.2 数据无量纲处理：min-max归一化
    4.3.3 初步特征分析：GRA(灰色关联度分析算法)、皮尔斯系数（热力图），获取关键特征值
    4.3.4 数据集构建（训练集+测试集），8:2
4.4 搭建模型
    4.4.1 LSTM神经网络模型
    4.4.2 XGBoost模型搭建
4.5 数据可视化及评估


# 绘制表格
import matplotlib.pyplot as plt

# 创建表格数据
data = {'姓名': ['张三', '李四', '王五'], '年龄': [18, 20, 22], '成绩': [85, 90, 92]}

# 创建表格
plt.table(cellText=[data['姓名'], data['年龄'], data['成绩']], colLabels=list(data.keys()), loc='center')

# 显示表格
plt.show()


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


20230101
20230831
sh600602
sh603918
sh688031
sz300598
sz002475

五、舆情挖掘与分析
1、搜集信息，文本预处理，例如：https://so.eastmoney.com/News/s?keyword=%E6%AC%A7%E8%8F%B2%E5%85%89
1）用jieba库实现中文分词
import jieba

text = "我喜欢吃苹果"
seg_list = jieba.cut(text, cut_all=False)
print(" ".join(seg_list))

2）停用词过滤&正则表达式过滤
import re
import nltk

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
text = "This is an example text for stopwords."
tokens = re.findall('\w+', text)
filtered_tokens = [token for token in tokens if not token.lower() in stopwords]
print(filtered_tokens)

2、特征提取
1）情感分析：用TextBlob库实现情感分析，polarity的值表示情感极性，范围在-1到1之间，值越大表示正面情感越强。
from textblob import TextBlob

text = "This is a happy day."
blob = TextBlob(text)
polarity = blob.sentiment.polarity
print(polarity)

2）主题模型：用gensim库实现主题模型
from gensim import corpora, models

texts = [["this", "is", "an", "example", "text"], ["another", "example", "text"]]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
for topic in lda.print_topics(num_words=3):
    print(topic)

3）关键词抽取，用TextRank算法实现关键词抽取
import jieba.analyse

text = "我爱北京天安门，天安门上太阳升。"
keywords = jieba.analyse.textrank(text, withWeight=True, topK=3)
for keyword, weight in keywords:
    print(keyword, weight)

4）生成词云
from wordcloud import WordCloud
# 生成词云
def create_word_cloud(word_dict):
    # 支持中文, SimHei.ttf可从以下地址下载：https://github.com/cystanford/word_cloud
    wc = WordCloud(
        font_path="./source/SimHei.ttf",
        background_color='white',
        max_words=25,
        width=1800,
        height=1200,
    )
    word_cloud = wc.generate_from_frequencies(word_dict)
    # 写词云图片
    word_cloud.to_file("wordcloud2.jpg")
    # 显示词云文件
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()

# 根据词频生成词云
create_word_cloud(word_dict)


3、事件检测，根据网络上的信息和趋势，识别和预测可能发生的突发事件，为决策者提供及时有效的决策依据
用基于机器学习和深度学习的方法实现事件检测
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["This is an example text for event detection.", "Another example text."]
labels = [0, 1]
vectorizer = TfidfVectorizer()
X = vectorizer

4、实现相似度分析算法后，需要对模型进行测试以验证其效果
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine_similarity

# 生成模拟数据
text1 = "文本1：你好，我是一个人工智能助手。"
text2 = "文本2：你好，我也是一个人工智能助手。"

# 分词
text1_words = [word.lower() for word in text1.split()]
text2_words = [word.lower() for word in text2.split()]

# 去除停用词
stopwords = set(pd.read_csv('stopwords.txt', header=None)[0])
text1_words = [word for word in text1_words if word.lower() not in stopwords]
text2_words = [word for word in text2_words if word.lower() not in stopwords]

# 词干化
text1_words = [' '.join(word for word in text1_words)[:-1] for word in text1_words]
text2_words = [' '.join(word for word in text2_words)[:-1] for word in text2_words]

# 计算余弦相似度
sim_1 = cosine_similarity(text1_words, text2_words)[0][0]
sim_2 = cosine_similarity(text2_words, text1_words)[0][0]

print(f"余弦相似度: {sim_1}")
print(f"皮尔逊相关系数: {sim_2}")