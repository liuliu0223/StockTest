'''
import pandas as pd
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

%matplotlib inline
pd.set_option('display.max_colwidth', 200)

import pandas_datareader as pdr
taafgaDict = {'腾讯':'0700.hk','阿里巴巴':'baba','苹果':'AAPL','Facebook':'FB','谷歌':'GOOG','亚马逊':'AMZN'}

start_date = pd.to_datetime('2010-01-01')
stop_date = pd.to_datetime('2016-03-01')

#spy = pdr.data.get_data_famafrench('600797.SS', start_date,stop_date)
spy = pdr.moex.MoexReader('SBER', '2020-07-02', '2020-07-03').read_all_boards()
spy.head()
'''


# 绘制表格
import pandas as pd
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

# 个股信息查询
# http://quote.eastmoney.com/concept/sh603777.html?from=classic
import akshare as ak


def get_sh_stockinfo(s_code):
    df = ak.stock_individual_info_em(symbol=s_code)
    print(df)

'''
# 股票市场总貌
import akshare as ak


# 上海证券交易所
# http://www.sse.com.cn/market/stockdata/statistic/
def sh_df():
    stock_sse_summary_df = ak.stock_sse_summary()
    print(stock_sse_summary_df)


# 深圳证券交易所
# 证券类别统计
# http://www.szse.cn/market/overview/index.html
def sz_df():
    stock_szse_summary_df = ak.stock_szse_summary()
    print(stock_szse_summary_df)


# 深圳证券交易所
# 地区交易排序
# http://www.szse.cn/market/overview/index.html
def sz_area():
    stock_szse_area_summary_df = ak.stock_szse_area_summary(date="202203")
    print(stock_szse_area_summary_df)


# 深圳证券交易所
# 股票行业成交
# http://docs.static.szse.cn/www/market/periodical/month/W020220511355248518608.html
def sz_sector():
    stock_szse_sector_summary_df = ak.stock_szse_sector_summary(symbol="当年", date="202204")
    print(stock_szse_sector_summary_df)


# 上海证券交易所
# 每日概况
#  http://www.sse.com.cn/market/stockdata/overview/day/
def sh_day():
    stock_sse_deal_daily_df = ak.stock_sse_deal_daily(date="20201111")
    print(stock_sse_deal_daily_df)


def get_account_statistics():
    # 股票账户统计月度
    """
    输出参数
    名称    类型    描述
    数据日期    object    -
    新增投资者-数量    float64    注意单位: 万户
    新增投资者-环比    float64    -
    新增投资者-同比    float64    -
    期末投资者-总量    float64    注意单位: 万户
    期末投资者-A股账户    float64    注意单位: 万户
    期末投资者-B股账户    float64    注意单位: 万户
    沪深总市值    float64    -
    沪深户均市值    float64    注意单位: 万
    上证指数-收盘    float64    -
    上证指数-涨跌幅    float64    -
    """
    account = ak.stock_account_statistics_em()
    account.set_index("数据日期", inplace=True)  # 设置索引值
    account.to_csv("I:\\bianchengxx\\pythonxx\\backtrader_001\\datas\\stock_account_statistics.csv")
    print(account)
'''

if __name__ == '__main__':
    stockcode = "002895"
    get_sh_stockinfo(stockcode)
    # sh_df()
    # sz_df()
    # sz_area()
    # sz_sector()
    # sh_day()
    #get_account_statistics()


# -*- coding:utf-8 -*-
# Python 实用宝典
# 量化投资原来这么简单(1)
# 2020/04/12

import backtrader as bt
import json

if __name__ == '__main__':

    # 初始化模型
    cerebro = bt.Cerebro()
    # 设定初始资金
    cerebro.broker.setcash(100000.0)

    # 策略执行前的资金
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    # 策略执行后的资金
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

