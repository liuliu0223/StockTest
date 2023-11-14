#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import datetime
import sys
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import baostock as bs
import talib as ta

import MyStrategy.PrepairData as prd

# const value
STAKE = 1500  # volume once
START_CASH = 150000  # initial cost
COMM_VALUE = 0.002   # 费率
WIN_ENV_FLAG = False  # windows环境设置
FILEDIR = 'stocks'
TRAIN_DIR = "train"
STOCK_INFO_FILE = "text.txt"

# globle value
stock_pnl = []  # Net profit
stock_list = []  # stock list
special_info = []  # 特别注意的事项
special_code = ""
crossover_list = []  # crossover的内容


class Strategymine(bt.Strategy):
    """
    主策略程序
    """
    params = dict(
        pfast=5,       # period for the fast moving average
        pslow=10      # period for the slow moving average
        #pslow=20     # period for the more slow moving average
        #pslow=30      # period for the more slow moving average
    )

    # 通用日志打印函数，可以打印下单、交易记录，非必须，可选
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    # 初始化函数，初始化属性、指标的计算，only once time
    def __init__(self):
        self.data_close = self.datas[0].close  # close data
        # initial data
        self.order = None
        self.buy_price = None
        self.buy_comm = None  # trial fund
        self.buy_cost = None  # cost
        self.pos = None  # pos
        self.cash_valid = None  # available fund
        self.valued = None  # total fund
        self.pnl = None  # profit
        self.sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        self.sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.dif = self.sma1 - self.sma2
        self.crossover = bt.ind.CrossOver(self.sma1, self.sma2)  # crossover signal

        self.crossover_buy = False
        self.crossover_sell = False

    # order statement information
    def notify_order(self, order):
        txt = ""
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 检查订单是否完成
        if order.status in [order.Completed]:
            price = order.executed.price
            comm = order.executed.comm
            cost = order.executed.value
            pos = self.getposition(self.data).size
            fund = self.broker.getvalue()
            if order.isbuy():
                self.crossover_buy = True
                txt = 'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, fund Value %.2f, pos Size %.2f' % \
                      (price, cost, comm, fund, pos)
                self.log(txt)
            elif order.issell():
                self.crossover_sell = True
                self.pos = self.getposition(self.data).size
                txt = 'SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, fund Value %.2f, pos Size %.2f' % \
                      (price, cost, comm, fund, pos)
                self.log(txt)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Canceled:
                self.log('order cancel!')
            elif order.status == order.Margin:
                self.log('fund not enough!')
            elif order.status == order.Rejected:
                self.log('order reject!')

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('notify_trade: business profit: %.2f, Net profit  %.2f' % (trade.pnl, trade.pnlcomm))

    #  loop in every business day
    def next(self):
        size = self.getposition(self.data).size
        price = self.getposition(self.data).price
        valid_cash = self.broker.getcash()
        fund = self.broker.getvalue()
        buy_comm = price * size * COMM_VALUE
        #self.date = datetime.datetime.strftime(self.datas[0].datetime.date(0), "%Y-%m-%d")
        self.date = self.datas[0].datetime.date(0)
        if self.crossover > 0:  # if golden cross, valid=datetime.datetime.now() + datetime.timedelta(days=3)
            crossover = 1
            self.log('Available Cash: %.2f, Total fund: %.2f, pos: %.2f' % (valid_cash, fund, size))
            if not self.position:  # Outside, buy
                self.order = self.buy()
                txt = 'Outside, golden cross buy, close: %.2f，Total fund：%.2f, pos: %.0f' % \
                      (self.data_close[0], fund, size)

                self.log('Outside, golden cross buy, close: %.2f，Total fund：%.2f, pos: %.2f' %
                         (self.data_close[0], fund, size))
            else:
                if (valid_cash - price * STAKE - buy_comm) > 0:
                    self.log('Available Cash: %.2f, Total fund: %.2f, pos: %.2f' % (valid_cash, fund, size))
                    self.order = self.buy()
                    self.log('Outside, golden cross buy, close: %.2f，Total fund：%.2f， pos: %.2f' %
                             (self.data_close[0], valid_cash, size))
        elif self.crossover < 0:  # Inside and dead cross
            crossover = -1
            self.log('CrossOver<0: Available Cash: %.2f, Total fund: %.2f, pos: %.2f' % (valid_cash, fund, size))
            if self.position:
                if fund > START_CASH * 1.03:
                    self.order = self.close(size=size)
                else:
                    self.order = self.close(size=size)
                self.log('Inside dead cross, sell, close:  %.2f，Total fund：%.2f, pos: %.2f' %
                         (self.data_close[0], fund, size))
        else:
            crossover = 0
            self.log('CrossOver=0:Available Cash: %.2f, Total fund: %.2f, pos: %.2f' % (valid_cash, fund, size))
        crossover_list.append({'date': datetime.datetime.strptime(str(self.date), "%Y-%m-%d"), 'crossover': crossover})


# 从指定文件中读取数据，并运行回测函数
def run_strategy(f_startdate, f_enddate, f_data, f_stake=STAKE):
    sdf = pd.DataFrame(f_data)
    sdf.set_index('date', inplace=True)
    from_date = datetime.datetime.strptime(f_startdate, "%Y%m%d")
    end_date = datetime.datetime.strptime(f_enddate, "%Y%m%d")
    data = bt.feeds.PandasData(dataname=sdf, fromdate=from_date, todate=end_date)  # 加载数据
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()  # 初始化回测系统
    cerebro.adddata(data)  # 将数据传入回测系统
    cerebro.broker.setcash(START_CASH)  # set initial fund
    cerebro.broker.setcommission(commission=COMM_VALUE)  # set trad rate 0.2%
    stake = f_stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)  # set trade volume
    cerebro.addstrategy(Strategymine)  # period = [(5, 10), (20, 100), (2, 10)]) , 运行策略
    cerebro.run(maxcpus=1)  # 运行回测系统
    #cerebro.plot(style='candlestick', title='stock')  # 画图


# 计算MACD指标参数
def computeMACD(f_code, f_startdate, f_enddate):
    plt.switch_backend('agg')
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # set Chinese to draw picture
    plt.rcParams["axes.unicode_minus"] = False  # 设置画图时的负号显示

    rs = prd.prepare_data_k(f_code, f_startdate, f_enddate)
    # 打印结果集
    result_list = []
    df2 = None
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        result_list.append(rs.get_row_data())
    if len(result_list) > 0:
        df = pd.DataFrame(result_list, columns=rs.fields)
        # 剔除停盘数据
        df2 = df[df['tradeStatus'] == '1']
        # 获取 dif,dea,hist，它们的数据类似是 tuple，且跟 df2 的 date 日期一一对应
        # 记住了 dif,dea,hist 前 33 个为 Nan，所以推荐用于计算的数据量一般为你所求日期之间数据量的 3 倍
    # 这里计算的 hist 就是 dif-dea,而很多证券商计算的 MACD=hist*2=(dif-dea)*2
    dif, dea, hist = ta.MACD(df2['close'].astype(float).values, fastperiod=12, slowperiod=26, signalperiod=9)
    df3 = pd.DataFrame({'dif': dif[33:], 'dea': dea[33:], 'hist': hist[33:]},
                       index=df2['date'][33:],
                       columns=['dif', 'dea', 'hist'])
    code = prd.getcodebytype(f_code, ctype=None)
    df_info = prd.get_sh_stock(code)
    code_name = df_info.values[5][1]
    plot_name = 'MACD_' + code_name
    df3.plot(title=plot_name)
    #plt.show()
    Special_ops = []
    # 寻找 MACD 金叉和死叉
    datenumber = int(df3.shape[0])
    for i in range(datenumber - 1):
        if ((df3.iloc[i, 0] <= df3.iloc[i, 1]) & (df3.iloc[i + 1, 0] >= df3.iloc[i + 1, 1])):
            delta_dif = df3.iloc[i + 1, 0] - df3.iloc[i, 0]
            delta_dea = df3.iloc[i + 1, 1] - df3.iloc[i, 1]
            txt = "MACD golden cross date：" + df3.index[i + 1] + "; latest dif:" \
                        + str(round(float(df3.iloc[i + 1, 0]), 2)) + "; latest dea: " \
                        + str(round(float(df3.iloc[i + 1, 1]), 2)) + "; delta_dif:" \
                        + str(round(float(delta_dif), 2)) + "; delta_dea: " \
                        + str(round(float(delta_dea), 2))

            Special_ops.append({'code': f_code, 'date': df3.index[i + 1], 'msg': txt})
            print("MACD 金叉的日期：" + df3.index[i + 1])
        if ((df3.iloc[i, 0] >= df3.iloc[i, 1]) & (df3.iloc[i + 1, 0] <=df3.iloc[i + 1, 1])):
            print("MACD 死叉的日期：" + df3.index[i + 1])
            delta_dif = df3.iloc[i + 1, 0] - df3.iloc[i, 0]
            delta_dea = df3.iloc[i + 1, 1] - df3.iloc[i, 1]
            txt = "MACD dead cross date：" + df3.index[i + 1] + "; latest dif:" \
                        + str(round(float(df3.iloc[i + 1, 0]), 2)) + "; latest dea: " \
                        + str(round(float(df3.iloc[i + 1, 1]), 2)) + "; delta_dif:" \
                        + str(round(float(delta_dif), 2)) + "; delta_dea: " \
                        + str(round(float(delta_dea), 2))
            Special_ops.append({'code': f_code, 'date': df3.index[i + 1], 'msg': txt})
    bs.logout()
    plt.close()
    return (dif, dea, hist, Special_ops)


def calculateEMA(period, closeArray, emaArray=[]):
    length = len(closeArray)
    nanCounter = np.count_nonzero(np.isnan(closeArray))
    if not emaArray:
        emaArray.extend(np.tile([np.nan], (nanCounter + period - 1)))
        firstema = np.mean(closeArray[nanCounter:nanCounter + period - 1])
        emaArray.append(firstema)
        for i in range(nanCounter + period, length):
            ema = (2 * closeArray[i] + (period - 1) * emaArray[-1]) / (period + 1)
            emaArray.append(ema)
    return np.array(emaArray)


def calculateMACD(closeArray, shortPeriod=12, longPeriod=26, signalPeriod=9):
    ema12 = calculateEMA(shortPeriod, closeArray, [])
    ema26 = calculateEMA(longPeriod, closeArray, [])
    diff = ema12 - ema26

    dea = calculateEMA(signalPeriod, diff, [])
    macd = (diff - dea)*2

    fast_values = diff   # 快线
    slow_values = dea    # 慢线
    diff_values = macd   # macd
    # return fast_values, slow_values, diff_values  # 返回所有的快慢线和macd值
    return fast_values[-1], slow_values[-1], diff_values[-1]    # 返回最新的快慢线和macd值
    # return round(fast_values[-1],5), round(slow_values[-1],5), round(diff_values[-1],5)


if __name__ == '__main__':
    print(sys.path)
    # get the code data from websit
    codes = prd.get_codes(STOCK_INFO_FILE)
    startdate = str(codes[0]).replace('\n', '')  # 回测开始时间
    enddate = str(codes[1]).replace('\n', '')  # 回测结束时间
    it = 0
    code = ""
    filepath = ""
    train_file = ""
    special_ops = None
    for code in codes:
        if it > 1:
            code = str(codes[it]).replace('\n', '')  # "sz300598"
            stock_file = prd.prepare_data(code, startdate, enddate)
            # 加载数据，进行数据处理和分析
            df = prd.load_data(stock_file)
            code_name = prd.get_sh_stock(code).values[5][1]  # code include sh
            # code = 'sh.600000'，startdate = '2017-03-01'，enddate = '2017-12-01'
            (dif, dea, hist, special_ops) = computeMACD(code, startdate, enddate)
            print(f'-----{code},{code_name} analyze end!')
        it += 1
