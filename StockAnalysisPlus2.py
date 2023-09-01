#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import datetime
import os
import backtrader as bt
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# const value
STAKE = 2500  # volume once
START_CASH = 150000  # initial cost
COMM_VALUE = 0.002   # 费率

# globle value
stock_pnl = []  # Net profit
stock_list = []  # stock list
special_info = []  # 特别注意的事项
special_code = ""


class MyStrategy(bt.Strategy):
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

    def is_special(self, special_date, txt):
        yesterday_spe = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
        if datetime.datetime.strftime(special_date, "%Y-%m-%d") == yesterday_spe:
            special_info.append({'date': self.datas[0].datetime.date(0),
                                 'code': special_code,
                                 'info': txt})

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
        self.cross_list = []  # judge the date before golden cross and date after dead cross
        #self.cross_1 = bt.ind.CrossUp(self.crossover)
        #self.cross_2 = bt.ind.CrossUp(self.sma2)

    # order statement information
    def notify_order(self, order):
        txt = ""
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 检查订单是否完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.crossover_buy = True
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
                self.pos = self.getposition(self.data).size
                self.buy_cost = order.executed.value
                self.valued = self.broker.getvalue()
                txt = 'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, fund Value %.2f, pos Size %.2f' % \
                      (order.executed.price,
                        order.executed.value,
                        order.executed.comm,
                        self.valued,
                        self.pos)
                self.log(txt)
            elif order.issell():
                self.crossover_sell = True
                self.pos = self.getposition(self.data).size
                txt = 'SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, fund Value %.2f, pos Size %.2f' % \
                      (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          self.valued,
                          self.pos)
                self.log(txt)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Canceled:
                self.log('order cancel!')
            elif order.status == order.Margin:
                self.log('fund not enough!')
            elif order.status == order.Rejected:
                self.log('order reject!')
        self.order = None
        if self.crossover_buy:
            self.is_special(self.datas[0].datetime.date(0), txt)
        elif self.crossover_sell:
            self.is_special(self.datas[0].datetime.date(0), txt)
        self.crossover_buy = False
        self.crossover_sell = False

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('business profit: %.2f, Net profit: %.2f' % (trade.pnl, trade.pnlcomm))

    #  loop in every business day
    def next(self):
        size = self.getposition(self.data).size
        price = self.getposition(self.data).price
        self.cash_valid = self.broker.getcash()
        self.valued = self.broker.getvalue()
        self.buy_comm = price * size * COMM_VALUE
        if not self.position:  # Outside, buy
            if self.crossover > 0:  # if golden cross, valid=datetime.datetime.now() + datetime.timedelta(days=3)
                self.log('Available Cash: %.2f, Total fund: %.2f' % (self.cash_valid, self.valued))
                self.order = self.buy()
                self.is_special(self.datas[0].datetime.date(0),
                                'Outside, golden cross buy, close: %.2f，Total fund：%.2f' %
                                (self.data_close[0], self.valued))
                self.log('Outside, golden cross buy, close: %.2f，Total fund：%.2f' %
                         (self.data_close[0], self.valued))
        else:
            if self.crossover > 0:
                if (self.cash_valid - price * STAKE - self.buy_comm) > 0:
                    self.log('Available Cash: %.2f, Total fund: %.2f' % (self.cash_valid, self.valued))
                    self.order = self.buy()
                    self.is_special(self.datas[0].datetime.date(0),
                                    'Outside, golden cross buy, close: %.2f，Total fund：%.2f' %
                                    (self.data_close[0], self.valued))
                    self.log('Outside, golden cross buy, close: %.2f' % self.data_close[0])
            elif self.crossover < 0:  # Inside and dead cross
                if self.valued > START_CASH * 1.03:
                    self.order = self.close(size=size)
                    self.is_special(self.datas[0].datetime.date(0),
                                    'Inside dead cross, sell, close: %.2f，Total fund：%.2f' %
                                    (self.data_close[0], self.valued))
                    self.log('Inside dead cross, sell, close:  %.2f，Total fund：%.2f' %
                             (self.data_close[0], self.valued))


def get_codes(file_name):
    file = None
    try:
        path = os.getcwd() + '\\' + file_name
        print(path + '\n')
        file = open(path, 'r')
        return file.readlines()
    finally:
        if file is not None:
            file.close()


def get_file(f_code):
    basic_path = os.getcwd()
    code_file = str(basic_path + '\\' + f'{f_code}.csv')
    return code_file


def get_sh_stock(s_code):
    s_code = s_code[2:]
    df = ak.stock_individual_info_em(symbol=s_code)
    return df


def get_stocks():
    filename = "text.txt"
    if MyStrategy.params.pslow == 20:
        filename = "text520.txt"
    elif MyStrategy.params.pslow == 30:
        filename = "text530.txt"
    return filename


# 准备历史数据做预测评估
def prepare_data(f_code, f_startdate, f_enddate):
    csv_file = str(get_file(f_code))
    print("file的title信息：" + csv_file)
    get_sh_stock(f_code)
    file = open(csv_file, 'w', encoding='utf-8')
    # 默认返回不复权的数据; qfq: 返回前复权后的数据; hfq: 返回后复权后的数据; hfq-factor: 返回后复权因子; qfq-factor: 返回前复权因子
    stock_hfq_df = ak.stock_zh_a_daily(symbol=f_code, start_date=f_startdate, end_date=f_enddate,
                                       adjust="qfq")  # 接口参数格式 股票代码必须含有sh或zz的前缀
    if stock_hfq_df is None:
        print("Warning, run_strategy: stock_hfq_df is None!")
    else:
        stock_hfq_df.to_csv(file, encoding='utf-8')
        file.close()
    return csv_file


def get_consider(filepath):
    df = pd.read_csv(filepath, parse_dates=True, index_col="date")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", utc=True)
    it = 0
    max_list = []
    min_list = []
    close_list = []
    while it < len(df):
        max_list.append(df['high'].values[it])
        min_list.append(df['low'].values[it])
        close_list.append(df['close'].values[it])
        it += 1
    max_median = np.median(max_list)
    max_value = np.max(max_list)
    min_median = np.median(min_list)
    min_value = np.min(min_list)
    close_median = np.median(close_list)
    # print(f"initial cost: {START_CASH} \nPeriod：{startdate}:{enddate}")
    print('Max median value: %.2f, max value: %.2f' % (max_median, max_value))
    print('Min median value: %.2f, min value: %.2f' % (min_median, min_value))
    print('Close median value: %.2f' % close_median)
    return df


# 从指定文件中读取数据，并运行回测函数
def run_strategy(f_startdate, f_enddate, f_file):
    df = get_consider(f_file)
    from_date = datetime.datetime.strptime(f_startdate, "%Y%m%d")
    end_date = datetime.datetime.strptime(f_enddate, "%Y%m%d")
    data = bt.feeds.PandasData(dataname=df, fromdate=from_date, todate=end_date)  # 加载数据
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()  # 初始化回测系统
    cerebro.adddata(data)  # 将数据传入回测系统
    #start_cash = START_CASH
    cerebro.broker.setcash(START_CASH)  # set initial fund
    cerebro.broker.setcommission(commission=COMM_VALUE)  # set trad rate 0.2%
    stake = STAKE
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)  # set trade volume
    cerebro.addstrategy(MyStrategy)  # period = [(5, 10), (20, 100), (2, 10)]) , 运行策略
    cerebro.run(maxcpus=1)  # 运行回测系统

    port_value = cerebro.broker.getvalue()  # trade over, get total fund
    pnl = port_value - START_CASH  # figure out profit
    if pnl is None:
      stock_pnl.append(0)
    else:
        stock_pnl.append(pnl)
    print('Begin Fund: %.2f, Total Cash: %.2f' % (START_CASH, round(port_value, 2)))
    print(f"Net Profit: {round(pnl, 2)}\n\n")
    # cerebro.plot(style='candlestick')  # 画图


# sorted by strategy result
def stock_rank():
    i = 0
    data_stock = []
    while i < len(stock_list):
        data_stock.append({'name': stock_list[i], 'pnl': round(float(stock_pnl[i]), 2)})
        i += 1
    # 字典序列排序
    data_stock.sort(key=lambda x: -x['pnl'])
    # 按照收益从高到低排序可选对象
    j = 0
    pnl = 0
    comm_cash = 0
    rate_profit = 0
    while j < len(data_stock):
        # pnl = int(data_stock[j]['pnl']) + pnl
        if int(data_stock[j]['pnl']) > 0:  # 收益额为正的收益总额
            pnl = int(data_stock[j]['pnl']) + pnl
            comm_cash = comm_cash + START_CASH
        print(f"{data_stock[j]}")
        j += 1
    rate_profit = round(pnl/comm_cash * 100, 2)
    print(f"截止到{datetime.date.today()}日的正向profit total pnl: {pnl}, profit rate: {rate_profit}%\n")


if __name__ == '__main__':
    plt.switch_backend('agg')
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # set Chinese to draw picture
    plt.rcParams["axes.unicode_minus"] = False  # 设置画图时的负号显示
    codes = get_codes(get_stocks())
    startdate = str(codes[0]).replace('\n', '')  # 回测开始时间
    enddate = str(codes[1]).replace('\n', '')  # 回测结束时间
    it = 0
    code = ""
    for code in codes:
        if it > 1:
            code = str(codes[it]).replace('\n', '')  # "sz300598"
            special_code = code
            filepath = prepare_data(code, startdate, enddate)
            file_size = os.path.getsize(filepath)
            if file_size > 0:
                code_value = get_sh_stock(code)
                # print(code_value)
                print(f"code: {code_value.value[4]}, name：{code_value.value[5]}")
                stock_list.append(code_value.value[5])
                run_strategy(startdate, enddate, filepath)
            else:
                break
        it += 1
    stock_rank()  # 列出优选对象
    print(f"Test Date: {datetime.date.today()}")
    print(f"Initial Fund: {START_CASH}\nPeriod：{startdate}~{enddate}")
    # set special operation in the special date
    it2 = 0
    code = ""
    code_name = ""
    while it2 < len(special_info):
        code_value = get_sh_stock(special_info[it2]['code'])
        code_name = code_value.values[5][1]
        code = code_value.values[4][1]
        print(f"\n[Specal opt is :{code_name}]")
        print('%s, %s' % (datetime.date.strftime(special_info[it2]['date'], "%Y-%m-%d"), special_info[it2]['info']))
        if str(code)[0] == "6":
            code = "sh" + code
        else:
            code = "sz" + code
        filepath = get_file(code)  # code needs sh or sz
        df = get_consider(filepath)
        it2 += 1

