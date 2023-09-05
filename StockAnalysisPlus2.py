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
STAKE = 1500  # volume once
START_CASH = 150000  # initial cost
COMM_VALUE = 0.002   # 费率
WIN_ENV_FLAG = True  # windows环境设置
FILEDIR = "stocks"

# globle value
stock_pnl = []  # Net profit
stock_list = []  # stock list
special_info = []  # 特别注意的事项
special_code = ""


class MyStock:
    # 通用日志打印函数，可以打印下单、交易记录，非必须，可选
    def log(self, txt, dt=None):
        s_date = datetime.datetime.now().strftime("%Y-%m-%d")
        print('%s , %s' % (s_date, txt))

    # 初始化函数，初始化属性、指标的计算，only once time
    def __init__(self):
        self.code = ""  # stock code
        self.name = ""  # stock name
        self.date = '1900-01-01'
        self.open = 0.0
        self.close = 0.0
        self.high = 0.0
        self.low = 0.0
        self.volume = 0.0
        self.outstanding_share = 0
        self.turnover = 0

    def getcodebytype(self, code, ctype='Numberal'):  # ctype='Numeral', 600202; ctype='String' sh600202
        s_code = ""
        num_code_type = False
        if len(code) == 6:
            num_code_type = True
            if "6" == code[:1]:
                s_code = "sh" + code
            else:
                s_code = "sz" + code
        else:
            s_code = code
        if ctype == 'Numeral':
            if num_code_type:
                return code
            else:
                return code[2:]
        else:
            return s_code

    def get_df(self, code):
        s_code = self.getcodebytype(code, ctype='String')
        file_path = get_file(s_code)
        df = pd.read_csv(file_path, parse_dates=True, index_col='date')
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', utc=True)
        return df

    def get_stock_by_date(self, s_code, s_date):
        sdf = ak.stock_individual_info_em(symbol=s_code[2:])  # code begin with numeral
        self.name = sdf.values[5]
        self.code = s_code
        df = self.get_df(s_code)
        i = 0
        stock_info = None
        while i < len(df):
            t_date = df['open'].index[i]
            if datetime.datetime.strftime(t_date, "%Y-%m-%d") == s_date:
                self.date = s_date
                self.open = df['open'].values[i]
                self.close = df['close'].values[i]
                self.high = df['high'].values[i]
                self.low = df['low'].values[i]
                self.volume = df['volume'].values[i]
                self.outstanding_share = df['outstanding_share'].values[i]
                self.turnover = df['turnover'].values[i]
                stock_info = {'date': self.date, 'open': self.open, 'close': self.close, 'high': self.high,
                              'low': self.low, 'volume': self.volume, 'outstanding_share': self.outstanding_share,
                              'turnover': self.turnover}
                self.log('Stock: %s, code: %s, date: %s, open: %.2f, close: %.2f, high: %.2f, low: %.2f, volume: %.2f, \
                 outstanding_share: %.2f, turnover: %.2f' % (self.name, self.code, self.date, self.open, self.close,
                                                             self.high, self.low, self.volume, self.outstanding_share,
                                                             self.turnover))
                break
            i += 1
        return stock_info


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
        my_stock = MyStock()
        operator = ""
        consider_date = ""
        if txt.lower().find('buy') > 0:
            operator = 'BUY'
        elif txt.lower().find('sell') > 0:
            operator = 'SELL'
        yesterday_spe = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
        # check the date before cross date
        before_yesterday = (datetime.datetime.now() + datetime.timedelta(days=-2)).strftime("%Y-%m-%d")
        history_date = [yesterday_spe, before_yesterday]
        i = 0
        while i < len(history_date):
            temp_date = int(datetime.datetime.strptime(history_date[i], "%Y-%m-%d").weekday())
            if temp_date > 5:
                consider_date = (datetime.datetime.strptime(history_date[i], "%Y-%m-%d") +
                                 datetime.timedelta(days=-3)).strftime("%Y-%m-%d")
                history_date[i] = consider_date
            i += 1

        if datetime.datetime.strftime(special_date, "%Y-%m-%d") == history_date[0]:
            special_info.append({'date': consider_date,
                                 'code': special_code,
                                 'info': txt,
                                 'operator': operator})
            stock_info = my_stock.get_stock_by_date(special_code, history_date[1])
            if operator == 'BUY':
                txt = 'Before golden cross and BUY operation, \nstock info: ' \
                      'open: %.2f, close: %.2f, high: %.2f, low: %.2f' % \
                      (stock_info['open'], stock_info['close'], stock_info['high'], stock_info['low'])
                special_info.append({'date': history_date[1],
                                     'code': special_code,
                                     'info': txt,
                                     'operator': operator})
            elif operator == 'SELL':
                txt = 'Before dead cross and SELL operation, \nstock info: \n' \
                      'open: %.2f, close: %.2f, high: %.2f, low: %.2f' % \
                      (stock_info['open'], stock_info['close'], stock_info['high'], stock_info['low'])
                special_info.append({'date': history_date[1],
                                     'code': special_code,
                                     'info': txt,
                                     'operator': operator})

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
        valid_cash = self.broker.getcash()
        fund = self.broker.getvalue()
        buy_comm = price * size * COMM_VALUE
        if not self.position:  # Outside, buy
            if self.crossover > 0:  # if golden cross, valid=datetime.datetime.now() + datetime.timedelta(days=3)
                self.log('Available Cash: %.2f, Total fund: %.2f, pos: %.2f' % (valid_cash, fund, size))
                self.order = self.buy()
                txt = 'Outside, golden cross buy, close: %.2f，Total fund：%.2f, pos: %.0f' % \
                      (self.data_close[0], fund, size)
                self.is_special(self.datas[0].datetime.date(0), txt)
                self.log('Outside, golden cross buy, close: %.2f，Total fund：%.2f, pos: %.2f' %
                         (self.data_close[0], fund, size))
        else:
            if self.crossover > 0:
                if (valid_cash - price * STAKE - buy_comm) > 0:
                    self.log('Available Cash: %.2f, Total fund: %.2f, pos: %.2f' % (valid_cash, fund, size))
                    self.order = self.buy()
                    self.is_special(self.datas[0].datetime.date(0),
                                    'Outside, golden cross buy, close: %.2f，Total fund：%.2f， pos: %.2f' %
                                    (self.data_close[0], valid_cash, size))
                    self.log('Outside, golden cross buy, close: %.2f，Total fund：%.2f， pos: %.2f' %
                             (self.data_close[0], valid_cash, size))
            elif self.crossover < 0:  # Inside and dead cross
                if fund > START_CASH * 1.03:
                    self.order = self.close(size=size)
                    self.is_special(self.datas[0].datetime.date(0),
                                    'Inside dead cross, sell, close: %.2f，Total fundda：%.2f, pos: %.2f' %
                                    (self.data_close[0], fund, size))
                    self.log('Inside dead cross, sell, close:  %.2f，Total fund：%.2f, pos: %.2f' %
                             (self.data_close[0], fund, size))


def get_work_path(pack_name):
    if pack_name == "":
        if WIN_ENV_FLAG:
            return os.getcwd() + '\\'
        else:
            return os.getcwd() + '/'
    else:
        if WIN_ENV_FLAG:
            return str(os.getcwd() + '\\' + pack_name + '\\').strip()
        else:
            return str(os.getcwd() + '/' + pack_name + '/').strip()


def get_codes(file_name):
    file = None
    try:
        path = get_work_path("") + file_name
        print(path + '\n')
        file = open(path, 'r')
        return file.readlines()
    finally:
        if file is not None:
            file.close()


def get_file(f_code):
    code_file = str(get_work_path(FILEDIR) + f'{f_code}.csv')
    return code_file


def getcodebytype(code, ctype='Numberal'):  # ctype='Numeral', 600202; ctype='String' sh600202
    s_code = ""
    num_code_type = False
    if len(code) == 6:
        num_code_type = True
        if "6" == code[:1]:
            s_code = "sh" + code
        else:
            s_code = "sz" + code
    else:
        s_code = code
    if ctype == 'Numberal':
        if num_code_type:
            return code
        else:
            return code[2:]
    else:
        return s_code


def get_sh_stock(s_code):   # stock code mustbe begin with numeral
    code = getcodebytype(s_code, ctype='Numberal')
    df = ak.stock_individual_info_em(symbol=code)
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
    get_sh_stock(f_code)  # 去掉代码前缀 sh or sz
    file = open(csv_file, 'w', encoding='utf-8')
    # 默认返回不复权的数据; qfq: 返回前复权后的数据; hfq: 返回后复权后的数据; hfq-factor: 返回后复权因子; qfq-factor: 返回前复权因子
    stock_hfq_df = ak.stock_zh_a_daily(symbol=f_code, start_date=f_startdate, end_date=f_enddate,
                                       adjust="qfq")  # 接口参数格式 股票代码必须含有sh或sz的前缀
    if stock_hfq_df is None:
        print("Warning, run_strategy: stock_hfq_df is None!")
    else:
        stock_hfq_df.to_csv(file, encoding='utf-8')
        file.close()
    return csv_file


def get_consider(f_filepath):
    file_path = f_filepath
    sdf = pd.read_csv(file_path, parse_dates=True, index_col='date')
    sdf.index = pd.to_datetime(sdf.index, format="%Y-%m-%d", utc=True)
    it = 0
    max_list = []
    min_list = []
    close_list = []
    while it < len(sdf):
        max_list.append(sdf['high'].values[it])
        min_list.append(sdf['low'].values[it])
        close_list.append(sdf['close'].values[it])
        it += 1
    max_median = np.median(max_list)
    max_value = np.max(max_list)
    max_std = np.std(max_list)
    min_median = np.median(min_list)
    min_value = np.min(min_list)
    min_std = np.std(min_list)
    close_median = np.median(close_list)
    close_max = np.max(close_list)
    close_min = np.min(close_list)
    close_std = np.std(close_list)
    # print(f"initial cost: {START_CASH} \nPeriod：{startdate}:{enddate}")
    print('Max median value: %.2f, max value: %.2f, max std: %.2f' % (max_median, max_value, max_std))
    print('Min median value: %.2f, min value: %.2f, min std: %.2f' % (min_median, min_value, min_std))
    print('Close median value: %.2f, Close max: %.2f, Close min: %.2f, Close std: %.2f' %
          (close_median, close_max, close_min, close_std))
    return sdf


# 从指定文件中读取数据，并运行回测函数
def run_strategy(f_startdate, f_enddate, f_file):
    sdf = get_consider(f_file)
    from_date = datetime.datetime.strptime(f_startdate, "%Y%m%d")
    end_date = datetime.datetime.strptime(f_enddate, "%Y%m%d")
    data = bt.feeds.PandasData(dataname=sdf, fromdate=from_date, todate=end_date)  # 加载数据
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()  # 初始化回测系统
    cerebro.adddata(data)  # 将数据传入回测系统
    cerebro.broker.setcash(START_CASH)  # set initial fund
    cerebro.broker.setcommission(commission=COMM_VALUE)  # set trad rate 0.2%
    cerebro.addsizer(bt.sizers.FixedSize, stake=STAKE)  # set trade volume
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
    pnl = 0.0
    comm_cash = 0
    rate_profit = 0
    while j < len(data_stock):
        # pnl = int(data_stock[j]['pnl']) + pnl
        if int(data_stock[j]['pnl']) > 0:  # 收益额为正的收益总额
            pnl = int(data_stock[j]['pnl']) + pnl
            comm_cash = comm_cash + START_CASH
        print(f"{data_stock[j]}")
        j += 1
    if pnl > 0:
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
    print(f"Initial Fund: {START_CASH}, Stack: {STAKE}\nPeriod：{startdate}~{enddate}")
    # set special operation in the special date
    it2 = 0
    code = ""
    code_name = ""
    while it2 < len(special_info):
        code_value = get_sh_stock(special_info[it2]['code'])   #  code include sh
        code_name = code_value.values[5][1]
        code = code_value.values[4][1]
        print(f"\n[Specal opt is :{code_name}]")
        print('%s, %s' % (special_info[it2]['date'], special_info[it2]['info']))
        s_code = getcodebytype(code, ctype='String')
        filepath = get_file(s_code)  # code needs sh or sz
        df = get_consider(filepath)
        it2 += 1

