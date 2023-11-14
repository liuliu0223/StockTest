#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import datetime
import os
import backtrader as bt
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import MyStrategy.StrategyMini as smini
# const value
STAKE = 1500  # volume once
START_CASH = 150000  # initial cost
COMM_VALUE = 0.002   # 费率
WIN_ENV_FLAG = False  # windows环境设置
FILEDIR = 'stocks'
CODES_FILE = 'text.txt'

# globle value
stock_pnl = []  # Net profit
stock_list = []  # stock list
special_info = []  # 特别注意的事项
special_code = ""


class MyStock:
    # 通用日志打印函数，可以打印下单、交易记录，非必须，可选
    def log(self, txt, dt=None):
        dt = dt or self.date
        print('MyStock：%s , %s' % (dt.isoformat(), txt))

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

    def get_df(self, code):
        s_code = self.getcodebytype(code, ctype='String')
        file_path = get_file(s_code)
        df = pd.read_csv(file_path, parse_dates=True, index_col='date')
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', utc=True)
        return df

    def get_stock_by_date(self, s_code, s_date):
        code = self.getcodebytype(s_code, ctype='Numberal')
        sdf = ak.stock_individual_info_em(symbol=code)  # code begin with numeral
        self.name = sdf.values[5][1]
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
                '''
                self.log('Stock: %s, code: %s, date: %s, open: %.2f, close: %.2f, high: %.2f, low: %.2f, volume: %.2f, \
                 outstanding_share: %.2f, turnover: %.2f' % (self.name, self.code, self.date, self.open, self.close,
                                                             self.high, self.low, self.volume, self.outstanding_share,
                                                             self.turnover)) 
                '''
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
        operator = ""
        if txt.lower().find('buy') > 0:
            operator = 'BUY'
        elif txt.lower().find('sell') > 0:
            operator = 'SELL'
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        yesterday_spe = get_business_day(today, days=-1)
        if datetime.datetime.strftime(special_date, "%Y-%m-%d") == yesterday_spe:
            special_info.append({'date': yesterday_spe,
                                 'code': special_code,
                                 'info': txt,
                                 'operator': operator})
        elif datetime.datetime.strftime(special_date, "%Y-%m-%d") == today:
            special_info.append({'date': today,
                                 'code': special_code,
                                 'info': txt,
                                 'operator': operator})

    # 初始化函数，初始化属性、指标的计算，only once time
    def __init__(self):
        self.data_close = self.datas[0].close  # close data
        self.date = self.datas[0].datetime.date(0)
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
                                    'Inside dead cross, sell, close: %.2f，Total fund：%.2f, pos: %.2f' %
                                    (self.data_close[0], fund, size))
                    self.log('Inside dead cross, sell, close:  %.2f，Total fund：%.2f, pos: %.2f' %
                             (self.data_close[0], fund, size))


def get_work_path(pack_name):
    return os.path.join(os.getcwd(), pack_name)
    #print(f"get_work_path: os.getcwd=%s, \npack_name=%s, get_work_path=%s\n" % (os.getcwd(), pack_name, work_path))


def get_codes(name):
    file = None
    try:
        path = os.path.join(get_work_path(""), name)
        print(path + '\n')
        file = open(path, 'r')
        return file.readlines()
    finally:
        if file is not None:
            file.close()


def get_file(f_code):
    code_file = os.path.join(get_work_path(FILEDIR), f'{f_code}.csv')
    return code_file


# This is to get the stock code style
# function: getcodebytype
# input:
#     code(String)
#     ctype(String):ctype='Numeral', means return string begin with Numeral, such as 600202;
#                   ctype='String' , means return string with character, such as sh600202;
#                   ctype=None, means return string with character, such as sh600202
# return: stock code (string)
def getcodebytype(code, ctype):
    s_code = code
    num_code_type = False
    if len(code) == 6:
        num_code_type = True

    if ctype is None:
        if num_code_type:
            if "6" == code[:1]:
                s_code = "sh" + code
            else:
                s_code = "sz" + code
    elif ctype == 'Numeral':
        if num_code_type:
            return s_code
        else:
            return s_code[2:]
    elif ctype == 'SpecialString':
        if num_code_type:
            if "6" == code[:1]:
                s_code = "sh." + code
            else:
                s_code = "sz." + code
        else:
            if s_code.find('.') < 0:
                s_code = s_code[:2] + '.' + s_code[2:]
    else:
        if num_code_type:
            if "6" == code[:1]:
                s_code = "sh" + code
            else:
                s_code = "sz" + code
        elif len(s_code) > 8:
            s_code = s_code[:2] + s_code[3:]
    return s_code


# calculate the business date before one date, if the date is weekend, the get the nearest business date before
# days=-1: yesterday, days=-2: the day before yesterday
# return date string type
def get_business_day(s_date, days=-1):
    before_date = "1900-01-01"
    d_days = 0
    # s_date = '2023-09-04'  0Monday1Tuesday2Wednesday3Thursday4Friday5Saturday6Sunday
    week_days = int(datetime.datetime.strptime(s_date, "%Y-%m-%d").weekday())
    if days is not None:
        d_days = days
    if week_days == 0:
        d_days = d_days - 2
        before_date = (datetime.datetime.strptime(s_date, "%Y-%m-%d") +
                       datetime.timedelta(d_days)).strftime("%Y-%m-%d")
    elif week_days == 6:
        d_days = d_days - 1
        before_date = (datetime.datetime.strptime(s_date, "%Y-%m-%d") +
                       datetime.timedelta(d_days)).strftime("%Y-%m-%d")
    else:
        before_date = (datetime.datetime.strptime(s_date, "%Y-%m-%d") +
                       datetime.timedelta(d_days)).strftime("%Y-%m-%d")
    return before_date


def get_sh_stock(s_code):   # stock code must be begin with numeral, s_code=600602
    code = getcodebytype(s_code, ctype='Numeral')
    df = ak.stock_individual_info_em(symbol=code)
    return df


def get_stocks():
    filename = CODES_FILE
    '''
    if MyStrategy.params.pslow == 20:
        filename = "text520.txt"
    elif MyStrategy.params.pslow == 30:
        filename = "text530.txt"
    '''
    return filename


# 准备历史数据做预测评估
def prepare_data(f_code, f_startdate, f_enddate):
    csv_file = str(get_file(f_code))
    print("prepare_data: file path=" + csv_file)
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


# get the stock information from csv file, then to calculate the stock's min, max, std and mean value
def get_consider(f_filepath):
    file_path = f_filepath
    sdf = pd.read_csv(file_path, parse_dates=True, index_col='date')
    sdf.index = pd.to_datetime(sdf.index, format="%Y-%m-%d", utc=True)

    max_list = sdf['high'].values.tolist()
    close_list = sdf['close'].values.tolist()
    min_list = sdf['low'].values.tolist()

    max_median = np.median(max_list)
    max_value = np.max(max_list)
    max_std = np.std(max_list)
    max_mean = np.mean(max_list)
    min_median = np.median(min_list)
    min_value = np.min(min_list)
    min_std = np.std(min_list)
    min_mean = np.mean(min_list)
    close_median = np.median(close_list)
    close_max = np.max(close_list)
    close_min = np.min(close_list)
    close_std = np.std(close_list)
    close_mean = np.mean(close_list)
    # print(f"initial cost: {START_CASH} \nPeriod：{startdate}:{enddate}")
    print('Max median value: %.2f, max value: %.2f, max std: %.2f, max mean: %.2f' %
          (max_median, max_value, max_std, max_mean))
    print('Min median value: %.2f, min value: %.2f, min std: %.2f, min mean: %.2f' %
          (min_median, min_value, min_std, min_mean))
    print('Close median value: %.2f, Close max: %.2f, Close min: %.2f, Close std: %.2f, Close mean: %.2f' %
          (close_median, close_max, close_min, close_std, close_mean))
    return sdf, close_mean


# 从指定文件中读取数据，并运行回测函数
def run_strategy(f_startdate, f_enddate, code):
    filepath = prepare_data(code, startdate, enddate)
    sdf, close_mean = get_consider(filepath)
    from_date = datetime.datetime.strptime(f_startdate, "%Y%m%d")
    end_date = datetime.datetime.strptime(f_enddate, "%Y%m%d")
    data = bt.feeds.PandasData(dataname=sdf, fromdate=from_date, todate=end_date)  # 加载数据
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()  # 初始化回测系统
    cerebro.adddata(data)  # 将数据传入回测系统
    cerebro.broker.setcash(START_CASH)  # set initial fund
    cerebro.broker.setcommission(commission=COMM_VALUE)  # set trad rate 0.2%
    stake = STAKE
    if close_mean > 90:
        stake = 500
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
    #cerebro.plot(style='candlestick')  # 画图


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


def market_info(date):
    date_ = str(date) # date type："20211227"
    stock_szse_deal_daily_df = ak.stock_sse_deal_daily(date_)
    stock_sse_deal_df = ak.stock_szse_summary(date_)
    print(stock_szse_deal_daily_df)
    print(stock_sse_deal_df)
    # 涨停股池
    stock_zt_pool_previous_em_df = ak.stock_zt_pool_previous_em(date=date_)
    print(stock_zt_pool_previous_em_df)
    # 跌停股池
    stock_zt_pool_dtgc_em_df = ak.stock_zt_pool_dtgc_em(date=date_)
    print(stock_zt_pool_dtgc_em_df)
    return stock_zt_pool_previous_em_df, stock_zt_pool_dtgc_em_df


if __name__ == '__main__':
    plt.switch_backend('agg')
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # set Chinese to draw picture
    plt.rcParams["axes.unicode_minus"] = False  # 设置画图时的负号显示
    codes = get_codes(get_stocks())
    startdate = str(codes[0]).replace('\n', '')  # 回测开始时间 date type："20211227"
    enddate = str(codes[1]).replace('\n', '')  # 回测结束时间 date type："20211227"

    it = 0
    code = ""
    Special_ops = []
    result_list = []
    # 第一段：买卖回测模拟
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
                (dif, dea, hist, Special_ops) = smini.computeMACD(code, startdate, enddate)
                if len(Special_ops) > 0:
                    result_list.append(Special_ops)
                stock_list.append(code_value.value[5])
                run_strategy(startdate, enddate, code)
            else:
                break
        it += 1
    # 第二段：优选股
    stock_rank()  # 列出优选对象
    # 第三段：特殊操作提醒 SMA均线长短期交叉买卖提示
    print(f"Test Date: {datetime.date.today()}")
    print(f"Initial Fund: {START_CASH}, Stack: {STAKE}\nPeriod：{startdate}~{enddate}")

    my_stock = MyStock()
    # MACD均线 操作提醒
    print("MACD info Begin:")
    it5 = 0
    tmp_date = datetime.date.today()
    ss_date = datetime.datetime.strftime(tmp_date, "%Y-%m-%d")
    '''
    if datetime.datetime.strptime(enddate, "%Y-%m-%d") - tmp_date > 0:
        ss_date = datetime.datetime.strftime(tmp_date, "%Y-%m-%d")
    else:
        ss_date = str(enddate[:4] + "-" + enddate[4:6] + "-" + enddate[6:])
        '''
    b4_s_date = get_business_day(ss_date, days=-1)
    while it5 < len(result_list) & len(result_list) > 0:
        tmp_result = result_list[it5]
        it6 = 0
        while it6 < len(tmp_result):
            c_code = getcodebytype(tmp_result[it6]['code'], "")
            c_date = tmp_result[it6]['date']
            c_msg = tmp_result[it6]['msg']
            if c_date == b4_s_date:
                code_value = get_sh_stock(c_code)  # code include sh
                code_name = code_value.values[5][1]
                #print(f"MACD Date: {c_date}")
                print(f"{code_name} : {c_msg}")
            it6 += 1
        it5 += 1
    print("MACD info End!\n")
    # set special operation in the special date
    it2 = 0
    code = ""
    code_name = ""
    while it2 < len(special_info):
        code_value = get_sh_stock(special_info[it2]['code'])   # code include sh
        code_name = code_value.values[5][1]
        code = code_value.values[4][1]
        s_date = special_info[it2]['date']
        b4_s_date = get_business_day(s_date, days=-1)
        b4_s_date_info = my_stock.get_stock_by_date(code, b4_s_date)
        print(f"\n[Specal opt is :{code_name}]")
        print('%s, %s' % (s_date, special_info[it2]['info']))
        print(f"{b4_s_date}, open: {b4_s_date_info['open']}, close: {b4_s_date_info['close']}, "
              f"high: {b4_s_date_info['high']}, low: {b4_s_date_info['low']}")
        s_code = getcodebytype(code, ctype='String')
        # MACD均线 操作提醒
        it3 = 0
        while it3 < len(result_list) & len(result_list) > 0:
            tmp_result = result_list[it3]
            it4 = 0
            while it4 < len(tmp_result):
                c_code = getcodebytype(tmp_result[it4]['code'], "")
                c_date = tmp_result[it4]['date']
                c_msg = tmp_result[it4]['msg']
                b4_s_date = get_business_day(s_date, days=-1)
                b4_s_date_info = my_stock.get_stock_by_date(code, b4_s_date)
                if c_code == s_code and c_date == s_date:
                    b4_s_date = get_business_day(c_date, days=-1)
                    b4_s_date_info = my_stock.get_stock_by_date(code, b4_s_date)
                    print('%s, %s' % (c_date, c_msg))
                it4 += 1
            it3 += 1
        # MACD均线 操作提醒结束
        filepath = get_file(s_code)  # code needs sh or sz
        # 第四段：均值，极值提示
        df = get_consider(filepath)
        it2 += 1

'''
# 当天股市涨跌池情况
    print("Stock Market analysis begin! \n")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    yesterday_spe = get_business_day(today, days=-1)
    yesterday = yesterday_spe.replace("-", "")
    #yesterday = "20231024"
    upstocks, downstocks = market_info(yesterday)
    print("Stock Market analysis end!")
'''
