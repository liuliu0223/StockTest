#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import datetime
import os
import backtrader as bt
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

STAKE = 1500  # 每次买入的数量
START_CASH = 150000  # 初始投资金额
COMM_VALUE = 0.002   # 手续费率
stock_pnl = []  # 净利润
stock_list = []  # 股票清单
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

    def is_specialday(self, specialdate, txt):
        #if datetime.datetime.strftime(specialdate, "%Y-%m-%d") == datetime.date.today():
        if datetime.datetime.strftime(specialdate, "%Y-%m-%d") == (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime("%Y-%m-%d"):
            special_info.append({'date': self.datas[0].datetime.date(0),
                                 'code': special_code,
                                 'info': txt})

    # 初始化函数，初始化属性、指标的计算，整个回测系统运行期间只执行一次
    def __init__(self):
        self.data_close = self.datas[0].close  # 指定价格序列
        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buy_price = None
        self.buy_comm = None  # 交易额
        self.buy_cost = None  # 成本
        self.pos = None  # 股票仓位
        self.cash_valid = None  # 可用资金
        self.valued = None  # 总收益
        self.pnl = None  # 利润
        self.sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        self.sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.dif = self.sma1 - self.sma2
        self.crossover = bt.ind.CrossOver(self.sma1, self.sma2)  # crossover signal
        self.crossover_buy = False
        self.crossover_sell = False

    # 订单状态消息通知函数
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
                txt = 'BUY EXECUTED, Price: %.2f, Cost: %.2f, 佣金Comm %.2f, 当前总资产Value %.2f, 仓位 pos %.2f' % (order.executed.price,
                        order.executed.value,
                        order.executed.comm,
                        self.valued,
                        self.pos)
                self.log(txt)
            elif order.issell():
                self.crossover_sell = True
                self.pos = self.getposition(self.data).size
                txt = 'SELL EXECUTED, Price: %.2f, Cost: %.2f, 佣金Comm %.2f, 当前总资产Value %.2f, 仓位Size %.2f' %(order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          self.valued,
                          self.pos)
                self.log(txt)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Canceled:
                self.log('订单取消')
            elif order.status == order.Margin:
                self.log('保证金不足')
            elif order.status == order.Rejected:
                self.log('拒绝')
        self.order = None
        if self.crossover_buy:
            self.is_specialday(self.datas[0].datetime.date(0), txt)
        elif self.crossover_sell:
            self.is_specialday(self.datas[0].datetime.date(0), txt)
        self.crossover_buy = False
        self.crossover_sell = False

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('交易利润, 毛利润 %.2f, 净利润 %.2f' % (trade.pnl, trade.pnlcomm))

    # 每个交易日都会依次循环调用
    def next(self):
        # 也可以直接获取持仓
        size = self.getposition(self.data).size
        price = self.getposition(self.data).price
        self.cash_valid = self.broker.getcash()
        self.valued = self.broker.getvalue()
        self.buy_comm = price * size * COMM_VALUE
        if not self.position:  # 不在场内，则可以买入
            if self.crossover > 0:  # 如果金叉,valid=datetime.datetime.now() + datetime.timedelta(days=3)
                self.log('当前可用资金: %.2f, 当前总资产: %.2f' % (self.cash_valid, self.valued))
                self.order = self.buy()
                self.is_specialday(self.datas[0].datetime.date(0), '不在场内，金叉,买入, 盘终价: %.2f' % self.data_close[0])
                self.log('不在场内，金叉,买入, 盘终价: %.2f' % self.data_close[0])
        else:
            if self.crossover > 0:
                if (self.cash_valid - price * STAKE - self.buy_comm) > 0:
                    self.log('当前可用资金: %.2f, 当前总资产: %.2f' % (self.cash_valid, self.valued))
                    self.order = self.buy()
                    self.is_specialday(self.datas[0].datetime.date(0), '不在场内，金叉,买入, 盘终价: %.2f' % self.data_close[0])
                    self.log('不在场内，金叉,买入, 盘终价: %.2f' % self.data_close[0])
            elif self.crossover < 0:  # 在场内，且死叉
                if self.valued > START_CASH * 1.03:
                    self.order = self.close(size=size)
                    self.is_specialday(self.datas[0].datetime.date(0),
                                       '在场内，且死叉, 卖出，盘终价: %.2f，当前账户价值：%.2f' %
                                       (self.data_close[0], self.valued))
                    self.log('在场内，且死叉, 卖出，盘终价: %.2f，当前账户价值：%.2f' %
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


def createFile(f_code):
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
    csv_file = str(createFile(f_code))
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


# 从指定文件中读取数据，并运行回测函数
def run_strategy(f_startdate, f_enddate, f_file):
    df = pd.read_csv(f_file, parse_dates=True, index_col="date")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", utc=True)
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()  # 初始化回测系统
    from_date = datetime.datetime.strptime(f_startdate, "%Y%m%d")
    end_date = datetime.datetime.strptime(f_enddate, "%Y%m%d")
    data = bt.feeds.PandasData(dataname=df, fromdate=from_date, todate=end_date)  # 加载数据
    cerebro.adddata(data)  # 将数据传入回测系统
    start_cash = START_CASH
    cerebro.broker.setcash(start_cash)  # 设置初始资本
    cerebro.broker.setcommission(commission=COMM_VALUE)  # 设置交易手续费为 0.2%
    stake = STAKE
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)  # 设置买入数量
    cerebro.addstrategy(MyStrategy)  # period = [(5, 10), (20, 100), (2, 10)])
    print('组合期初资金: %.2f' % cerebro.broker.getvalue())
    cerebro.run(maxcpus=1)  # 运行回测系统
    print('组合期末资金: %.2f' % cerebro.broker.getvalue())
    port_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
    pnl = port_value - start_cash  # 盈亏统计
    if pnl is None:
        stock_pnl.append(0)
    else:
        stock_pnl.append(pnl)
    print(f"总资金: {round(port_value, 2)}")
    print(f"净收益: {round(pnl, 2)}\n\n")
    # cerebro.plot(style='candlestick')  # 画图


# 对策略算出来的营收数据进行排序
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
    reback_per = 0
    while j < len(data_stock):
        #pnl = int(data_stock[j]['pnl']) + pnl
        if int(data_stock[j]['pnl']) > 0:  # 收益额为正的收益总额
            pnl = int(data_stock[j]['pnl']) + pnl
            comm_cash = comm_cash + START_CASH
        print(f"{data_stock[j]}")
        j += 1
    reback_per = pnl/comm_cash
    print(f"截止到{datetime.date.today()}日的正向收益 total pnl: {pnl}, 收益率：{reback_per} \n")


if __name__ == '__main__':
    plt.switch_backend('agg')
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置画图时的中文显示
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
                print(code_value)
                stock_list.append(code_value.value[5])
                run_strategy(startdate, enddate, filepath)
            else:
                break
        it += 1
    print(f"预测日: {datetime.date.today()}")
    print(f"初始资金: {START_CASH}\n回测期间：{startdate}:{enddate}")
    stock_rank()  # 列出优选对象
    # 列出关键交易日及关注信息
    it2 = 0
    while it2 < len(special_info):
        code_value = get_sh_stock(special_info[it2]['code'])
        code_name = code_value.values[5][1]
        print(f"\nSpecal opt is :{code_name}")
        print('%s, %s' % (datetime.date.strftime(special_info[it2]['date'], "%Y-%m-%d"), special_info[it2]['info']))
        it2 += 1

