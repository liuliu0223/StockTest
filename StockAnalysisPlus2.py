import datetime
import os
import backtrader as bt
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

STAKE = 1500
START_CASH = 150000
COMM_VALUE = 0.002

class MyStrategy(bt.Strategy):
    """
    主策略程序
    """
    params = dict(
        pfast=5,  # period for the fast moving average
        pslow=10  # period for the slow moving average
    )

    # 通用日志打印函数，可以打印下单、交易记录，非必须，可选
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

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
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.dif = sma1 - sma2
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    # 订单状态消息通知函数
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 检查订单是否完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
                self.pos = self.getposition(self.data).size
                self.buy_cost = order.executed.value
                self.valued = self.broker.getvalue()
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, 佣金Comm %.2f, 当前总资产Value %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm,
                     self.valued) + ", 仓位：" + str(self.pos))
            elif order.issell():
                self.pos = self.getposition(self.data).size
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, 佣金Comm %.2f, 当前总资产Value %.2f, 仓位Size %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          self.valued,
                          float(self.pos)))

            #self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Canceled:
                self.log('订单取消')
            elif order.status == order.Margin:
                self.log('保证金不足')
            elif order.status == order.Rejected:
                self.log('拒绝')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('交易利润, 毛利润 %.2f, 净利润 %.2f' % (trade.pnl, trade.pnlcomm))

    # 每个交易日都会依次循环调用
    #1） sma5下穿sma10 ；
    #2）当前收盘价格 > 1.01 * 买入价格并且当前（sma5 - sma10） < 上期(sma5 - sma10) * 0.95；
    #3）当前收盘价格 < 0.97 * 买入价格;
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
                self.log('不在场内，金叉,买入, 盘终价: %.2f' % self.data_close[0])
        else:
            if self.crossover > 0:
                if (self.cash_valid - price * STAKE - self.buy_comm) > 0:
                    self.log('当前可用资金: %.2f, 当前总资产: %.2f' % (self.cash_valid, self.valued))
                    self.order = self.buy()
                    self.log('不在场内，金叉,买入, 盘终价: %.2f' % self.data_close[0])
            elif self.crossover < 0:  # 在场内，且死叉
                if self.valued > START_CASH * 1.05:
                    self.order = self.close(size=size)
                    self.log('在场内，且死叉, 卖出，盘终价: %.2f，当前账户价值：%.2f' %
                             (self.data_close[0], self.valued))


def getCodes(file_name):
    try:
        path = os.getcwd() + '\\' + file_name
        print(path + '\n')
        file = open(path, 'r')
        return file.readlines()
    finally:
        file.close()


def createnewfile(code):
    basicPath = os.getcwd()
    codefile = os.getcwd() + '\\' + f'{code}.csv'
    files = os.listdir(basicPath)
    for file in files:
        if codefile == (basicPath+str(file.title())):
            print("file的title信息：" + file.title() + '\n')
            os.close(file)
    print("codefile信息：" + codefile + '\n')
    return codefile


def get_sh_stockinfo(s_code):
    s_code = s_code[2:]
    df = ak.stock_individual_info_em(symbol=s_code)
    print(df)
    '''
    if not df.empty:
        for iter in df.index:
            if str(df.item[iter]).find("股票简称") > -1:
                print(str(df.item[iter]) + ":" + df.value[iter])'''


def getcodesfile():
    filename = "text.txt"
    if MyStrategy.params.pslow == 20:
        filename = "text520.txt"
    elif MyStrategy.params.pslow == 30:
        filename = "textall.txt"
    return filename


# 准备历史数据做预测评估
def prepare_data(f_code, f_startdate, f_enddate):
    csvfile = createnewfile(f_code)
    get_sh_stockinfo(f_code)
    print("创建一个CSV文件：" + csvfile)
    file = open(csvfile, 'w', encoding='utf-8')
    # 默认返回不复权的数据; qfq: 返回前复权后的数据; hfq: 返回后复权后的数据; hfq-factor: 返回后复权因子; qfq-factor: 返回前复权因子
    stock_hfq_df = ak.stock_zh_a_daily(symbol=f_code, start_date=f_startdate, end_date=f_enddate,
                                       adjust="qfq")  # 接口参数格式 股票代码必须含有sh或zz的前缀
    if stock_hfq_df is None:
        print("Warning, run_strategy: stock_hfq_df is None!")
    else:
        stock_hfq_df.to_csv(file, encoding='utf-8')
        file.close()
    return csvfile


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
    cerebro.broker.setcash(start_cash)  # 设置初始资本为 100000
    cerebro.broker.setcommission(commission=COMM_VALUE)  # 设置交易手续费为 0.2%
    stake = STAKE
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)  # 设置买入数量
    cerebro.addstrategy(MyStrategy)  # period = [(5, 10), (20, 100), (2, 10)])
    print('组合期初资金: %.2f' % cerebro.broker.getvalue())
    cerebro.run(maxcpus=1)  # 运行回测系统
    print('组合期末资金: %.2f' % cerebro.broker.getvalue())
    port_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
    pnl = port_value - start_cash  # 盈亏统计
    print(f"初始资金: {start_cash}\n回测期间：{startdate}:{enddate}")
    print(f"总资金: {round(port_value, 2)}")
    print(f"净收益: {round(pnl, 2)}\n\n")
    # cerebro.plot(style='candlestick')  # 画图


if __name__ == '__main__':
    plt.switch_backend('agg')
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置画图时的中文显示
    plt.rcParams["axes.unicode_minus"] = False  # 设置画图时的负号显示
    codes = getCodes(getcodesfile())
    startdate = str(codes[0]).replace('\n', '')  # 回测开始时间
    enddate = str(codes[1]).replace('\n', '')  # 回测结束时间
    it = 0
    for code in codes:
        if it > 1:
            code = str(codes[it]).replace('\n', '')  # "sz300598"
            filepath = prepare_data(code, startdate, enddate)
            file_size = os.path.getsize(filepath)
            if file_size > 0:
                run_strategy(startdate, enddate, filepath)
            else:
                break
        it = it + 1

