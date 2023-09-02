#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import datetime
import os
import backtrader as bt
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

    # 初始化函数，初始化属性、指标的计算，only once time
    def __init__(self):
        self.data_close = self.datas[0].close  # close data
        # initial data
        self.order = None
        self.pnl = None  # profit
        self.sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        self.sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(self.sma1, self.sma2)  # crossover signal

    # order statement information
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 检查订单是否完成
        if order.status in [order.Completed]:
            buy_price = order.executed.price
            buy_comm = order.executed.comm
            size = self.getposition(self.data).size
            cost = order.executed.value
            fund = self.broker.getvalue()
            if order.isbuy():
                txt = 'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, fund Value %.2f, pos Size %.2f' % \
                      (buy_price, cost, buy_comm, fund, size)
                self.log(txt)
            elif order.issell():
                txt = 'SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, fund Value %.2f, pos Size %.2f' % \
                      (buy_price, cost, buy_comm, fund, size)
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
        self.log('business profit: %.2f, Net profit: %.2f' % (trade.pnl, trade.pnlcomm))

    #  loop in every business day
    def next(self):
        size = self.getposition(self.data).size
        cost = self.broker.getcash()
        fund = self.broker.getvalue()
        if not self.position:  # Outside, buy
            if self.crossover > 0:  # if golden cross, valid=datetime.datetime.now() + datetime.timedelta(days=3)
                self.log('Available Cash: %.2f, Total fund: %.2f' % (cost, fund))
                self.order = self.buy()
                self.log('Outside, golden cross buy, close: %.2f，Total fund：%.2f' %
                         (self.data_close[0], fund))
        else:
            if self.crossover > 0:
                if self.cash_valid > 0:
                    self.log('Available Cash: %.2f, Total fund: %.2f' % (cost, fund))
                    self.order = self.buy()
                    self.log('Outside, golden cross buy, close: %.2f' % self.data_close[0])
            else:  # Inside and dead cross
                self.order = self.close(size=size)
                self.log('Inside dead cross, sell, close:  %.2f，Total fund：%.2f' %
                         (self.data_close[0], fund))
