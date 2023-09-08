#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import datetime
import os
import backtrader as bt
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import PrepairData as prd
import PreTreat as PT
import XGBoostModelTest as xgbt

WIN_ENV_FLAG = True
RUNDNUM = 960
time_windows = 5
FILEDIR = "stocks"
TRAIN_DIR = "train"
STOCK_INFO_FILE = "text.txt"
SPECIALDATE = "2023-09-05"


if __name__ == '__main__':
    # get the code data from websit
    codes = prd.get_codes(STOCK_INFO_FILE)
    startdate = str(codes[0]).replace('\n', '')  # 回测开始时间
    enddate = str(codes[1]).replace('\n', '')  # 回测结束时间
    it = 0
    code = ""
    filepath = ""
    train_file = ""
    persent_stock = 0.0
    for code in codes:
        if it > 1:
            code = str(codes[it]).replace('\n', '')  # "sz300598"
            filepath = prd.prepare_data(code, startdate, enddate)
            # 加载数据，进行数据处理和分析
            df = prd.load_data(filepath)
            #  test stock file
            #    filepath = 'C:\\01 Work\\13 program\PyProject1.0\stocks\sz300589.csv'
            #    df = prd.load_data(filepath)
            if df is None:
                print("There is no correct file in stocks!!!")
                break
            else:
                p_df = prd.pure_data(df, None)
                pre_deal = PT.PreTreadData(p_df)
                all_data_set = p_df.copy()  # date is index and already setted in func load_data()
                data_set_process = pre_deal.series_to_supervised(all_data_set, time_windows)  # 取近30天的数据
                train_file = pre_deal.create_trainfile(code, data_set_process)
                print("file的title信息：" + train_file)
                persent_stock = float(xgbt.xgb_train(data_set_process, RUNDNUM, time_windows))

            # 获取stock原始数据，找到对应日期的价格
            df = pd.read_csv(filepath, parse_dates=True, index_col='date')
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d', utc=True)
            '''
            d_days = 1
            after_date = (datetime.datetime.strptime(SPECIALDATE, "%Y-%m-%d") +
                           datetime.timedelta(d_days)).strftime("%Y-%m-%d")
                           '''
            for it2 in df.index:
                if datetime.datetime.strftime(it2, "%Y-%m-%d") == SPECIALDATE:
                    real_price = float('%.2f' % df['close'][it2])
                    pre_deal_price1 = real_price * (1 + persent_stock)
                    pre_deal_price2 = real_price * (1 - persent_stock)
                    print(f"2023-09-07 REAL Close price is %.2f；\n2023-09-07 predict price1 is %.2f, pre_deal_price2: %.2f"
                          % (real_price, pre_deal_price1, pre_deal_price2))
            # 获取结束
        it += 1



