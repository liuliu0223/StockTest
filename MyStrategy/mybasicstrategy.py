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
RUNDNUM = 720
time_windows = 10
FILEDIR = "stocks"
TRAIN_DIR = "train"
STOCK_INFO_FILE = "text.txt"


if __name__ == '__main__':
    # get the code data from websit
    codes = prd.get_codes(STOCK_INFO_FILE)
    startdate = str(codes[0]).replace('\n', '')  # 回测开始时间
    enddate = str(codes[1]).replace('\n', '')  # 回测结束时间
    it = 0
    code = ""
    filepath = ""
    train_file = ""
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
                xgbt.xgb_train(data_set_process, RUNDNUM, time_windows)
        it += 1