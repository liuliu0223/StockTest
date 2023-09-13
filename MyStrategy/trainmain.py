#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import datetime
import os
import backtrader as bt
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

import MyStrategy.PrepairData as prd
import MyStrategy.PreTreat as PT
import MyStrategy.XGBoostModelTest as xgbt
import MyStrategy.XgboostModel as xgbm

WIN_ENV_FLAG = False
RUNDNUM = 720
time_windows = 3
FILEDIR = "stocks"
TRAIN_DIR = "train"
STOCK_INFO_FILE = "text.txt"


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
    persent_stock = 0.0
    for code in codes:
        if it > 1:
            code = str(codes[it]).replace('\n', '')  # "sz300598"
            filepath = prd.prepare_data(code, startdate, enddate)
            # 加载数据，进行数据处理和分析
            df = prd.load_data(filepath)

            if df is None:
                print("There is no correct file in stocks!!!")
                break
            else:
                all_data_set = prd.pure_data(df, None)
                pre_deal = PT.PreTreadData(all_data_set)  # 数据预处理，清理空值

                data_set_process = pre_deal.series_to_supervised(all_data_set, time_windows)  # 取近time_windows天的数据，平移数据
                train_file = pre_deal.create_trainfile(code, data_set_process)
                print("file的title信息：" + train_file)
                delta = xgbt.xgb_train(data_set_process, RUNDNUM, time_windows)    # 归一化、混淆并训练数据
                result = pre_deal.create_trainfile((code + "_pred"), delta)
        it += 1