#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import pandas as pd
import MyStrategy.PrepairData as prd
import os

WIN_ENV_FLAG = False
FILEDIR = "stocks"
TRAIN_DIR = "train"
STOCK_INFO_FILE = "text.txt"


class PreTreadData:

    def __init__(self, data):
        data_set = data
        print("PreTread Class init:")

    # 依据特征重要性，选择low high open来进行预测close
    # 数据选择t-n, ...., t-2 t-1 与 t 来预测未来 t+1
    # 转换原始数据为新的特征列来进行预测,time_window可以用来调试用前几次的数据来预测

    def series_to_supervised(self, data, time_window=3):
        data_columns = ['open', 'high', 'low', 'close']   # 特征值属性名称，可以通过特征值判断传入列表
        data = data[data_columns]  # Note this is important to the important feature choice
        cols, names = list(), list()

        for i in range(time_window, -1, -1):
            # get the data
            cols.append(data.shift(i))  # 数据偏移量

            # get the column name
            if (i - 1) <= 0:
                suffix = '(t+%d)' % abs(i - 1)
            else:
                suffix = '(t-%d)' % (i - 1)
            names += [(colname + suffix) for colname in data_columns]

        # concat the cols into one dataframe
        # 数据按列拼接axis=1， axis=0（按行拼接）
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg.index = data.index.copy()
        # remove the nan value which is caused by pandas.shift
        agg = agg.dropna(inplace=False)
        # remove unused col (only keep the "close" fied for the t+1 period)
        # Note col "close" place in the columns
        len_ = len(data_columns) * time_window
        col_numbers_drop = []
        for i in range(len(data_columns) - 1):
            col_numbers_drop.append(len_ + i)

        agg.drop(agg.columns[col_numbers_drop], axis=1, inplace=True)  # inplace=True用替换后的数据

        return agg

    def create_trainfile(self, code, data_datafram):
        train_file = os.path.join(prd.get_work_path(TRAIN_DIR), f"{code}.csv")
        file = open(train_file, 'w', encoding='utf-8')
        data_datafram.to_csv(file, encoding='utf-8')
        #print(data_datafram.columns.values)
        #print(data_datafram.info())
        file.close()
        return train_file


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
                pre_deal = PreTreadData(p_df)
                all_data_set = p_df.copy()  # date is index and already setted in func load_data()
                data_set_process = pre_deal.series_to_supervised(all_data_set, 30)  # 取近30天的数据
                train_file = pre_deal.create_trainfile(code, data_set_process)
                print("file的title信息：" + train_file)
        it += 1


'''
    len_ = len(['open', 'high', 'low', 'close']) * 3
    col_numbers_drop = []
    for i in range(3):
        col_numbers_drop.append(len_ + i)
    print(col_numbers_drop)
'''

