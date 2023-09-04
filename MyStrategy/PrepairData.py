#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import datetime
import os
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


WIN_ENV_FLAG = False
FILEDIR = "stocks"
STOCK_INFO_FILE = "text.txt"


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


def get_codes(name):
    file = None
    try:
        path = get_work_path("") + name
        print(path + '\n')
        file = open(path, 'r')
        return file.readlines()
    finally:
        if file is not None:
            file.close()


# 转换文件中的代码信息，查找对应代码的内容
def get_sh_stock(s_code):
    s_code = s_code[2:]
    df = ak.stock_individual_info_em(symbol=s_code)
    return df


# 从接口中读取指定日期的数据，并存在制定路径的文件里
def prepare_data(f_code, f_startdate, f_enddate):
    csv_file = get_work_path(FILEDIR) + f_code + ".csv"
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


def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=True, index_col="date")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", utc=True)
    return df


#数据转换为对应的类型，判断转换的数据是否存在缺失值，并进行删除
def pure_date(df):
    print(f"df.info:{df.info()}")
    '''
    data = pd.DataFrame(index=df.index)
    data.columns = df.columns
    iter = 0
    while iter < len(df.columns):
        old_list = df.values[iter]
        if df.columns[iter] == 'date':
            new_typelist = list(map(datetime.date, old_list))
            data.values['date'] = new_typelist
        else:
            new_typelist = list(map(float, old_list))
            data.values[iter] = new_typelist
        iter += 1
'''
    # how = 'all', 只有当前行都是缺失值才删除
    # how = 'any', 只要当前行有一个缺失值就删除
    print(f"check the datas, there is null: \n {df.isnull().sum()}")
    data = df.isnull().sum()
    iter = 0
    while iter < len(data.values):
        if data.values[iter] > 0:
            df.dropna(axis=0, how='any')  # axis=0 删除全是缺失值的行；axis=1，删除全是缺失值的列
        iter += 1
    print(data)
    print(f"check the datas, there is null:{df.isnull().sum()}")
    return df

'''
wine = load_wine()
data = wine.data  # 数据
lables = wine.target  # 标签
feaures = wine.feature_names
df = pd.DataFrame(data, columns=feaures)  # 原始数据
'''


# 第一步：无量纲化，数据归一化，min-max处理
def standard_data(df):
    """
    df : 原始数据
    return : data 标准化的数据
    """
    data = pd.DataFrame(index=df.index)  # 列名，一个新的dataframe
    columns = df.columns.tolist()  # 将列名提取出来
    for col in columns:
        d = df[col]
        max = d.max()
        min = d.min()
        mean = d.mean()
        data[col] = ((d - mean) / (max - min)).tolist()
    #print(data)
    return data


#  某一列当做参照序列，其他为对比序列
def gra_one(f_data, m=0):
    """
    return:
    """
    columns = f_data.columns.tolist()  # 将列名提取出来
    # 第一步：无量纲化
    data = standard_data(f_data)
    reference_Seq = data.iloc[:, m]  # 参考序列
    data.drop(columns[m], axis=1, inplace=True)  # 删除参考列
    compare_Seq = data.iloc[:, 0:]  # 对比序列
    row, col = compare_Seq.shape
    # 第二步：参考序列 - 对比序列
    data_sub = np.zeros([row, col])
    for i in range(col):
        for j in range(row):
            data_sub[j, i] = abs(reference_Seq[j] - compare_Seq.iloc[j, i])
    # 找出最大值和最小值
    maxVal = np.max(data_sub)
    minVal = np.min(data_sub)
    cisi = np.zeros([row, col])
    for i in range(row):
        for j in range(col):
            cisi[i, j] = (minVal + 0.5 * maxVal) / (data_sub[i, j] + 0.5 * maxVal)
    # 第三步：计算关联度
    result = [np.mean(cisi[:, i]) for i in range(col)]
    result.insert(m, 1)  # 参照列为1
    return pd.DataFrame(result)


def gra(data):
    df = data.copy()
    columns = [str(s) for s in df.columns if s not in [None]]  # [1 2 ,,,12]
    # print(columns)
    df_local = pd.DataFrame(columns=columns)
    df.columns = columns
    for i in range(len(df.columns)):  # 每一列都做参照序列，求关联系数
        df_local.iloc[:, i] = gra_one(df, m=i)[0]
    df_local.index = columns
    return df_local


# 热力图展示
def ShowGRAHeatMap(DataFrame):
    # colormap = plt.cm.hsv
    # ylabels = DataFrame.columns.values.tolist()
    f, ax = plt.subplots(figsize=(15, 15))
    ax.set_title('STOCK GRA')
    # 设置展示一半，如果不需要注释掉mask即可
    mask = np.zeros_like(DataFrame)
    mask[np.triu_indices_from(mask)] = True  # np.triu_indices 上三角矩阵

    with sns.axes_style("white"):
        sns.heatmap(DataFrame, cmap="YlGnBu", annot=True, mask=mask,)
    plt.show()


if __name__ == '__main__':
    codes = get_codes(STOCK_INFO_FILE)
    startdate = str(codes[0]).replace('\n', '')  # 回测开始时间
    enddate = str(codes[1]).replace('\n', '')  # 回测结束时间
    it = 0
    code = ""
    filepath = ""
    for code in codes:
        if it > 1:
            code = str(codes[it]).replace('\n', '')  # "sz300598"
            filepath = prepare_data(code, startdate, enddate)
        it += 1
    df = load_data(filepath)
    pure_date(df)
    standard_data(df)
    data_stock_gra = gra(df)
    ShowGRAHeatMap(data_stock_gra)
