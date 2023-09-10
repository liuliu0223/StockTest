#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, LSTM
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from matplotlib import pyplot as plt
import numpy as np
from xgboost import plot_importance, plot_tree


def xgb_train(data, rundnum, time_windows):
    data_set_process = data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_set_process)

    train_size = int(len(data_set_process)*0.8)
    test_size = len(data_set_process) - train_size
    train_XGB, test_XGB = scaled_data[0:train_size, :], scaled_data[train_size:len(data_set_process), :]

    train_XGB_X, train_XGB_Y = \
        train_XGB[:, :(len(data_set_process.columns)-1)], train_XGB[:, (len(data_set_process.columns)-1)]
    test_XGB_X, test_XGB_Y = \
        test_XGB[:, :(len(data_set_process.columns)-1)], test_XGB[:, (len(data_set_process.columns)-1)]

    # 算法参数
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.1,  # 0.1
        'max_depth': 2,  # 5
        'lambda': 3,   # 3
        'subsample': 0.9,  # 0.7
        'colsample_bytree': 0.9,  # 0.7
        'min_child_weight': 5,  # 3
        'slient': 1,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.05,  # 0.1
        'seed': 1000,  # 1000
        'nthread': 2,
    }

    #生成数据集格式
    xgb_train = xgb.DMatrix(train_XGB_X, label=train_XGB_Y)
    xgb_test = xgb.DMatrix(test_XGB_X, label=test_XGB_Y)
    num_rounds = rundnum
    watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]

    #xgboost模型训练
    model_xgb = xgb.train(params, xgb_train, num_rounds, watchlist)

    #对测试集进行预测
    y_pred_xgb = model_xgb.predict(xgb_test)

    print(f"预测值:\n{y_pred_xgb}")
    print(f"max:{y_pred_xgb.max()}, min: {y_pred_xgb.min()}")
    # 模型结果可视化及评估
    plt.plot(test_XGB_Y, color='red', label='Real Price for Test set')
    plt.plot(y_pred_xgb, color='blue', label='Predicted Price for Test set')
    plt.title(f'Close Price Prediction for Test set: {time_windows}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    mape_xgb = np.mean(np.abs(y_pred_xgb-test_XGB_Y)/test_XGB_Y)*100
    print('XGBoostModelTest.py ----- XGBoost平均误差率为：{}%'.format(mape_xgb))  #平均误差率为1.1974%
    mape_xgb2 = float(np.mean(np.abs(y_pred_xgb-test_XGB_Y)/test_XGB_Y))

    # 反归一化处理
 #   arr = np.array(y_pred_xgb)
    tmp_data = pd.DataFrame(y_pred_xgb)
    tmp_data1 = pd.DataFrame(test_XGB)
    tmp_data1.drop(tmp_data1.columns[len(tmp_data1.columns)-1], axis=1, inplace=True)
    tmp_data1 = pd.concat([tmp_data1, tmp_data], axis=1)  # 把预测值列拼接到最后一列，为close(t+1)

    pred_data = scaler.inverse_transform(tmp_data1)
    print(f"预测值:\n{pred_data}")
    print(f"max:{pred_data.max()}, min: {pred_data.min()}")
    return (pred_data)
