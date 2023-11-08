#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, LSTM
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from matplotlib import pyplot as plt
import numpy as np


# prarams:
#         roundnum:迭代次数
#         time_windows:平移窗口数
#         size_rate:训练数据占比
def xgb_train(data, roundnum, time_windows=3, train_size_rate=0.8):
    data_set_process = data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_set_process)

    train_size = int(len(data_set_process)*train_size_rate)
    test_size = len(data_set_process) - train_size
    train_XGB, test_XGB = scaled_data[0:train_size, :], scaled_data[train_size:len(data_set_process), :]

    train_XGB_X, train_XGB_Y = \
        train_XGB[:, :(len(data_set_process.columns)-1)], train_XGB[:, (len(data_set_process.columns)-1)]
    test_XGB_X, test_XGB_Y = \
        test_XGB[:, :(len(data_set_process.columns)-1)], test_XGB[:, (len(data_set_process.columns)-1)]

    # 算法参数:sh688031（星环），20221001
    # (trainsize 0.97, XGBoost平均误差率为：5.056196451187134%)
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.05,  # 0.05 ，指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 3,  # 4，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 0.1,   # 0，[0, 0.1, 0.5, 1]
        'subsample': 0.7,  # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.5,  # 0.9
        'min_child_weight': 5,  # 5，推荐的候选值为：[1, 3, 5, 7]
        'slient': 0,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.05,  # 0.05 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 3000,  # 1000
        'nthread': 4,
    }

    # 生成数据集格式
    xgb_train = xgb.DMatrix(train_XGB_X, label=train_XGB_Y)
    xgb_test = xgb.DMatrix(test_XGB_X, label=test_XGB_Y)
    num_rounds = roundnum
    watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]

    # xgboost模型训练
    model_xgb = xgb.train(params, xgb_train, num_rounds, watchlist)

    # 对测试集进行预测
    y_pred_xgb = model_xgb.predict(xgb_test)

    # 模型结果可视化及评估
    plt.plot(test_XGB_Y, color='red', label='Real Price for Test set')
    plt.plot(y_pred_xgb, color='blue', label='Predicted Price for Test set')
    plt.title(f'Close Price Prediction for Test set: {time_windows}')
    plt.xlabel('Times')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    # 反归一化处理
    tmp_pred = pd.DataFrame(y_pred_xgb)
    tmp_test = pd.DataFrame(test_XGB)
    tmp_test.drop(tmp_test.columns[len(tmp_test.columns)-1], axis=1, inplace=True)
    tmp_test = pd.concat([tmp_test, tmp_pred], axis=1)  # 把预测值列拼接到最后一列，为close(t+1)

    transform_data = scaler.inverse_transform(tmp_test)
    pred_data = pd.DataFrame(transform_data)
    pred_data.drop(pred_data.columns[: len(pred_data.columns)-1], axis=1, inplace=True)
    max_pred = 0
    min_pred = 0
    std_pred = 0
    pred_list = pred_data.values.tolist()
    if pred_list is not None:
        max_pred = np.max(pred_list, axis=1)
        min_pred = np.min(pred_list, axis=1)
        std_pred = np.std(pred_list, axis=1)

    # 时间列表，与测试数据集相匹配
    date_list = pd.DataFrame(data).T.columns[train_size:len(data_set_process)]
    date_list = pd.DataFrame(date_list)
    pred_data = pd.concat([date_list, pred_data], axis=1)
    print(f"预测值:\n{pred_data}")

    # 打印参数值
    print(f"XGBboostModel.params: \n'booster': %s, 'objective': %s, 'gamma': %.2f, 'max_depth': %d, 'lambda': %.2f, "
          f"'subsample': %.2f, 'colsample_bytree': %.2f, 'min_child_weight': %d, 'slient': %d, 'eta': %.2f, "
          f"'seed': %d, 'nthread': %d" % (params['booster'], params['objective'], params['gamma'], params['max_depth'],
                                          params['lambda'], params['subsample'], params['colsample_bytree'],
                                          params['min_child_weight'], params['slient'], params['eta'], params['seed'],
                                          params['nthread']))

    mape_xgb = np.mean(np.abs(y_pred_xgb-test_XGB_Y)/test_XGB_Y)*100
    print('XGBoostModelTest.py ----- XGBoost平均误差率为：{}%'.format(mape_xgb))  # 平均误差率为1.1974%
    print(f'XGBoostModelTest.py ----- max:%.2f, min:%.2f, std:%.2f' % (max_pred, min_pred, std_pred))
    plt.close()
    return (pred_data)