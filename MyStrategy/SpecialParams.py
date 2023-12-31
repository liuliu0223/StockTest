#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

# 算法参数
# sh600741
def get_params(code):
    # 算法参数:sz002475（立讯）,20110101
    # time_windows=15，trainsize=0.97 XGBoost平均误差率为：1.9045542925596237%
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.15,  # 0.2 ，指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 3,  # 11，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 0,   # 0.5，[0, 0.1, 0.5, 1]
        'subsample': 0.9,  # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.8,  # 0.9
        'min_child_weight': 3,  # 5，推荐的候选值为：[1, 3, 5, 7]
        'slient': 1,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.05,  # 0.05 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 2500,  # 2500
        'nthread': 4,
    }

    # 算法参数:sh600741（华域），20120101，XGBoost平均误差率为：1.3984233140945435%，train size=0.97
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.06,  # 0.1 ，指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 3,  # 5，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 3,   # 3，[0, 0.1, 0.5, 1]
        'subsample': 0.9,  # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.9,  # 0.7
        'min_child_weight': 5,  # 3，推荐的候选值为：[1, 3, 5, 7]
        'slient': 1,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.05,  # 0.1 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 3000,  # 1000
        'nthread': 4,
    }

    # 算法参数:sh600602(云赛），20000101， XGBoost平均误差率为：3.516016900539398%
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.05,  # 0.06， 指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 5,  # 3，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 0,   # 3，[0, 0.1, 0.5, 1]
        'subsample': 0.9,  # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.8,  # 0.9
        'min_child_weight': 1,  # 5，推荐的候选值为：[1, 3, 5, 7]
        'slient': 1,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.1,  # 0.1 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 2500,  # 2500
        'nthread': 4,
    }

    # 去掉异常值
    # 算法参数:sh600602(云赛），20230601， XGBoost平均误差率为：10.529667139053345%, train size=0.95
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.05,     # 0.05, 指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 4,    # 5，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 1,       # 3，[0, 0.1, 0.5, 1]
        'subsample': 1,    # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.7,  # 0.9
        'min_child_weight': 1,  # 1，推荐的候选值为：[1, 3, 5, 7]
        'slient': 0,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.1,       # 0.1 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 2500,     # 2500
        'nthread': 4,
    }

    # 算法参数:sz002456（欧菲光），20110101， XGBoost平均误差率为：4.176051542162895%
    # (trainsize 0.96, XGBoost平均误差率为：2.2908413782715797%)
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0,  # 0.05 ，指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 4,  # 4，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 0,   # 4，[0, 0.1, 0.5, 1]
        'subsample': 0.9,  # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.9,  # 0.9
        'min_child_weight': 5,  # 5，推荐的候选值为：[1, 3, 5, 7]
        'slient': 0,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.05,  # 0.05 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 2500,  # 1000
        'nthread': 4,
    }
    # 算法参数:sz002363（隆基），20110101，XGBoost平均误差率为：XGBoost平均误差率为：5.596379190683365%
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.07,  # 0.05 ，指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 4,  # 4，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 4,   # 4，[0, 0.1, 0.5, 1]
        'subsample': 0.6,  # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.9,  # 0.9
        'min_child_weight': 0,  # 5，推荐的候选值为：[1, 3, 5, 7]
        'slient': 0,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.08,  # 0.05 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 2500,  # 1000
        'nthread': 4,
    }

    # 算法参数:sz002895(川恒），20180101， XGBoost平均误差率为：3.1654126942157745%，train size=0.85
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.04,  # 0.06， 指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 3,  # 5，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 0,   # 3，[0, 0.1, 0.5, 1]
        'subsample': 0.9,  # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.8,  # 0.9
        'min_child_weight': 1,  # 5，推荐的候选值为：[1, 3, 5, 7]
        'slient': 1,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.1,  # 0.1 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 2500,  # 2500
        'nthread': 4,
    }

    # 算法参数:sz300750(宁德），20190101， XGBoost平均误差率为：1.849798858165741%
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.05,     # 0.05, 指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 4,    # 5，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 1,       # 3，[0, 0.1, 0.5, 1]
        'subsample': 1,    # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.7,  # 0.9
        'min_child_weight': 1,  # 1，推荐的候选值为：[1, 3, 5, 7]
        'slient': 0,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.1,       # 0.1 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 2500,     # 2500
        'nthread': 4,
    }

    # 算法参数:sh688031（星环），20221001
    # (trainsize 0.97, XGBoost平均误差率为：5.056196451187134%)
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.05,  # 0.05 ，指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        'max_depth': 4,  # 4，指定树的最大深度，默认值为6，合理的设置可以防止过拟合。推荐的数值为：[3, 5, 6, 7, 9, 12, 15, 17, 25]
        'lambda': 0,   # 0，[0, 0.1, 0.5, 1]
        'subsample': 0.7,  # 0.9 ，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
        'colsample_bytree': 0.5,  # 0.9
        'min_child_weight': 5,  # 5，推荐的候选值为：[1, 3, 5, 7]
        'slient': 0,      # =1输出中间过程，=0不输出中间过程
        'eta': 0.067,  # 0.05 ，[0.01, 0.015, 0.025, 0.05, 0.1]
        'seed': 3000,  # 1000
        'nthread': 4,
    }

    return params