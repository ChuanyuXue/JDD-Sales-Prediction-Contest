import xgboost
import numpy as np
import pandas as pd
train = pd.read_csv('online_train_1126.csv')
test = pd.read_csv('online_test_1126.csv')

orig = ['history_sum_mean',	'history_sum_median', 'history_sum_max', 'history_sum_min']
train = train.drop('dt',axis = 1)
test = test.drop('dt', axis = 1)
feature = [x for x in train.columns if x not in ['label','shop_id']]

#手动zscore
for col in train[feature].columns:
    train[col] = (train[col] - train[col].mean()) / train[col].std(ddof=0)
for col in test[feature].columns:
    test[col] = (test[col] - test[col].mean()) / test[col].std(ddof=0)


xgbTrain = xgboost.DMatrix(train[feature], label=train['label'])
xgbVal = xgboost.DMatrix(test[feature])

def fair_obj(preds, dtrain):
    """y = c * abs(x) - c * np.log(abs(abs(x) + c))"""
    x = preds - dtrain.get_label()
    c = 100000
    den = abs(x) + c
    grad = (c * x) / den
    hess = (c * c) / (den * den)
    return grad, hess

param = {}
param['eta'] = 0.01
param['max_depth'] = 2
param['mmin_child_weight'] = 1
param['subsample'] = 0.8
param['colsample_bytree'] = 0.3
param['silent'] = 1

num_round = 123
modle = xgboost.train(param, xgbTrain, num_round,obj=fair_obj)
preds = modle.predict(xgbVal)

preds = pd.DataFrame(preds)
result = pd.concat([test['shop_id'], preds], axis=1)
result.to_csv('result1126_3.csv', index=False)