import xgboost
import operator
import numpy as np
import pandas as pd

train = pd.read_csv('underline_train_1126.csv')
test = pd.read_csv('underline_test_1126.csv')

# train = train.drop('dt',axis = 1)
# test = test.drop('dt', axis = 1)

print(len(train))
print(len(test))

feature = [x for x in train.columns if x not in ['label','shop_id','dt']]

#手动zscore
for col in train[feature].columns:
    train[col] = (train[col] - train[col].mean()) / train[col].std(ddof=0)
for col in test[feature].columns:
    test[col] = (test[col] - test[col].mean()) / test[col].std(ddof=0)



xgbTrain = xgboost.DMatrix(train[feature], label=train['label'])
xgbVal = xgboost.DMatrix(test[feature], label=test['label'])
watchlist = [(xgbTrain, 'train'),(xgbVal, 'eval')]

def mapeObj(preds, dtrain):
    gaps = dtrain.get_label()
    grad = np.sign(preds - gaps) / gaps
    hess = 1 / np.abs(preds - gaps)
    return grad, hess

def huber_approx_obj(preds, dtrain):
    d = preds - dtrain.get_label() # remove .get_labels() for sklearn
    h = 10000000  # h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess

def log_cosh_obj(preds, dtrain):
    x = preds - dtrain.get_label()
    grad = np.tanh(x)
    hess = 1 / np.cosh(x)**2
    return grad, hess

def fair_obj(preds, dtrain):
    """y = c * abs(x) - c * np.log(abs(abs(x) + c))"""
    x = preds - dtrain.get_label()
    c = 100000
    den = abs(x) + c
    grad = (c * x) / den
    hess = (c * c) / (den * den)
    return grad, hess

def wmaeEval(preds, dtrain):
    label = dtrain.get_label()
    return 'error', np.sum(np.abs(preds - label)) / np.sum(label)

param = {}
param['eta'] = 0.01
param['max_depth'] = 2
param['mmin_child_weight'] = 4
param['subsample'] = 0.8
param['colsample_bytree'] = 0.3
param['silent'] = 1


num_round = 10000
modle = xgboost.train(param, xgbTrain, num_round, watchlist, obj=fair_obj,feval=wmaeEval, early_stopping_rounds=100)
importance = modle.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.to_csv('特征重要性.csv',index=False)