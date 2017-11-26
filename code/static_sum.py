import numpy as np
import pandas as pd
import datetime as dt

sales_sum_all = pd.read_csv('t_sales_sum.csv')
sales_sum_all['dt'] = pd.to_datetime(sales_sum_all['dt'])

date_list = ['2016/6/30','2016/7/31','2016/8/31','2016/9/30','2016/10/31','2016/11/30','2016/12/31','2017/1/31','2017/2/28','2017/3/31','2017/4/30']
date_list = pd.to_datetime(date_list)

feature = pd.DataFrame()

#CV
for i in range(3, len(date_list)):
    d = date_list[i]
    sales_sum = sales_sum_all[sales_sum_all['dt'] <= date_list[i - 3]]
    #提取历史sum信息，不考虑时间序列
    history_sum_mean = sales_sum.groupby('shop_id', as_index=False)['sale_amt_3m'].mean()
    history_sum_median = sales_sum.groupby('shop_id', as_index=False)['sale_amt_3m'].median()
    history_sum_max = sales_sum.groupby('shop_id', as_index=False)['sale_amt_3m'].max()
    history_sum_min = sales_sum.groupby('shop_id', as_index=False)['sale_amt_3m'].min()
    history_sum = pd.merge(history_sum_mean,history_sum_median, on='shop_id')
    history_sum = pd.merge(history_sum, history_sum_max, on='shop_id')
    history_sum = pd.merge(history_sum, history_sum_min, on='shop_id')
    history_sum.columns = ['shop_id', 'history_sum_mean', 'history_sum_median', 'history_sum_max', 'history_sum_min']
    history_sum['dt'] = d
    print(d)
    feature = pd.concat([feature, history_sum])
#提取order里面的static信息


#合并输出
# label_train = label_train[['shop_id','sale_amt_3m']].groupby('shop_id', as_index = False).mean()
# label_train.columns = ['shop_id', 'label']
# train = pd.merge(history_sum, label_train, on='shop_id')
# test = history_sum

#写文件
# train.to_csv('online_train.csv',index=False)
# test.to_csv('online_test.csv', index=False)

feature.to_csv('static_feature.csv', index=False)