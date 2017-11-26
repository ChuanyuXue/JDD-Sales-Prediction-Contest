import numpy as np
import pandas as pd
import datetime

order = pd.read_csv('t_order.csv')
order['ord_dt'] = pd.to_datetime(order['ord_dt'])

#时间滑窗
date_list = ['2016/6/30','2016/7/31','2016/8/31','2016/9/30','2016/10/31','2016/11/30','2016/12/31','2017/1/31','2017/2/28','2017/3/31','2017/4/30']
delta = datetime.timedelta(days=30)
date_list = pd.to_datetime(date_list)



#特征提取
feature = pd.DataFrame()
for d in date_list:
    time_interval = order[(d - delta < order['ord_dt']) & (order['ord_dt'] <= d)]
    #sale_amt
    sale_amt = time_interval.groupby('shop_id', as_index=False).agg({'sale_amt':['sum','mean','median','max','min', 'std']})
    sale_amt.columns = ['shop_id','sale_amt_sum','sale_amt_mean','sale_amt_median','sale_amt_max', 'sale_amt_min','sale_amt_std']
    #offer_amt
    offer_amt = time_interval.groupby('shop_id', as_index=False).agg({'offer_amt':['sum', 'mean', 'median', 'max', 'min', 'std']})
    offer_amt.columns = ['shop_id', 'offer_amt_sum', 'offer_amt_mean', 'offer_amt_median', 'offer_amt_max','offer_amt_min', 'offer_amt_std']
    all_feature = pd.merge(sale_amt,offer_amt,on='shop_id')
    #offer_cnt
    offer_cnt = time_interval.groupby('shop_id', as_index=False).agg({'offer_cnt':['sum', 'mean', 'median', 'max', 'min', 'std']})
    offer_cnt.columns = ['shop_id', 'offer_cnt_sum', 'offer_cnt_mean', 'offer_cnt_median', 'offer_cnt_max','offer_cnt_min', 'offer_cnt_std']
    all_feature = pd.merge(all_feature, offer_cnt,on='shop_id')
    #rtn_cnt
    rtn_cnt = time_interval.groupby('shop_id', as_index=False).agg({'rtn_cnt':['sum', 'mean', 'median', 'max', 'min', 'std']})
    rtn_cnt.columns = ['shop_id', 'rtn_cnt_sum', 'rtn_cnt_mean', 'rtn_cnt_median', 'rtn_cnt_max','rtn_cnt_min', 'rtn_cnt_std']
    all_feature = pd.merge(all_feature, rtn_cnt, on='shop_id')
    #ord_cnt
    ord_cnt = time_interval.groupby('shop_id', as_index=False).agg({'ord_cnt':['sum', 'mean', 'median', 'max', 'min', 'std']})
    ord_cnt.columns = ['shop_id', 'ord_cnt_sum', 'ord_cnt_mean', 'ord_cnt_median', 'ord_cnt_max', 'ord_cnt_min','ord_cnt_std']
    all_feature = pd.merge(all_feature, ord_cnt, on='shop_id')
    #usr_cnt
    user_cnt = time_interval.groupby('shop_id', as_index=False).agg({'user_cnt':['sum', 'mean', 'median', 'max', 'min', 'std']})
    user_cnt.columns = ['shop_id', 'user_cnt_sum', 'user_cnt_mean', 'user_cnt_median', 'user_cnt_max', 'user_cnt_min','user_cnt_std']
    all_feature = pd.merge(all_feature, user_cnt, on='shop_id')
    all_feature['dt'] = d
    feature = pd.concat([feature, all_feature])
    print(d)
feature.to_csv('feature_all.csv', index=False)
print('finish')

