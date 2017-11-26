import numpy as np
import pandas as pd
import datetime

comment = pd.read_csv('t_comment.csv')
comment['create_dt'] = pd.to_datetime(comment['create_dt'])

#时间滑窗
date_list = ['2016/6/30','2016/7/31','2016/8/31','2016/9/30','2016/10/31','2016/11/30','2016/12/31','2017/1/31','2017/2/28','2017/3/31','2017/4/30']
delta = datetime.timedelta(days=30)
date_list = pd.to_datetime(date_list)

#特征提取
feature = pd.DataFrame()
for d in date_list:
    time_interval = comment[(d - delta < comment['create_dt']) & (comment['create_dt'] <= d)]
    #bad_amt
    bad_num = time_interval.groupby('shop_id', as_index=False).agg({'bad_num':['sum']})
    bad_num.columns = ['shop_id', 'bad_num_sum']
    #cmmt_num
    cmmt_num = time_interval.groupby('shop_id', as_index=False).agg({'cmmt_num': ['sum']})
    cmmt_num.columns = ['shop_id', 'cmmt_num_sum']
    all_feature = pd.merge(bad_num, cmmt_num, on='shop_id')
    #dis_num
    dis_num= time_interval.groupby('shop_id', as_index=False).agg({'dis_num': ['sum']})
    dis_num.columns = ['shop_id', 'dis_num_sum']
    all_feature = pd.merge(all_feature, dis_num, on='shop_id')
    #good_num
    good_num = time_interval.groupby('shop_id', as_index=False).agg({'good_num': ['sum']})
    good_num.columns = ['shop_id', 'good_num_sum']
    all_feature = pd.merge(all_feature, good_num, on='shop_id')
    #mid_num
    mid_num = time_interval.groupby('shop_id', as_index=False).agg({'mid_num': ['sum']})
    mid_num.columns = ['shop_id', 'mid_num_sum']
    all_feature = pd.merge(all_feature, mid_num, on='shop_id')
    all_feature.rename(columns={'create_dt':'dt'}, inplace=True)
    all_feature['dt'] = d
    feature = pd.concat([feature, all_feature])
    print(d)

#输出
feature.to_csv('comment_feature_all.csv', index=False)
