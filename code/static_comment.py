import numpy as np
import pandas as pd
import datetime as dt

commentall = pd.read_csv('t_comment.csv')
commentall['create_dt'] = pd.to_datetime(commentall['create_dt'])

date_list = ['2016/6/30','2016/7/31','2016/8/31','2016/9/30','2016/10/31','2016/11/30','2016/12/31','2017/1/31','2017/2/28','2017/3/31','2017/4/30']
date_list = pd.to_datetime(date_list)
feature = pd.DataFrame()
for d in date_list:
    comment = commentall[commentall['create_dt'] <= d]
    #bad_amt
    bad_num = comment.groupby('shop_id', as_index=False).agg({'bad_num':['sum','mean','median','max']})
    bad_num.columns = ['shop_id', 'bad_num_sum', 'bad_num_mean', 'bad_num_median', 'bad_num_max']
    #cmmt_num
    cmmt_num = comment.groupby('shop_id', as_index=False).agg({'cmmt_num': ['sum', 'mean', 'median', 'max']})
    cmmt_num.columns = ['shop_id', 'cmmt_num_sum', 'cmmt_num_mean', 'cmmt_num_median', 'cmmt_num_max']
    all_feature = pd.merge(bad_num, cmmt_num, on='shop_id')
    #dis_num
    dis_num= comment.groupby('shop_id', as_index=False).agg({'dis_num': ['sum', 'mean', 'median', 'max']})
    dis_num.columns = ['shop_id', 'dis_num_sum', 'dis_num_mean', 'dis_num_median', 'dis_num_max']
    all_feature = pd.merge(all_feature, dis_num, on='shop_id')
    #good_num
    good_num = comment.groupby('shop_id', as_index=False).agg({'good_num': ['sum', 'mean', 'median', 'max']})
    good_num.columns = ['shop_id', 'good_num_sum', 'good_num_mean', 'good_num_median', 'good_num_max']
    all_feature = pd.merge(all_feature, good_num, on='shop_id')
    #mid_num
    mid_num = comment.groupby('shop_id', as_index=False).agg({'mid_num': ['sum', 'mean', 'median', 'max']})
    mid_num.columns = ['shop_id', 'mid_num_sum', 'mid_num_mean', 'mid_num_median', 'mid_num_max']
    all_feature = pd.merge(all_feature, mid_num, on='shop_id')

    all_feature['dt'] = d
    feature = pd.concat([feature, all_feature])
    print(d)

#重命名列名
for i in feature.columns[1:]:
    feature.rename(columns={i:i+'_history'}, inplace=True)
feature.rename(columns={'dt_history':'dt', 'shop_id_history':'shop_id'}, inplace=True)
#输出
feature.to_csv('static_comment.csv', index=False)