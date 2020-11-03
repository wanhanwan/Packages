# encoding: utf-8
# date: 2020/8/7
# author: wanhanwan
# email: wanshuai_shufe@163.com

from FactorLib.data_source.base_data_source_h5 import h5_2

data = h5_2.load_factor2('adj_nav', '/fund/')
data.tail()

import pandas as pd
nav = pd.read_csv("~/tushare_fund_nav.csv")
nav['IDs'] = nav['IDs'].astype('str').str.zfill(6)
nav['date'] = pd.to_datetime(nav['date'])
nav.head()

writer = nav.set_index(['date', 'IDs']).sort_index()
writer.head()
h5_2.save_factor2(writer['adj_nav'], '/fund/')
h5_2.save_factor2(writer['unit_nav'], '/fund/')
