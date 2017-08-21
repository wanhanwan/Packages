# -*- coding: utf-8 -*-

"""从兴业因子数据中读取因子，保存成h5格式"""

from data_source import h5
import pandas as pd
import os
root = 'D:/data/XYData20170731/XYData'
dirs = [x for x in os.listdir(root) if x not in ['基础数据']]
for d in dirs:
    print(d)
    xy_path = root + '/' + d + '/'

    # 读取数据
    all_files=os.listdir(xy_path)
    for file in all_files:
        data = pd.read_csv(os.path.join(xy_path, file), header=0, index_col=0, parse_dates=True)
        data.columns = data.columns.str[:6]
        data = data.stack().to_frame().rename_axis(['date', 'IDs']). \
            rename_axis({0: file[:-4].replace('-', '_')}, axis=1)
        h5.save_factor(data, xy_path[22:])
