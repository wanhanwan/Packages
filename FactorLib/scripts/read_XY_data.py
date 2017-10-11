# -*- coding: utf-8 -*-

"""从兴业因子数据中读取因子，保存成h5格式"""

from FactorLib.data_source.base_data_source_h5 import h5
import pandas as pd
import os
import re

trgdirs = ['Value', 'Technical', 'Sentiment', 'Quality', 'Others', 'Momentum', 'Growth', 'StyleFactor']
root = r'G:\data\xydata_history_zip\XYData20170929'
srcdirs = [x for x in os.listdir(root) if x not in ['基础数据', 'ElementaryFactor']]


def parsetrg(src):
    if src == 'ValueBiasFactor':
        return 'Others'
    for x in trgdirs:
        if re.search(x, src):
            return x
    return 'Others'


def trgfile(src, rootdir):
    if re.match(rootdir+'-', src) is None:
        return '_'.join([rootdir, src])
    return src.replace('-', '_')


for d in srcdirs:
    print(d)
    xy_path = root + '/' + d + '/'
    target = parsetrg(d)

    # 读取数据
    all_files=os.listdir(xy_path)
    for file in all_files:
        data = pd.read_csv(os.path.join(xy_path, file), header=0, index_col=0, parse_dates=True)
        data.columns = data.columns.str[:6]
        trgname = trgfile(file[:-4], d)
        data = data.stack().to_frame(trgname).rename_axis(['date', 'IDs'])
        h5.save_factor(data, '/XYData/%s/' % target)
