# encoding: utf-8
# author: wanhanwan
from FactorLib.data_source.base_data_source_h5 import h5
from FactorLib.data_source.h5db import H5DB
from pathlib import Path


# data = h5.load_factor('StyleFactor_VG', '/XYData/StyleFactor/')
# h5.save_factor2(data['StyleFactor_VG'], '/temp/')


# 读取
# ids = ['000001', '000002', '600000']
dates = ['20181018', '20190315']
# tmp = h5.load_factor2('StyleFactor_VG', '/temp/', ids=ids, dates=dates)
# print(tmp)

h5_2 = H5DB("D:/data/factors")
# for d, f in h5.walk('/XYData/'):
#     data = h5.load_factor(f, d)
#     h5_2.save_factor2(data[f], d, if_exists='replace')
#     print([d, f])

# import pandas as pd
# for file_pth in Path("D:/data/factors/growth").glob(pattern='*.h5'):
#     data = pd.read_hdf(file_pth, key='data')
#     date_name = None
#     IDs_name = None
#     if 'ticker' in data.columns:
#         IDs_name = 'ticker'
#     elif 'IDs' in data.columns:
#         IDs_name = 'IDs'
#     if 'date' in data.columns:
#         date_name = 'date'
#     elif 'tradeDate' in data.columns:
#         date_name = 'tradeDate'
#     data = pd.pivot_table(data, values=file_pth.stem, index=date_name, columns=IDs_name)
#     data.rename_axis(index='date', columns='IDs', inplace=True)
#     data.name = file_pth.stem
#     h5_2.save_factor2(data, '/growth/tmp/', if_exists='replace')
data = h5_2.read_h5file('ROE', '/Profitability/', check_A='IDs')
last_5 = h5_2.fin_data_loader.last_n_periods_with_consistend_report_dt(data, 'ROE', dates, n=4)
print(last_5)
