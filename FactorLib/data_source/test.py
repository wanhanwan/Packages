# encoding: utf-8
# author: wanhanwan
from FactorLib.data_source.base_data_source_h5 import h5_2, fund
from FactorLib.data_source.h5db import H5DB
from pathlib import Path


# data = h5.load_factor('StyleFactor_VG', '/XYData/StyleFactor/')
# h5.save_factor2(data['StyleFactor_VG'], '/temp/')

a=fund.get_fund_by_stock('603160', period='20191231', min_ratio=0.02)
a.sort_values('stk_mkv_ratio', ascending=False, inplace=True)
