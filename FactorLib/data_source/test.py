# encoding: utf-8
# author: wanhanwan
from FactorLib.data_source.base_data_source_h5 import h5_2, fund, bcolz_db
from FactorLib.data_source.h5db import H5DB
from pathlib import Path


# data = h5_2.load_factor2('StyleFactor_VG', '/XYData/StyleFactor/', stack=True,dates=['20200306'])
# bdb = BcolzDB("D:/data/factors")
# bdb.save_factor(data, '/bcolz/', 'test')


bcolz_db.load_factor('daily_indicator', '/opt/', idx_name='code',idx_start='10000984',idx_end='10000984')


from FactorLib.data_source.rqdata import rq_data, id_convert
from FactorLib.utils.tool_funcs import date_to_str
rq_data.get_factor(id_convert('000001'),'dividend_yield_ttm',start_date='2020-03-01', end_date='2020-03-25')