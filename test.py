import pandas as pd
from FactorLib.data_source.base_data_source_h5 import data_source
from FactorLib.data_source.ncdb import NCDB
from FactorLib.data_source.h5db import H5DB
from FactorLib.riskmodel.riskmodel_data_source import RiskDataSource

nc = NCDB(r'D:\data\risk_model\xy')
h5 = H5DB(r'D:\data\risk_model\xy')
# data = nc.load_factor('risk_factor', '/factorData/')
#
# h5.save_h5file(data, '/factorData/', name='risk_factor')

# dates = data_source.trade_calendar.get_trade_days('20100101', '20180630', '1m')
# ds = RiskDataSource('xy')
# data = ds.load_factors(factor_names=['Size'], dates=dates)

h5.read_h5file_attr('risk_factor', '/factorData/', 'factors')
