"""更新盈利类因子"""
import pandas as pd
import numpy as np
from FactorLib.data_source.wind_financial_data_api import balancesheet, incomesheet


# ROE ttm
def roe_ttm(start, end, **kwargs):
    net_profit = incomesheet.load_ttm('净利润(不含少数股东损益)', start, end)
    net_asset = balancesheet.load_latest_period('股东权益合计(不含少数股东权益)', start, end)
    roe = net_asset.iloc[:, 0] / net_profit.iloc[:, 0]
    roe = roe.to_frame('roe_ttm')
    kwargs['data_source'].h5DB.save_factor(roe, '/stock_profit/')


ProfitFuncListDaily = [roe_ttm]


if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source
    # bp('20170630', '20170904', data_source=data_source)
    # epttm_divide_median('20070101', '20180109', data_source=data_source)
    roe_ttm('20160101', '20180115', data_source=data_source)