"""更新盈利类因子"""
import pandas as pd
import numpy as np
from FactorLib.data_source.wind_financial_data_api import balancesheet, incomesheet
from FactorLib.data_source.tsl_data_source import TSLDBOnline


# ROE ttm
def roe_ttm(start, end, **kwargs):
    net_profit = incomesheet.load_ttm('净利润(不含少数股东损益)', start, end)
    net_asset = balancesheet.load_latest_period('股东权益合计(不含少数股东权益)', start, end)
    roe = net_asset.iloc[:, 0] / net_profit.iloc[:, 0]
    roe = roe.to_frame('roe_ttm')
    kwargs['data_source'].h5DB.save_factor(roe, '/stock_profit/')


# 净利润TTM， 包含业绩快报数据
def net_profit_incl_express(start, end, **kwargs):
    tsl = TSLDBOnline()
    datasource = kwargs['data_source']
    dates = datasource.trade_calendar.get_trade_days(start, end, '1d')
    data = tsl.load_netprofit_ttm_incl_express(dates=dates)
    datasource.h5DB.save_factor(data, '/stock_profit/')


# 扣非净利润
def net_profit_deduct_nonprofit(start, end, **kwargs):
    tsl = TSLDBOnline()
    datasource = kwargs['data_source']
    dates = datasource.trade_calendar.get_trade_days(start, end, '1d')
    data = tsl.load_latest_year('扣除非经常性损益后的净利润', dates=dates)
    datasource.h5DB.save_factor(data, '/stock_profit/')


# 上年年报扣非净利润
def net_profit_deduct_nonprofit_ly1(start, end, **kwargs):
    tsl = TSLDBOnline()
    datasource = kwargs['data_source']
    dates = datasource.trade_calendar.get_trade_days(start, end, '1d')
    data = tsl.load_latest_year('扣除非经常性损益后的净利润', n=1, dates=dates)
    data.columns = ['net_profit_deduct_nonprofit_ly1']
    datasource.h5DB.save_factor(data, '/stock_profit/')


ProfitFuncListDaily = [roe_ttm, net_profit_deduct_nonprofit, net_profit_incl_express, net_profit_deduct_nonprofit_ly1]


if __name__ == '__main__':
    from FactorLib.data_source.base_data_source_h5 import data_source
    # bp('20170630', '20170904', data_source=data_source)
    # epttm_divide_median('20070101', '20180109', data_source=data_source)
    net_profit_deduct_nonprofit_ly1('20180101', '20180117', data_source=data_source)