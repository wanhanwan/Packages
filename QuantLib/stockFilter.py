# coding: utf-8
"""
股票过滤函数，将一些不符合条件的股票删除。
目前已经实现的函数是：
    1. 删除停牌、ST、一字涨跌停的股票(无法交易的股票)
    2. 删除上市不满一定天数的股票
"""


from FactorLib.data_source.base_data_source_h5 import data_source
from QuantLib.tools import df_rolling
import pandas as pd
import numpy as np


def _difference(stocksA, stocksB):
    """取在stocksA不在stocksB中的股票"""
    diff_index = stocksA.index.difference(stocksB.index)
    new_stocks = pd.DataFrame([1]*len(diff_index), index=diff_index, columns=stocksA.columns)
    return new_stocks


def _union(stockA, stockB):
    """取stockA和stockB的合集"""
    union_index = stockA.index.union(stockB.index)
    new_stocks = pd.DataFrame([1]*len(union_index), index=union_index, columns=stockA.columns)
    return new_stocks


def _intersection(stockA, stockB):
    """取stockA和stockB的交集"""
    intersection_index = stockA.index.intersection(stockB.index)
    if isinstance(stockA, pd.DataFrame):
        new_stocks = pd.DataFrame([1]*len(intersection_index), index=intersection_index, columns=stockA.columns)
    else:
        new_stocks = pd.DataFrame([1]*len(intersection_index), index=intersection_index, columns=[stockA.name])
    return new_stocks


# 剔除ST
def _dropst(stocklist):
    alldates = stocklist.index.get_level_values(0).unique().tolist()
    st_stocks = data_source.sector.get_st(alldates)
    return _difference(stocklist, st_stocks)


# 剔除一字涨跌停
def _drop_updown_limit(stocklist):
    alldates = stocklist.index.get_level_values(0).unique().tolist()
    uplimit = data_source.sector.get_uplimit(alldates).to_frame()
    downlimit = data_source.sector.get_downlimit(alldates).to_frame()
    updown = _union(uplimit, downlimit)
    return _difference(stocklist, updown)


# 剔除停牌
def _drop_suspendtrading(stocklist, hold_days=0):
    alldates = stocklist.index.get_level_values(0).unique().tolist()
    if hold_days > 0:
        start = data_source.trade_calendar.tradeDayOffset(min(alldates), -hold_days)
        end = max(alldates).strftime("%Y%m%d")
        dates = data_source.trade_calendar.get_trade_days(start, end)
        suspend = data_source.sector.get_suspend(dates).unstack().dropna(axis=1, how='all').fillna(1)
        suspend = df_rolling(suspend, hold_days, np.nanmin, axis=0).stack()
        suspend = suspend[suspend == 0]
    else:
        suspend = data_source.sector.get_suspend(alldates)
    return _difference(stocklist, suspend)
drop_suspendtrading = _drop_suspendtrading


# 返回股票列表中停牌的股票
def suspendtrading(stocklist, date):
    suspend = data_source.sector.get_suspend([date]).index.get_level_values(1).tolist()
    return list(set(stocklist).intersection(set(suspend)))


# 删除上市不满一定天数的股票
def _drop_newstocks(stocklist, months=12):
    all_dates = stocklist.index.get_level_values(0).unique().tolist()
    qualified_stocks = data_source.sector.get_ashare_onlist(all_dates, months_filter=months)
    return _intersection(stocklist, qualified_stocks)


# 删除摘掉ST不满一定天数股票
def _drop_latest_st(stocklist, months=6):
    all_dates = stocklist.index.get_level_values(0).unique().tolist()
    unst = data_source.sector.get_latest_unst(all_dates, months=months)
    return _difference(stocklist, unst)


def typical(stocklist):
    """
    剔除如下股票：
        1. st
        2. 最近10日内停牌
        3. 上市不满6个月
        4. 一字涨跌停
    :param stocklist:
    :return:
    """
    from functools import partial
    __funclist = [_dropst, partial(_drop_suspendtrading, hold_days=10),
                  partial(_drop_newstocks, months=6), _drop_updown_limit]
    for func in __funclist:
        stocklist = func(stocklist)
    return stocklist

def drop_st_suspend(stocklist):
    """
    剔除如下股票：
        1. st
        2. 最近10日内停牌
    :param stocklist:
    :return:
    """
    from functools import partial
    __funclist = [_dropst, partial(_drop_suspendtrading, hold_days=10)]
    for func in __funclist:
        stocklist = func(stocklist)
    return stocklist


def typical_add_latest_st(stocklist, st_months):
    return _drop_latest_st(typical(stocklist), st_months)


def realtime_trade_limit(stocklist):
    """实时行情限制，剔除当日停牌和涨跌停的股票"""
    from FactorLib.data_source.wind_plugin import realtime_quote
    stock_ids = stocklist.index.get_level_values(1).tolist()
    date = stocklist.index.get_level_values(0).unique().tolist()
    data = realtime_quote(['rt_last', 'rt_susp_flag', 'rt_high_limit', 'rt_low_limit'], ids=stock_ids)
    limit_stocks = data.query("rt_susp_flag%10 !=0.0 or abs(rt_last-rt_high_limit)<0.02 or abs(rt_last-rt_low_limit)<0.02")
    limit_stocks.index = pd.MultiIndex.from_product([date, limit_stocks.index], names=['date', 'IDs'])
    return _difference(stocklist, limit_stocks)


def realtime_typical(stocklist):
    """
    剔除如下股票：
        1. st
        2. 最近10日内停牌
        3. 上市不满6个月
        4. 涨跌停
    :param stocklist:
    :return:
    """
    from functools import partial
    __funclist = [_dropst, partial(_drop_suspendtrading, hold_days=10),
                  partial(_drop_newstocks, months=6), realtime_trade_limit]
    for func in __funclist:
        stocklist = func(stocklist)
    return stocklist


def realtime_typical2(stocklist):
    """
    剔除如下股票：
        1. st
        2. 最近10日内停牌
        4. 涨跌停
    :param stocklist:
    :return:
    """
    from functools import partial
    __funclist = [_dropst, partial(_drop_suspendtrading, hold_days=10), realtime_trade_limit]
    for func in __funclist:
        stocklist = func(stocklist)
    return stocklist


def drop_false_growth(data, upper_limit=3.0, upper_type='v', use_data=None):
    """
    如果增长率数值大于阈值，并且最近一年发生过并购
    行为，那么就剔除这些股票
    """
    from FactorLib.data_source.wind_financial_data_api import incomesheet
    all_dates = data.index.get_level_values('date').unique()
    merge = data_source.sector.get_index_members('merge_acc', dates=all_dates)
    if use_data is None:
        filter_data = incomesheet.load_incr_tb('净利润(不含少数股东损益)', n=1, dates=list(all_dates.strftime('%Y%m%d')))
    else:
        filter_data = use_data.reindex(data.index)

    if upper_type == 'q':
        limit = filter_data.groupby('date').quantile(1-upper_limit)
        limit.columns = ['limit']
        filter_data = filter_data.join(limit)
    else:
        filter_data['limit'] = upper_limit

    to_drop = (filter_data.iloc[:, 0] > filter_data.iloc[:, 1]) & (merge['merge_acc'] == 1)
    return data[~data.index.isin(to_drop[to_drop == 1].index)]

