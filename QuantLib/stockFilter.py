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
    new_stocks = pd.DataFrame([1]*len(intersection_index), index=intersection_index, columns=stockA.columns)
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


def typical_add_latest_st(stocklist, st_months):
    return _drop_latest_st(typical(stocklist), st_months)

