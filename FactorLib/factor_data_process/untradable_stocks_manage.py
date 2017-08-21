from QuantLib import stockFilter
import pandas as pd


def drop_untradable_stocks(factor, **kwargs):
    """去掉不可以交易的股票"""
    old_data = factor.data
    stocklist = pd.DataFrame([[1]*len(factor)], columns=['a'], index=factor.index)
    new_stocks = stockFilter.typical(stocklist)
    factor.data = old_data.reindex(new_stocks.index)


def drop_untradable_new_latestst(factor, **kwargs):
    """去掉上市不满6个月的新股以及涨跌停、复牌、恢复ST"""
    old_data = factor.data
    stocklist = pd.DataFrame([[1]*len(factor)], columns=['a'], index=factor.index)
    new_stocks = stockFilter.typical_add_latest_st(stocklist, months=3)
    factor.data = old_data.reindex(new_stocks.index)


FuncList = {'drop_untradable_stocks': drop_untradable_stocks,
            'drop_new_and_untradable': drop_untradable_new_latestst}