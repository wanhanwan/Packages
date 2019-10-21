# coding: utf-8
import pandas as pd
import numpy as np

from FactorLib.data_source.base_data_source_h5 import data_source, tc
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.performance import factor_information_coefficient


def cal_ic_by_alphalens(factor_data, prices=None, group_by=None, periods=(20,), **kwargs):
    """调用alphalens计算因子IC
    """
    factor_data = factor_data.copy()
    if isinstance(factor_data, pd.DataFrame):
        factor_data = factor_data.iloc[:, 0]
    factor_data.index.names = ['date', 'asset']

    if prices is None:
        start = factor_data.index.get_level_values('date').min()
        start = tc.tradeDayOffset(start, -5)
        end = factor_data.index.get_level_values('date').max()
        end = tc.tradeDayOffset(end, max(periods))
        prices = data_source.load_factor('adj_close', '/stocks/', start_date=start,
                                         end_date=end)['adj_close'].unstack()
    elif isinstance(prices, pd.DataFrame):
        if prices.index.nlevels == 2:
            prices = prices.iloc[:, 0].unstack()
    else:
        raise ValueError('prices 格式非法!')
    
    merge_data = get_clean_factor_and_forward_returns(factor_data, prices,
                                                      group_by, periods=periods, **kwargs)
    by_group = group_by is not None
    ic = factor_information_coefficient(merge_data, group_adjust=False, by_group=by_group)
    return ic


def cal_factor_group_return(factor_data, periods=(20,), prices=None, group_by=None,
    quantiles=5, freq='1d', **kwargs):
    """基于alphalens计算因子收益率"""
    stocklist = pd.DataFrame(np.ones(len(factor_data)), index=factor_data.index, columns=['stocklist'])
    stocklist = stockFilter.typical(stocklist)
    factor_data = factor_data.reindex(stocklist.index)

    start = factor_data.index.get_level_values('date').min()
    start = tc.tradeDayOffset(start, -5)
    end = factor_data.index.get_level_values('date').max()
    end = tc.tradeDayOffset(end, max(periods)+1, freq=freq)
    if prices is None:
        prices = data_source.load_factor('adj_close', '/stocks/', start_date=start,
                                         end_date=end)['adj_close'].unstack()
    elif isinstance(prices, pd.DataFrame):
        if prices.index.nlevels == 2:
            prices = prices.iloc[:, 0].unstack()
    else:
        raise ValueError('prices 格式非法!')
    if freq != '1d':
        date_index = tc.get_trade_days(start, end, freq, retstr=None)
        prices = prices.reindex(date_index, copy=False)
    if_groupby = group_by is not None
    merge_data = get_clean_factor_and_forward_returns(factor_data, prices, group_by,
                                                      periods=periods, binning_by_group=if_groupby,
                                                      **kwargs)
    return merge_data


if __name__ == '__main__':
    bp_div_median = data_source.load_factor('bp_divide_median', '/stock_value/')
    ic = cal_ic(bp_div_median,  factor_name='bp_divide_median', rank=True, stock_validation='typical')

