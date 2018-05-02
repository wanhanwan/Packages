# coding: utf-8
import pandas as pd
import numpy as np
from . import stockFilter

from FactorLib.data_source.base_data_source_h5 import data_source, tc
from FactorLib.data_source.tseries import move_dtindex
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.performance import factor_information_coefficient


# 计算因子的IC值
def cal_ic(factor_data, factor_name, window='1m', rank=False, stock_validation=None,
           retstocknums=False):
    """
    每一期的因子值与下一期的股票股票收益率做相关性检验(IC)

    :param factor_data:dataframe 因子数据

    :param factor_name: 因子名称

    :param window: offset, IC值的时间窗口

    :param rank: 若为True，返回RankIC

    :param stock_validation: str: 剔除非法股票, 支持stockFilter中定义的函数名

    :return: ic

    """
    def corr(data, rank):
        if rank:
            return data.corr(method='spearman').iloc[0, 1]
        else:
            return data.corr(method='pearson').iloc[0, 1]

    if stock_validation is not None:
        valid_idx = getattr(stockFilter, stock_validation)(factor_data)
        new_factor = factor_data[factor_name].reindex(valid_idx.index)
    else:
        new_factor = factor_data[factor_name]

    start_dt = new_factor.index.get_level_values(0).min()
    offset_of_start_dt = data_source.trade_calendar.tradeDayOffset(start_dt, 1, '1d')
    end_dt = new_factor.index.get_level_values(0).max()
    offset_of_end_dt = data_source.trade_calendar.tradeDayOffset(end_dt, 1, window)
    ids = new_factor.index.get_level_values(1).unique().tolist()

    max_data_of_ret = data_source.h5DB.get_date_range('daily_returns', '/stocks/')[1]
    if max_data_of_ret < offset_of_start_dt:
        if retstocknums:
            return None, 0
        return None
    ret = data_source.get_fix_period_return(ids, freq=window, start_date=offset_of_start_dt, end_date=offset_of_end_dt)
    future_ret = move_dtindex(ret, -1, window).rename(columns={'daily_returns_%':'future_ret'})
    new_factor = pd.concat([new_factor, future_ret], axis=1, join='inner')

    ic = new_factor.groupby(level=0).apply(corr, rank=rank)
    if retstocknums:
        return ic, new_factor.groupby(level=0)[factor_name].count()
    return ic


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

