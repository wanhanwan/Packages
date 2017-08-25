import pandas as pd
from . import stockFilter

from FactorLib.data_source.base_data_source_h5 import data_source
from FactorLib.data_source.tseries import date_shift

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
    offset_of_start_dt = data_source.trade_calendar.tradeDayOffset(start_dt, -1, window)
    end_dt = new_factor.index.get_level_values(0).max()
    offset_of_end_dt = data_source.trade_calendar.tradeDayOffset(end_dt, 1, window)
    ids = new_factor.index.get_level_values(1).unique().tolist()

    ret = data_source.get_fix_period_return(ids, freq=window, start_date=offset_of_start_dt, end_date=offset_of_end_dt)
    future_ret = date_shift(ret, -1).rename(columns={'daily_returns_%':'future_ret'})
    new_factor = pd.concat([new_factor, future_ret], axis=1, join='inner')

    ic = new_factor.groupby(level=0).apply(corr, rank=rank)
    if retstocknums:
        return ic, new_factor.groupby(level=0)[factor_name].count()
    return ic


if __name__ == '__main__':
    bp_div_median = data_source.load_factor('bp_divide_median', '/stock_value/')
    ic = cal_ic(bp_div_median,  factor_name='bp_divide_median', rank=True, stock_validation='typical')

